"""
scripts/ingest_faq.py — chunk, embed, and load FAQ documents into the vector DB.

Usage:
  python scripts/ingest_faq.py --dir docs/faq/ --backend chroma
  python scripts/ingest_faq.py --dir docs/faq/ --backend pgvector

Supported document formats: .txt, .md, .pdf (requires pdfplumber)

Each chunk is stored with metadata:
  source_doc, section, product_type, heritage_brand, last_updated

Set product_type and heritage_brand in a .meta.yaml file alongside each document:
  docs/faq/pension_guide.txt
  docs/faq/pension_guide.meta.yaml   ← { product_type: pension, heritage_brand: Brand_A }
"""

import argparse
import hashlib
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

CHUNK_SIZE    = 500    # tokens (approximate — using word count as proxy)
CHUNK_OVERLAP = 50


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def load_meta(doc_path: Path) -> dict:
    meta_path = doc_path.with_suffix(".meta.yaml")
    if meta_path.exists():
        try:
            import yaml
            with open(meta_path) as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            pass
    return {}


def read_document(path: Path) -> str:
    if path.suffix == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        except ImportError:
            print(f"  pdfplumber not installed — skipping {path}")
            return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_sections(text: str) -> list[tuple[str, str]]:
    """Split text on markdown headings or lines in ALL CAPS as section markers."""
    sections = []
    pattern  = re.compile(r"^#{1,3}\s+(.+)$|^([A-Z][A-Z\s]{5,})$", re.MULTILINE)
    matches  = list(pattern.finditer(text))
    for i, m in enumerate(matches):
        title   = (m.group(1) or m.group(2) or "").strip()
        start   = m.end()
        end     = matches[i+1].start() if i+1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            sections.append((title, content))
    if not sections:
        sections = [("General", text)]
    return sections


def ingest_chroma(chunks_with_meta: list[dict]) -> None:
    import chromadb
    from chromadb.utils import embedding_functions

    ef     = embedding_functions.SentenceTransformerEmbeddingFunction(
                 model_name=os.getenv("EMBEDDING_MODEL","all-MiniLM-L6-v2"))
    client = chromadb.HttpClient(host=os.getenv("VECTOR_DB_URL","localhost"))
    coll   = client.get_or_create_collection(
                 os.getenv("VECTOR_DB_COLLECTION","faq_docs"), embedding_function=ef)

    ids, docs, metas = [], [], []
    for c in chunks_with_meta:
        ids.append(c["chunk_id"])
        docs.append(c["text"])
        metas.append({k: v for k,v in c.items() if k not in ("chunk_id","text") and v is not None})

    batch = 100
    for i in range(0, len(ids), batch):
        coll.upsert(ids=ids[i:i+batch], documents=docs[i:i+batch], metadatas=metas[i:i+batch])
        print(f"  Upserted batch {i//batch + 1} ({len(ids[i:i+batch])} chunks)")


def ingest_pgvector(chunks_with_meta: list[dict]) -> None:
    import psycopg2
    from sentence_transformers import SentenceTransformer
    model  = SentenceTransformer(os.getenv("EMBEDDING_MODEL","all-MiniLM-L6-v2"))
    conn   = psycopg2.connect(os.environ["VECTOR_DB_URL"])
    table  = os.getenv("VECTOR_DB_COLLECTION","faq_chunks")

    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                chunk_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                embedding vector(384),
                source_doc TEXT, section TEXT,
                product_type TEXT, heritage_brand TEXT, last_updated TEXT
            )
        """)
        conn.commit()

        texts = [c["text"] for c in chunks_with_meta]
        embeddings = model.encode(texts, show_progress_bar=True)
        for c, emb in zip(chunks_with_meta, embeddings):
            cur.execute(f"""
                INSERT INTO {table} (chunk_id, content, embedding, source_doc, section,
                    product_type, heritage_brand, last_updated)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    content=EXCLUDED.content, embedding=EXCLUDED.embedding
            """, (c["chunk_id"], c["text"], emb.tolist(), c.get("source_doc"),
                  c.get("section"), c.get("product_type"), c.get("heritage_brand"),
                  c.get("last_updated")))
        conn.commit()
    print(f"  Ingested {len(chunks_with_meta)} chunks into {table}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",     required=True, help="Directory containing FAQ documents")
    parser.add_argument("--backend", default="chroma", choices=["chroma","pgvector"])
    args = parser.parse_args()

    doc_dir = Path(args.dir)
    now     = datetime.now(timezone.utc).isoformat()
    all_chunks = []

    for path in sorted(doc_dir.rglob("*")):
        if path.suffix not in (".txt",".md",".pdf") or path.stem.endswith(".meta"):
            continue
        print(f"Processing: {path}")
        meta = load_meta(path)
        text = read_document(path)
        if not text.strip():
            continue

        # Attempt to parse as JSON first (to handle structured FAQ databases cleanly)
        is_json = False
        try:
            import json
            data = json.loads(text)
            if isinstance(data, dict) and "faqs" in data:
                is_json = True
                for faq in data["faqs"]:
                    qid = faq.get("id", str(len(all_chunks)))
                    q = faq.get("question", "")
                    a = faq.get("answer", "")
                    tags = ", ".join(faq.get("context_tags", []))
                    
                    # Create clean natural language representation of the FAQ for superior vector matching
                    combo_text = f"Question: {q}\nAnswer: {a}\nContext Tags: {tags}"
                    chunk_id = hashlib.sha256(f"{path.name}:{qid}".encode()).hexdigest()[:16]
                    
                    all_chunks.append({
                        "chunk_id":      chunk_id,
                        "text":          combo_text,
                        "source_doc":    path.name,
                        "section":       faq.get("category", "General FAQ"),
                        "product_type":  meta.get("product_type"),
                        "heritage_brand":meta.get("heritage_brand"),
                        "last_updated":  now,
                    })
        except Exception:
            pass  # Fallback to standard text extraction if it's not valid JSON

        if is_json:
            continue

        # Fallback for standard .txt, .md, .pdf files
        for section_title, section_text in extract_sections(text):
            for i, chunk_text_content in enumerate(chunk_text(section_text)):
                chunk_id = hashlib.sha256(
                    f"{path.name}:{section_title}:{i}".encode()
                ).hexdigest()[:16]
                all_chunks.append({
                    "chunk_id":      chunk_id,
                    "text":          chunk_text_content,
                    "source_doc":    path.name,
                    "section":       section_title,
                    "product_type":  meta.get("product_type"),
                    "heritage_brand":meta.get("heritage_brand"),
                    "last_updated":  now,
                })

    print(f"\nTotal chunks: {len(all_chunks)}")
    if not all_chunks:
        print("No chunks generated. Check that document files exist in --dir.")
        return

    if args.backend == "chroma":
        ingest_chroma(all_chunks)
    else:
        ingest_pgvector(all_chunks)

    print("Done.")


if __name__ == "__main__":
    main()
