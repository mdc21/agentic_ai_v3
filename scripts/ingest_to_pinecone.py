"""
scripts/ingest_to_pinecone.py — ingest FAQ documents specifically for Pinecone Cloud.

Usage:
  python scripts/ingest_to_pinecone.py --dir docs/faq/
"""

import argparse
import hashlib
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Reuse chunking logic from main ingest script
try:
    from ingest_faq import chunk_text, load_meta, read_document, extract_sections
except ImportError:
    # Inline fallback if script is run in isolation
    def chunk_text(text: str, chunk_size=500, overlap=50) -> list[str]:
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
            except ImportError: pass
        return {}

    def read_document(path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="ignore")

    def extract_sections(text: str) -> list[tuple[str, str]]:
        sections = []
        pattern  = re.compile(r"^#{1,3}\s+(.+)$|^([A-Z][A-Z\s]{5,})$", re.MULTILINE)
        matches  = list(pattern.finditer(text))
        for i, m in enumerate(matches):
            title   = (m.group(1) or m.group(2) or "").strip()
            start   = m.end()
            end     = matches[i+1].start() if i+1 < len(matches) else len(text)
            content = text[start:end].strip()
            if content: sections.append((title, content))
        if not sections: sections = [("General", text)]
        return sections

def ingest_pinecone(chunks_with_meta: list[dict]):
    from pinecone import Pinecone
    from sentence_transformers import SentenceTransformer
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    if not api_key or not index_name:
        print("Error: PINECONE_API_KEY and PINECONE_INDEX_NAME must be set in .env")
        sys.exit(1)

    print(f"Connecting to Pinecone index: {index_name}")
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Generating embeddings for {len(chunks_with_meta)} chunks...")
    texts = [c["text"] for c in chunks_with_meta]
    embeddings = model.encode(texts, show_progress_bar=True)

    vectors_to_upsert = []
    for c, emb in zip(chunks_with_meta, embeddings):
        # Pinecone metadata must be a flat dictionary of strings/numbers/bools/lists of strings
        metadata = {
            "text": c["text"],
            "source_doc": c.get("source_doc", "unknown"),
            "section": c.get("section", "unknown"),
            "product_type": c.get("product_type") or "all",
            "heritage_brand": c.get("heritage_brand") or "all",
            "last_updated": c.get("last_updated", "")
        }
        vectors_to_upsert.append({
            "id": c["chunk_id"],
            "values": emb.tolist(),
            "metadata": metadata
        })

    print(f"Upserting to Pinecone...")
    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i : i + batch_size]
        index.upsert(vectors=batch)
        print(f"  Upserted batch {i//batch_size + 1}")

    print("Success! Ingestion complete.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Directory containing FAQ documents")
    args = parser.parse_args()

    doc_dir = Path(args.dir)
    now = datetime.now(timezone.utc).isoformat()
    all_chunks = []

    for path in sorted(doc_dir.rglob("*")):
        if path.suffix not in (".txt", ".md", ".json") or path.stem.endswith(".meta"):
            continue
        
        print(f"Processing: {path}")
        meta = load_meta(path)
        
        # Default text/markdown handling (read first to check content)
        text = read_document(path)
        if not text.strip(): continue

        if path.suffix == ".json" or text.strip().startswith("{"):
            # Handle JSON FAQ format
            import json
            try:
                data = json.loads(text)
                faqs = data.get("faqs", [])
                for faq in faqs:
                    q = faq.get("question", "")
                    a = faq.get("answer", "")
                    tags = ", ".join(faq.get("context_tags", []))
                    combined_text = f"Question: {q}\nAnswer: {a}\nContext Tags: {tags}"
                    
                    chunk_id = hashlib.sha256(
                        f"{path.name}:{faq.get('id')}".encode()
                    ).hexdigest()[:16]
                    
                    all_chunks.append({
                        "chunk_id":      chunk_id,
                        "text":          combined_text,
                        "source_doc":    path.name,
                        "section":       faq.get("category", "General"),
                        "product_type":  meta.get("product_type", "all"),
                        "heritage_brand":meta.get("heritage_brand", "all"),
                        "last_updated":  now,
                    })
                print(f"  Extracted {len(faqs)} granular FAQ entries from JSON.")
                continue
            except Exception as e:
                print(f"  Error parsing JSON {path}: {e}")
                continue

        # Default text/markdown handling
        text = read_document(path)
        if not text.strip(): continue

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

    if not all_chunks:
        print("No chunks generated.")
        return

    ingest_pinecone(all_chunks)

if __name__ == "__main__":
    main()
