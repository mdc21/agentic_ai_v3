"""
app/ingest_utils.py — Utilities for ingesting FAQ data into the vector DB from within the application.
Used for real-time Knowledge Base synchronization.
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict

logger = logging.getLogger(__name__)

def sync_pension_faqs_to_pinecone(faq_file_path: str):
    """
    Reads the pension_faqs.txt JSON file, generates embeddings, and upserts to Pinecone.
    Designed to run within the Streamlit process context.
    """
    from sentence_transformers import SentenceTransformer
    from pinecone import Pinecone
    
    # 1. Load Data
    if not os.path.exists(faq_file_path):
        raise FileNotFoundError(f"FAQ file not found: {faq_file_path}")
        
    with open(faq_file_path, 'r') as f:
        data = json.load(f)
        
    faqs = data.get("faqs", [])
    if not faqs:
        return "No FAQs found to sync."

    # 2. Setup Pinecone & Model
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    if not api_key or not index_name:
        raise ValueError("Pinecone credentials missing in environment (.env)")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    model = SentenceTransformer(model_name)

    # 3. Preparation
    now = datetime.now(timezone.utc).isoformat()
    all_chunks = []
    
    for faq in faqs:
        q = faq.get("question", "")
        a = faq.get("answer", "")
        tags = ", ".join(faq.get("context_tags", []))
        
        # Consistent format with standard ingestion
        combo_text = f"Question: {q}\nAnswer: {a}\nContext Tags: {tags}"
        
        # Chunk ID matches the format in ingest_to_pinecone.py for overwriting consistency
        source_name = os.path.basename(faq_file_path)
        chunk_id = hashlib.sha256(f"{source_name}:{faq.get('id')}".encode()).hexdigest()[:16]
        
        all_chunks.append({
            "id": chunk_id,
            "text": combo_text,
            "category": faq.get("category", "General"),
            "product_type": "pension" # Default for this file
        })

    # 4. Ingest in batches
    batch_size = 50
    total_synced = 0
    
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        
        texts = [b["text"] for b in batch]
        embeddings = model.encode(texts)
        
        vectors = []
        for item, emb in zip(batch, embeddings):
            vectors.append({
                "id": item["id"],
                "values": emb.tolist(),
                "metadata": {
                    "text": item["text"],
                    "source_doc": os.path.basename(faq_file_path),
                    "section": item["category"],
                    "product_type": item["product_type"],
                    "last_updated": now
                }
            })
            
        index.upsert(vectors=vectors)
        total_synced += len(vectors)
        
    return f"Successfully synced {total_synced} entries to Pinecone Cloud."
