import os
import sys

os.environ["VECTOR_DB_BACKEND"] = "chroma"
os.environ["VECTOR_DB_COLLECTION"] = "faq_docs"
os.environ["EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"

from pathlib import Path
root = Path("/Users/shilpadhall/agentic_ai_projects/agentic_ai_v3")
sys.path.append(str(root))

from app.rag_client import RAGClient
client = RAGClient()
res = client.query("what tax is applicable to pension encashment?")
print(f"Query: {res.query}")
print(f"Answerable: {res.answerable}")
for i, chunk in enumerate(res.chunks):
    print(f"--- Chunk {i+1} ---")
    print(f"ID: {chunk.chunk_id}")
    print(f"Score: {chunk.score:.4f}")
    print(f"Text: {chunk.text[:100]}...")
