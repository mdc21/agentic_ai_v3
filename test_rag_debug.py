import os
from app.rag_client import RAGClient
from app.session_cache import SessionCache

# Mock cache
cache = SessionCache()
rag = RAGClient(cache)

query = "what is the process to surrender life policy"
print(f"Testing RAG for: '{query}'")

res = rag.query(query, session_id="test_debug")

print(f"\nAnswerable: {res.answerable}")
print(f"Number of chunks: {len(res.chunks)}")

for i, chunk in enumerate(res.chunks):
    print(f"\n--- Result {i+1} ---")
    print(f"ID: {chunk.chunk_id}")
    print(f"Score: {chunk.score:.4f}")
    print(f"Text: {chunk.text[:200]}...")
