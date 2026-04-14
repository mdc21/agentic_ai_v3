import os
from pinecone import Pinecone
from dotenv import load_dotenv

def clear_index():
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")

    if not api_key or not index_name:
        print("Error: PINECONE_API_KEY or PINECONE_INDEX_NAME missing in .env")
        return

    print(f"Connecting to Pinecone index: {index_name}")
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    print(f"Deleting all vectors in index '{index_name}'...")
    try:
        index.delete(delete_all=True)
        print("Success! Index cleared.")
    except Exception as e:
        print(f"Error clearing index: {e}")

if __name__ == "__main__":
    clear_index()
