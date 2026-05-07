from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("rag-project")


def query_index(vector, top_k=5):
    results = index.query(
        vector=vector.tolist(),
        top_k=top_k,  
        include_metadata=True
    )
    return results["matches"]