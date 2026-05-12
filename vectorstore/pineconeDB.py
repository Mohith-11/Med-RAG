from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "rag-project"
INDEX_DIM   = 1024   # e5-large-v2 output dimension
INDEX_METRIC = "cosine"


def get_or_create_index():
    """Create index if it doesn't exist; connect to it if it does."""
    existing = [i.name for i in pc.list_indexes()]

    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=INDEX_DIM,
            metric=INDEX_METRIC,
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"✅ Created new Pinecone index '{index_name}' (dim={INDEX_DIM})")
    else:
        print(f"ℹ️  Using existing Pinecone index '{index_name}'")

    return pc.Index(index_name)


def reset_index():
    """Delete and recreate the index — call ONLY before a full re-ingest."""
    existing = [i.name for i in pc.list_indexes()]

    if index_name in existing:
        pc.delete_index(index_name)
        print(f"❌ Deleted old index '{index_name}'")

    pc.create_index(
        name=index_name,
        dimension=INDEX_DIM,
        metric=INDEX_METRIC,
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"✅ Recreated index '{index_name}' (dim={INDEX_DIM})")
    return pc.Index(index_name)


# Module-level index object — safe to import from multiple modules
index = get_or_create_index()


def upsert_chunks(chunks, vectors):
    batch = []

    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        batch.append({
            "id": f"{chunk['source']}_{chunk['page']}_{chunk['parent_id']}_{i}",
            "values": vector.tolist(),
            "metadata": {
                "text":   chunk["parent"],          # full parent chunk (retrieval context)
                "child":  chunk["text"],            # child chunk (used for embedding)
                "page":   chunk["page"],
                "source": chunk.get("source", "unknown"),
            }
        })

    index.upsert(vectors=batch)