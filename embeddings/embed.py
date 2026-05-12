import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# Switched to e5-small-v2: naturally outputs 384-dim to match Pinecone index
model = SentenceTransformer("intfloat/e5-small-v2")

# e5-small output dimension
EMBED_DIM = 384


# ✅ Embed passages (documents)
def embed_passages(texts):
    texts = [f"passage: {t}" for t in texts]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32
    )

    # 🔥 MRL truncation
    embeddings = embeddings[:, :EMBED_DIM]

    return embeddings


# ✅ Embed query (user question)
def embed_query(query):
    embedding = model.encode(
        [f"query: {query}"],
        normalize_embeddings=True
    )

    # 🔥 MRL truncation
    embedding = embedding[:, :EMBED_DIM]

    return embedding[0]   # keep same format