import os
from dotenv import load_dotenv
import torch
from sentence_transformers import SentenceTransformer

load_dotenv()

# Switched to e5-large-v2 for higher accuracy
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("intfloat/e5-large-v2", device=device)

# e5-large output dimension
EMBED_DIM = 1024


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