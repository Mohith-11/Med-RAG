import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# ✅ Load E5 model
model = SentenceTransformer("intfloat/e5-base")

# 🔥 MRL dimension (change here if needed)
EMBED_DIM = 384   # try 768 / 512 / 384 / 256


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