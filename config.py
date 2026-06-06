"""Centralized configuration for the Oncology RAG pipeline."""
import os
from dotenv import load_dotenv

load_dotenv()

# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "med-rag-index")

# LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Retrieval
TOP_K_RETRIEVE = 10
TOP_K_RERANK = 5
MIN_CHUNK_WORDS = 40

# Embedding (defaults chosen based on selected model)
# Use 'intfloat/e5-large-v2' for best accuracy if you have sufficient RAM/VRAM
EMBEDDING_MODEL      = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
# e5-large variants produce 1024-dim vectors and require smaller batches
if "e5-large" in EMBEDDING_MODEL or EMBEDDING_MODEL.startswith("intfloat/e5") or "e5" in EMBEDDING_MODEL:
	_default_dim = 1024
	_default_batch = 32
else:
	_default_dim = 384
	_default_batch = 64

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", _default_dim))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", _default_batch))

# Data
PDF_FOLDER = "data/pdfs"
