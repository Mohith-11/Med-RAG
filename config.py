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

# Embedding
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 64

# Data
PDF_FOLDER = "data/pdfs"
