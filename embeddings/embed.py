import os
from dotenv import load_dotenv

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from sentence_transformers import SentenceTransformer
import config

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"


def _load_model():
    """Load e5-large-v2 with the lowest practical CPU memory footprint."""
    model_name = getattr(config, "EMBEDDING_MODEL", "intfloat/e5-large-v2")
    offload_folder = os.path.join(os.path.dirname(__file__), ".hf_offload", model_name.replace('/', '_'))
    os.makedirs(offload_folder, exist_ok=True)

    load_attempts = [
        {"low_cpu_mem_usage": True, "torch_dtype": torch.float16},
        {
            "low_cpu_mem_usage": True,
            "device_map": "auto",
            "max_memory": {0: "256MiB", "cpu": "512MiB"},
            "offload_folder": offload_folder,
            "offload_state_dict": True,
            "torch_dtype": torch.float16,
        },
        {
            "low_cpu_mem_usage": True,
            "device_map": "auto",
            "max_memory": {0: "256MiB", "cpu": "512MiB"},
            "offload_folder": offload_folder,
            "offload_state_dict": True,
            "torch_dtype": torch.bfloat16,
        },
        {"low_cpu_mem_usage": True, "torch_dtype": torch.float32},
    ]

    last_error = None
    for model_kwargs in load_attempts:
        try:
            return SentenceTransformer(
                model_name,
                device=device,
                model_kwargs=model_kwargs,
            )
        except Exception as exc:
            last_error = exc

    raise last_error


# Lazy wrapper so importing this module doesn't eagerly allocate memory.
class _LazyModel:
    def __init__(self):
        self._model = None

    def _ensure(self):
        if self._model is None:
            try:
                self._model = _load_model()
            except Exception as exc:
                msg = (
                    "Failed to load 'intfloat/e5-large-v2' on this machine. "
                    "This is usually due to insufficient RAM or paging-file size (Windows) or insufficient GPU memory. "
                    "Options: 1) Increase Windows paging file size; 2) Run on a machine with more RAM/VRAM; "
                    "3) Use a smaller embedding model (e.g. 'all-MiniLM-L6-v2') by setting EMBEDDING_MODEL in config.py. "
                    f"Original error: {exc}"
                )
                raise RuntimeError(msg) from exc

    def encode(self, *args, **kwargs):
        self._ensure()
        return self._model.encode(*args, **kwargs)

    def __getattr__(self, name):
        self._ensure()
        return getattr(self._model, name)

# Lazy loader wrapper to avoid import-time OOMs

class LazyST:
    def __init__(self):
        self._model = None

    def _ensure(self):
        if self._model is None:
            try:
                self._model = _load_model()
            except Exception as exc:
                msg = (
                    f"Failed to load '{getattr(config, 'EMBEDDING_MODEL', 'intfloat/e5-large-v2')}' on this machine. "
                    "This is usually due to insufficient RAM or paging-file size (Windows) or insufficient GPU memory. "
                    "Options: 1) Increase Windows paging file size; 2) Run on a machine with more RAM/VRAM; "
                    "3) Use a smaller embedding model (e.g. 'all-MiniLM-L6-v2') by setting EMBEDDING_MODEL in config.py. "
                    f"Original error: {exc}"
                )
                raise RuntimeError(msg) from exc

    def encode(self, *args, **kwargs):
        self._ensure()
        return self._model.encode(*args, **kwargs)


model = LazyST()

# Keep e5-large-v2, but load it in a memory-friendlier way on Windows.
model = LazyST()

# e5-large output dimension
EMBED_DIM = 1024


# ✅ Embed passages (documents)
def embed_passages(texts):
    texts = [f"passage: {t}" for t in texts]

    batch_size = getattr(config, "EMBEDDING_BATCH_SIZE", 32)
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size
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


# ✅ Embed a short text as a passage (document-side prefix)
# Used for HyDE-style retrieval: embed an answer-template query with the
# same "passage:" prefix used at ingestion time, so it matches answer vocabulary.
def embed_passage(text):
    """Embed text using the document-side 'passage:' prefix (e5-large-v2)."""
    embedding = model.encode(
        [f"passage: {text}"],
        normalize_embeddings=True
    )
    embedding = embedding[:, :EMBED_DIM]
    return embedding[0]