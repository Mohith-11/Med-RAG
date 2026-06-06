from pinecone import Pinecone
import os
import time
import sys
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("rag-project")

# Transient network error types to catch
_NETWORK_ERRORS = (
    ConnectionError,
    TimeoutError,
    OSError,        # covers [Errno 11001] getaddrinfo / socket errors
)
try:
    import urllib3
    _NETWORK_ERRORS = _NETWORK_ERRORS + (urllib3.exceptions.MaxRetryError,
                                         urllib3.exceptions.NewConnectionError)
except ImportError:
    pass


def query_index(vector, top_k=5, max_retries: int = 5, base_delay: float = 2.0):
    """Query the Pinecone index with exponential backoff on transient errors.

    Args:
        vector:      The query embedding (numpy array or list).
        top_k:       Number of nearest neighbours to retrieve.
        max_retries: Maximum number of attempts before re-raising.
        base_delay:  Initial wait time in seconds; doubles each retry.

    Returns:
        List of Pinecone match dicts.

    Raises:
        The last exception if all retries are exhausted.
    """
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            results = index.query(
                vector=vector.tolist(),
                top_k=top_k,
                include_metadata=True,
            )
            return results["matches"]
        except _NETWORK_ERRORS as exc:
            last_exc = exc
            wait = base_delay ** attempt          # 2, 4, 8, 16, 32 s
            print(
                f"\n[PINECONE] Network error on attempt {attempt}/{max_retries}: "
                f"{type(exc).__name__} – retrying in {wait:.0f}s…",
                file=sys.stderr,
            )
            time.sleep(wait)
        except Exception as exc:
            # Non-network errors (auth, bad vector dim, etc.) – fail fast
            raise

    # All retries exhausted
    print(
        f"\n[PINECONE] All {max_retries} retries failed. Last error: {last_exc}",
        file=sys.stderr,
    )
    raise last_exc