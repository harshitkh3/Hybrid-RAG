"""
Core components for embeddings, sparse encoding, and Pinecone client management.
"""

from src.core.patches import apply_pinecone_upsert_patch
from src.core.embeddings import get_embeddings
from src.core.sparse import get_bm25_encoder, fit_and_save_bm25
from src.core.pinecone_client import get_pinecone_client, get_or_create_index

__all__ = [
    "apply_pinecone_upsert_patch",
    "get_embeddings",
    "get_bm25_encoder",
    "fit_and_save_bm25",
    "get_pinecone_client",
    "get_or_create_index",
]
