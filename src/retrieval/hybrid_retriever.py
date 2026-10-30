import logging
from typing import Optional, List
from langchain_core.documents import Document
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder
from src.config import settings
from src.core.embeddings import get_embeddings
from src.core.sparse import get_bm25_encoder
from src.core.pinecone_client import get_or_create_index
# Ensure patch is loaded
import src.core.patches  # noqa: F401

logger = logging.getLogger(__name__)

_retriever_instance: Optional[PineconeHybridSearchRetriever] = None


def get_hybrid_retriever(
    alpha: Optional[float] = None,
    top_k: Optional[int] = None,
    sparse_encoder: Optional[BM25Encoder] = None,
) -> PineconeHybridSearchRetriever:
    """
    Initializes or retrieves the PineconeHybridSearchRetriever singleton.
    Alpha determines balance:
      1.0 = Pure dense (semantic)
      0.0 = Pure sparse (keyword BM25)
      0.5 = Balanced hybrid search
    """
    global _retriever_instance

    target_alpha = alpha if alpha is not None else settings.DEFAULT_ALPHA
    target_top_k = top_k if top_k is not None else settings.DEFAULT_TOP_K

    if _retriever_instance is not None:
        _retriever_instance.alpha = target_alpha
        _retriever_instance.top_k = target_top_k
        return _retriever_instance

    embeddings = get_embeddings()
    encoder = sparse_encoder or get_bm25_encoder()
    index = get_or_create_index()

    logger.info(
        "Creating PineconeHybridSearchRetriever (namespace='%s', alpha=%.2f, top_k=%d)",
        settings.PINECONE_NAMESPACE,
        target_alpha,
        target_top_k,
    )

    _retriever_instance = PineconeHybridSearchRetriever(
        embeddings=embeddings,
        sparse_encoder=encoder,
        index=index,
        namespace=settings.PINECONE_NAMESPACE,
        top_k=target_top_k,
        alpha=target_alpha,
    )
    return _retriever_instance


def retrieve_documents(
    query: str,
    alpha: Optional[float] = None,
    top_k: Optional[int] = None,
) -> List[Document]:
    """
    Executes hybrid retrieval for a given query with custom alpha and top_k settings.
    """
    retriever = get_hybrid_retriever(alpha=alpha, top_k=top_k)
    return retriever.invoke(query)
