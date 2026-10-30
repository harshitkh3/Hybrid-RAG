import logging
from typing import Dict, Any, Optional
from src.config import settings
from src.core.sparse import fit_and_save_bm25, get_bm25_encoder
from src.core.pinecone_client import get_or_create_index
from src.ingestion.loader import load_pdf_documents
from src.ingestion.splitter import split_documents
from src.retrieval.hybrid_retriever import get_hybrid_retriever

logger = logging.getLogger(__name__)


def run_ingestion_pipeline(
    docs_dir: Optional[str] = None,
    reindex: bool = False,
    bm25_path: Optional[str] = None,
    batch_size: int = 100,
) -> Dict[str, Any]:
    """
    Executes the complete document ingestion pipeline:
    1. Loads PDF documents from disk.
    2. Chunks documents with overlap while preserving metadata.
    3. Fits/updates and persists the BM25 sparse encoder.
    4. Upserts dense and sparse representations into the Pinecone hybrid index.
    """
    target_dir = docs_dir or settings.DOCS_DIR
    target_bm25 = bm25_path or settings.BM25_PATH
    namespace = settings.PINECONE_NAMESPACE

    index = get_or_create_index()
    stats = index.describe_index_stats()
    ns_info = stats.get("namespaces", {}).get(namespace, {})
    ns_count = ns_info.get("vector_count", 0)

    logger.info("Current vector count in namespace '%s': %d", namespace, ns_count)

    if ns_count > 0 and not reindex:
        message = f"Namespace '{namespace}' already contains {ns_count} vectors. Skipping ingestion (set reindex=True to overwrite)."
        logger.info(message)
        return {
            "status": "skipped",
            "message": message,
            "namespace": namespace,
            "existing_vectors": ns_count,
            "chunks_ingested": 0,
        }

    # Load and chunk documents
    documents = load_pdf_documents(target_dir)
    if not documents:
        raise ValueError(f"No documents were loaded from '{target_dir}'.")

    chunks = split_documents(documents)
    if not chunks:
        raise ValueError("Document splitting resulted in zero non-empty chunks.")

    # Fit and save BM25 sparse encoder
    corpus = [c.page_content for c in chunks]
    bm25 = fit_and_save_bm25(corpus, target_bm25)

    # If reindexing, clear existing namespace
    if ns_count > 0 and reindex:
        logger.info("Reindex requested: deleting existing vectors in namespace '%s'...", namespace)
        try:
            index.delete(delete_all=True, namespace=namespace)
        except Exception as e:
            logger.warning("Could not delete vectors from namespace '%s': %s", namespace, e)

    # Initialize retriever
    retriever = get_hybrid_retriever(sparse_encoder=bm25)

    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    total_chunks = len(texts)
    logger.info("Upserting %d chunks into Pinecone namespace '%s' in batches of %d...", total_chunks, namespace, batch_size)

    for i in range(0, total_chunks, batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_metadatas = metadatas[i : i + batch_size]
        retriever.add_texts(texts=batch_texts, metadatas=batch_metadatas, namespace=namespace)
        logger.info("  Upserted chunks %d to %d of %d", i + 1, min(i + batch_size, total_chunks), total_chunks)

    # Fetch updated stats
    updated_stats = index.describe_index_stats()
    updated_ns_count = updated_stats.get("namespaces", {}).get(namespace, {}).get("vector_count", total_chunks)

    message = f"Successfully ingested {total_chunks} chunks into namespace '{namespace}'."
    logger.info(message)
    return {
        "status": "success",
        "message": message,
        "namespace": namespace,
        "total_documents": len(documents),
        "chunks_ingested": total_chunks,
        "current_vector_count": updated_ns_count,
    }
