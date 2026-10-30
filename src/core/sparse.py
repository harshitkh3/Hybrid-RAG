import os
import logging
from typing import List, Optional
from pinecone_text.sparse import BM25Encoder
from src.config import settings

logger = logging.getLogger(__name__)


def get_bm25_encoder(path: Optional[str] = None) -> BM25Encoder:
    """
    Loads an existing BM25 encoder from disk if available, or returns a new encoder instance.
    """
    bm25_path = path or settings.BM25_PATH
    encoder = BM25Encoder()

    if os.path.exists(bm25_path):
        logger.info("Loading existing BM25 encoder from '%s'", bm25_path)
        return encoder.load(bm25_path)

    logger.warning("BM25 encoder file not found at '%s'. Returning un-fitted encoder.", bm25_path)
    return encoder


def fit_and_save_bm25(corpus: List[str], path: Optional[str] = None) -> BM25Encoder:
    """
    Fits a BM25 encoder on the provided text corpus and saves it to disk.
    """
    bm25_path = path or settings.BM25_PATH
    logger.info("Fitting BM25 encoder on %d text chunks...", len(corpus))
    encoder = BM25Encoder()
    encoder.fit(corpus)
    encoder.dump(bm25_path)
    logger.info("BM25 encoder successfully saved to '%s'", bm25_path)
    return encoder
