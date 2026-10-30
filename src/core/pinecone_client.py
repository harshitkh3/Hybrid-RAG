import logging
from typing import Optional
from pinecone import Pinecone, ServerlessSpec, Index
from src.config import Settings, settings as default_settings
# Ensure upsert patch is applied
import src.core.patches  # noqa: F401

logger = logging.getLogger(__name__)

_pinecone_client: Optional[Pinecone] = None


def get_pinecone_client(settings: Optional[Settings] = None) -> Pinecone:
    """
    Returns a singleton instance of the Pinecone client.
    """
    global _pinecone_client
    if _pinecone_client is not None:
        return _pinecone_client

    cfg = settings or default_settings
    api_key = cfg.pinecone_key
    _pinecone_client = Pinecone(api_key=api_key)
    logger.info("Initialized Pinecone client.")
    return _pinecone_client


def get_or_create_index(
    settings: Optional[Settings] = None,
    client: Optional[Pinecone] = None,
) -> Index:
    """
    Connects to the Pinecone index, creating it with dotproduct metric and
    the configured dimension if it does not already exist.
    """
    cfg = settings or default_settings
    pc = client or get_pinecone_client(cfg)
    index_name = cfg.PINECONE_INDEX_NAME

    existing_indexes = pc.list_indexes().names()
    if index_name not in existing_indexes:
        logger.info(
            "Creating Pinecone index '%s' (dim=%d, metric='dotproduct', cloud=%s, region=%s)...",
            index_name,
            cfg.EMBEDDING_DIMENSION,
            cfg.PINECONE_CLOUD,
            cfg.PINECONE_REGION,
        )
        pc.create_index(
            name=index_name,
            dimension=cfg.EMBEDDING_DIMENSION,
            metric="dotproduct",
            spec=ServerlessSpec(cloud=cfg.PINECONE_CLOUD, region=cfg.PINECONE_REGION),
        )

    logger.info("Connected to Pinecone index: '%s'", index_name)
    return pc.Index(index_name)
