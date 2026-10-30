import logging
from typing import Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config import Settings, settings as default_settings

logger = logging.getLogger(__name__)

_embedding_instance: Optional[GoogleGenerativeAIEmbeddings] = None


def get_embeddings(settings: Optional[Settings] = None) -> GoogleGenerativeAIEmbeddings:
    """
    Returns a singleton instance of GoogleGenerativeAIEmbeddings configured
    for Vertex AI text-embedding-004.
    """
    global _embedding_instance
    if _embedding_instance is not None:
        return _embedding_instance

    cfg = settings or default_settings

    kwargs = {
        "model": cfg.EMBEDDING_MODEL,
    }

    if cfg.VERTEX_AI:
        kwargs.update({
            "vertexai": True,
            "project": cfg.GOOGLE_PROJECT_ID,
            "region": cfg.GOOGLE_REGION,
        })
    elif cfg.GOOGLE_API_KEY:
        kwargs["google_api_key"] = cfg.GOOGLE_API_KEY

    logger.info("Initializing Google GenAI Embeddings model=%s (vertexai=%s)", cfg.EMBEDDING_MODEL, cfg.VERTEX_AI)
    _embedding_instance = GoogleGenerativeAIEmbeddings(**kwargs)
    return _embedding_instance
