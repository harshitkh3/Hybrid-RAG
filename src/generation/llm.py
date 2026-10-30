import logging
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import Settings, settings as default_settings

logger = logging.getLogger(__name__)

_llm_instance: Optional[ChatGoogleGenerativeAI] = None


def get_llm(
    temperature: Optional[float] = None,
    settings: Optional[Settings] = None,
) -> ChatGoogleGenerativeAI:
    """
    Returns a ChatGoogleGenerativeAI LLM instance configured for Gemini 2.5 Flash.
    """
    global _llm_instance
    cfg = settings or default_settings
    temp = temperature if temperature is not None else cfg.LLM_TEMPERATURE

    if _llm_instance is not None and temperature is None:
        return _llm_instance

    kwargs = {
        "model": cfg.LLM_MODEL,
        "temperature": temp,
    }

    if cfg.VERTEX_AI:
        kwargs.update({
            "vertexai": True,
            "project": cfg.GOOGLE_PROJECT_ID,
            "location": cfg.GOOGLE_LOCATION,
        })
    elif cfg.GOOGLE_API_KEY:
        kwargs["google_api_key"] = cfg.GOOGLE_API_KEY

    logger.info("Initializing LLM model=%s (temp=%.2f, vertexai=%s)", cfg.LLM_MODEL, temp, cfg.VERTEX_AI)
    instance = ChatGoogleGenerativeAI(**kwargs)

    if temperature is None:
        _llm_instance = instance
    return instance
