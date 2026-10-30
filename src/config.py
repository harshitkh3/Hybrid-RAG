import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Pinecone Configuration
    PINECONE_API: Optional[str] = None
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_INDEX_NAME: str = "hybrid-rag"
    PINECONE_NAMESPACE: str = "ai-docs"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"

    # Google Vertex AI / GenAI Configuration
    GOOGLE_PROJECT_ID: str = "ai-api-2335"
    GOOGLE_LOCATION: str = "us-central1"
    GOOGLE_REGION: str = "global"
    GOOGLE_API_KEY: Optional[str] = None
    VERTEX_AI: bool = True

    # Model Configuration
    EMBEDDING_MODEL: str = "text-embedding-004"
    EMBEDDING_DIMENSION: int = 768
    LLM_MODEL: str = "gemini-2.5-flash"
    LLM_TEMPERATURE: float = 0.0

    # Ingestion & Chunking
    DOCS_DIR: str = "./ai-docs"
    BM25_PATH: str = "bm25_ai_docs.json"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150

    # Retrieval Defaults
    DEFAULT_ALPHA: float = 0.5
    DEFAULT_TOP_K: int = 5

    # Server Configuration
    SERVER_HOST: str = "127.0.0.1"
    SERVER_PORT: int = 8000

    @property
    def pinecone_key(self) -> str:
        key = (
            self.PINECONE_API_KEY
            or self.PINECONE_API
            or os.environ.get("PINECONE_API")
            or os.environ.get("PINECONE_API_KEY")
        )
        if not key:
            raise ValueError(
                "Pinecone API key not found. Please set PINECONE_API or PINECONE_API_KEY in your environment or .env file."
            )
        return key


settings = Settings()
