import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config import settings
from src.api.routes import router

app = FastAPI(
    title="Hybrid RAG API",
    description=(
        "Production Hybrid Retrieval-Augmented Generation (Hybrid RAG) service over AI research documents. "
        "Combines Google Vertex AI dense embeddings (text-embedding-004), Pinecone serverless hybrid search "
        "with BM25 sparse keyword representations, and Gemini 2.5 Flash for strictly-grounded generative answers."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for external frontend or web apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes under root and under /api/v1 prefix
app.include_router(router, prefix="")
app.include_router(router, prefix="/api/v1")


@app.get("/", summary="Root Welcome Endpoint", tags=["General"])
async def root():
    return {
        "service": "Hybrid RAG API",
        "version": "0.1.0",
        "description": "Hybrid Retrieval-Augmented Generation with Pinecone, Google Vertex AI, and Gemini",
        "docs_url": "/docs",
        "health_url": "/health",
        "ask_endpoint": "/ask",
        "stream_endpoint": "/ask/stream",
    }


def main():
    uvicorn.run(
        "main:app",
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()
