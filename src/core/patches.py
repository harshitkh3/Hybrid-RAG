import logging
from pinecone import Index

logger = logging.getLogger(__name__)


def apply_pinecone_upsert_patch() -> None:
    """
    Compatibility patch: LangChain Community's PineconeHybridSearchRetriever
    passes vectors positionally to index.upsert, whereas the modern Pinecone client (v5+)
    requires keyword-only arguments.
    """
    if getattr(Index, "_is_patched_for_langchain", False):
        return

    orig_upsert = Index.upsert

    def patched_upsert(self, *args, **kwargs):
        if args:
            kwargs["vectors"] = args[0]
        return orig_upsert(self, **kwargs)

    Index.upsert = patched_upsert
    Index._is_patched_for_langchain = True
    logger.debug("Applied Pinecone Index.upsert compatibility patch for LangChain.")


# Apply patch immediately on module import
apply_pinecone_upsert_patch()
