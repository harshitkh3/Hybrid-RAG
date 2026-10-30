import logging
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import settings

logger = logging.getLogger(__name__)


def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """
    Splits documents into granular text chunks using RecursiveCharacterTextSplitter,
    preserving metadata such as source and page numbers.
    """
    size = chunk_size or settings.CHUNK_SIZE
    overlap = chunk_overlap or settings.CHUNK_OVERLAP

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks = text_splitter.split_documents(documents)
    # Filter out empty or whitespace-only chunks
    chunks = [c for c in raw_chunks if c.page_content and c.page_content.strip()]

    logger.info(
        "Split %d pages into %d chunks (chunk_size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        size,
        overlap,
    )
    return chunks
