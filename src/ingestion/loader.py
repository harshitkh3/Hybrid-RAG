import os
import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from src.config import settings

logger = logging.getLogger(__name__)


def load_pdf_documents(docs_dir: str = None) -> List[Document]:
    """
    Scans the document directory and loads all PDF files page by page using PyPDFLoader.
    """
    directory = docs_dir or settings.DOCS_DIR
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Document directory '{directory}' does not exist.")

    pdf_files = sorted([f for f in os.listdir(directory) if f.lower().endswith(".pdf")])
    if not pdf_files:
        logger.warning("No PDF documents found in '%s'", directory)
        return []

    logger.info("Found %d PDF documents in '%s'", len(pdf_files), directory)
    all_docs: List[Document] = []

    for idx, filename in enumerate(pdf_files, start=1):
        file_path = os.path.join(directory, filename)
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            all_docs.extend(pages)
            logger.info("  [%02d/%02d] %s -> %d pages", idx, len(pdf_files), filename, len(pages))
        except Exception as e:
            logger.error("Failed to load PDF '%s': %s", filename, e, exc_info=True)

    logger.info("Total loaded pages across all documents: %d", len(all_docs))
    return all_docs
