from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

RAG_SYSTEM_TEMPLATE = """You are an expert AI research assistant. Answer the question using ONLY the provided context below.

Formatting & Output Rules:
1. Base your answer solely on the information provided in the Context passages.
2. If the answer is not present in the context, clearly state: "I don't know based on the provided documents."
3. Do NOT use Markdown formatting such as bold (**text** or __text__), italics (*text*), or markdown headers (###).
4. Output in clean, readable plain text suitable for terminal/CLI display. Use simple dashes (-) or numbers (1, 2, 3) for lists without any asterisks or bold formatting.
5. End your answer with a "Sources Cited:" section listing the document filename and page number for each source utilized, in plain text.

Context:
{context}

Question:
{question}

Answer:
"""


def format_docs_with_metadata(docs: List[Document]) -> str:
    """
    Formats retrieved LangChain documents with explicit document source and page headers.
    """
    formatted = []
    for doc in docs:
        src = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", "Unknown Page")
        formatted.append(f"[Document: {src} | Page: {page}]\n{doc.page_content}")
    return "\n\n".join(formatted)


def get_rag_prompt() -> ChatPromptTemplate:
    """
    Returns the ChatPromptTemplate for grounded hybrid RAG generation.
    """
    return ChatPromptTemplate.from_template(RAG_SYSTEM_TEMPLATE)
