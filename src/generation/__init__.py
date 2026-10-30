"""
Generative components: LLM management, prompts, and LCEL RAG chains.
"""

from src.generation.llm import get_llm
from src.generation.prompts import get_rag_prompt, format_docs_with_metadata
from src.generation.rag_chain import build_rag_chain, ask_question, stream_answer

__all__ = [
    "get_llm",
    "get_rag_prompt",
    "format_docs_with_metadata",
    "build_rag_chain",
    "ask_question",
    "stream_answer",
]
