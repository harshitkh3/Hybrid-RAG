import logging
from typing import Dict, Any, Optional, Iterator, List
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.config import settings
from src.retrieval.hybrid_retriever import get_hybrid_retriever, retrieve_documents
from src.generation.llm import get_llm
from src.generation.prompts import get_rag_prompt, format_docs_with_metadata

logger = logging.getLogger(__name__)


def build_rag_chain(
    alpha: Optional[float] = None,
    top_k: Optional[int] = None,
):
    """
    Constructs an end-to-end LangChain Expression Language (LCEL) Hybrid RAG chain.
    """
    target_alpha = alpha if alpha is not None else settings.DEFAULT_ALPHA
    target_top_k = top_k if top_k is not None else settings.DEFAULT_TOP_K

    retriever = get_hybrid_retriever(alpha=target_alpha, top_k=target_top_k)
    prompt = get_rag_prompt()
    llm = get_llm()

    chain = (
        {
            "context": retriever | format_docs_with_metadata,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def ask_question(
    question: str,
    alpha: Optional[float] = None,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Answers a research question grounded strictly on retrieved hybrid search documents.
    Returns the generated answer together with detailed source citations and scores.
    """
    target_alpha = alpha if alpha is not None else settings.DEFAULT_ALPHA
    target_top_k = top_k if top_k is not None else settings.DEFAULT_TOP_K

    logger.info("Executing RAG query: '%s' (alpha=%.2f, top_k=%d)", question, target_alpha, target_top_k)

    # Retrieve context documents
    docs = retrieve_documents(question, alpha=target_alpha, top_k=target_top_k)

    # Format context and run generation
    context_str = format_docs_with_metadata(docs)
    prompt = get_rag_prompt()
    llm = get_llm()

    formatted_prompt = prompt.format(context=context_str, question=question)
    response = llm.invoke(formatted_prompt)
    answer = response.content if hasattr(response, "content") else str(response)

    # Extract source citations from retrieved docs
    sources: List[Dict[str, Any]] = []
    for doc in docs:
        score = doc.metadata.get("score")
        if isinstance(score, float):
            score = round(score, 4)
        sources.append({
            "document": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page"),
            "snippet": doc.page_content[:250].strip() + ("..." if len(doc.page_content) > 250 else ""),
            "score": score,
        })

    return {
        "question": question,
        "answer": answer,
        "alpha": target_alpha,
        "top_k": target_top_k,
        "sources": sources,
    }


def stream_answer(
    question: str,
    alpha: Optional[float] = None,
    top_k: Optional[int] = None,
) -> Iterator[str]:
    """
    Streams the RAG response token by token for real-time terminal or web clients.
    """
    chain = build_rag_chain(alpha=alpha, top_k=top_k)
    for chunk in chain.stream(question):
        yield chunk
