import argparse
import sys
from src.generation.rag_chain import ask_question, stream_answer
from src.ingestion.indexer import run_ingestion_pipeline


def main():
    parser = argparse.ArgumentParser(description="Hybrid RAG CLI: Query or Ingest research papers")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question using Hybrid RAG")
    ask_parser.add_argument("question", type=str, help="Question to ask")
    ask_parser.add_argument("--alpha", type=float, default=0.5, help="Hybrid weight (0.0=sparse, 1.0=dense, 0.5=hybrid)")
    ask_parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    ask_parser.add_argument("--stream", action="store_true", help="Stream the response")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Run document ingestion into Pinecone hybrid index")
    ingest_parser.add_argument("--docs-dir", type=str, default=None, help="Directory containing PDF files")
    ingest_parser.add_argument("--reindex", action="store_true", help="Force reindexing even if vectors exist")

    args = parser.parse_args()

    if args.command == "ask":
        if args.stream:
            print("=" * 80)
            print(f"QUESTION: {args.question}")
            print("=" * 80 + "\nANSWER:\n")
            for chunk in stream_answer(args.question, alpha=args.alpha, top_k=args.top_k):
                print(chunk, end="", flush=True)
            print("\n\n" + "=" * 80)
        else:
            res = ask_question(args.question, alpha=args.alpha, top_k=args.top_k)
            print("=" * 80)
            print(f"QUESTION: {res['question']}")
            print("=" * 80)
            print(f"\nANSWER:\n\n{res['answer']}\n")
            print("=" * 80)
            print(f"RETRIEVED CITATIONS ({len(res['sources'])} chunks):")
            for i, src in enumerate(res["sources"], 1):
                print(f"  [{i}] Source: {src['document']} (Page {src['page']}) | Score: {src['score']}")
            print("=" * 80)

    elif args.command == "ingest":
        print("Starting ingestion pipeline...")
        result = run_ingestion_pipeline(docs_dir=args.docs_dir, reindex=args.reindex)
        print(f"Result: {result['status']}")
        print(f"Message: {result['message']}")
        if "chunks_ingested" in result:
            print(f"Chunks ingested: {result['chunks_ingested']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
