# Hybrid RAG: Enterprise Research Retrieval-Augmented Generation

An enterprise-grade, modular **Hybrid Retrieval-Augmented Generation (Hybrid RAG)** service built with **FastAPI**, **Pinecone Serverless**, **Google Vertex AI / Gemini 2.5 Flash**, and **BM25 Sparse Encoding**.

This system allows users to ask complex questions over academic and technical research papers, retrieving relevant passages using a convex combination of dense semantic embeddings and sparse keyword representations, and generating strictly-grounded answers with precise citations (document names and page numbers).

---

## Architecture Overview

```
                      +---------------------------------------+
                      |          User Query / Request         |
                      +-------------------+-------------------+
                                          |
                        +-----------------+-----------------+
                        |                                   |
              [Dense Vectorization]                [Sparse Vectorization]
           Google text-embedding-004                  BM25Encoder
                   (768 dim)                      (Term & Doc Frequency)
                        |                                   |
                        +-----------------+-----------------+
                                          |
                                          v
                    +-------------------------------------------+
                    |        Pinecone Hybrid Search Index       |
                    | (metric: dotproduct, alpha-weighted blend)|
                    +---------------------+---------------------+
                                          |
                              Top-K Document Chunks
                             (Text & Page Metadata)
                                          |
                                          v
                    +-------------------------------------------+
                    |             Grounded Prompt               |
                    |    (Strict Context & Citation Rules)     |
                    +---------------------+---------------------+
                                          |
                                          v
                    +-------------------------------------------+
                    |             Gemini 2.5 Flash              |
                    |          (Grounded Generation)            |
                    +---------------------+---------------------+
                                          |
                                          v
                    +-------------------------------------------+
                    |      FastAPI REST / Streaming Output      |
                    |      - Answer text                        |
                    |      - Source citations & page numbers    |
                    +-------------------------------------------+
```

### Key Highlights

- **Convex Hybrid Retrieval**: Blends dense semantic search ($\alpha \cdot \text{dense}$) and sparse BM25 exact keyword matching ($(1 - \alpha) \cdot \text{sparse}$) on Pinecone Serverless with the `dotproduct` metric.
- **Strictly Grounded Generation**: Powered by Google `gemini-2.5-flash`, configured to answer solely based on retrieved passages, preventing hallucination, and requiring explicit page citations.
- **Modular Production Codebase**: Divided cleanly into dedicated modules for configuration, core clients, ingestion, retrieval, generation, and API routing.
- **High-Performance FastAPI Service**: Provides REST endpoints for query evaluation, token streaming, automated document ingestion, and health monitoring.
- **CLI Utility**: Includes a command-line interface for querying and batch ingestion directly from the terminal.

---

## Project Structure

```
├── ai-docs/                     # Source PDF documents for indexing
├── src/
│   ├── __init__.py              # Package entrypoint
│   ├── config.py                # Environment and configuration settings
│   ├── core/                    # Core infrastructure & clients
│   │   ├── __init__.py
│   │   ├── embeddings.py        # Google Vertex AI dense embeddings (text-embedding-004)
│   │   ├── sparse.py            # BM25 sparse encoder setup, fit, and persistence
│   │   ├── pinecone_client.py   # Pinecone index connection and lifecycle
│   │   └── patches.py           # Pinecone upsert compatibility patch for LangChain
│   ├── ingestion/               # Document ingestion pipeline
│   │   ├── __init__.py
│   │   ├── loader.py            # PDF loading with PyPDFLoader
│   │   ├── splitter.py          # Chunking with RecursiveCharacterTextSplitter
│   │   └── indexer.py           # Ingestion coordinator & vector upsert
│   ├── retrieval/               # Retrieval logic
│   │   ├── __init__.py
│   │   └── hybrid_retriever.py  # PineconeHybridSearchRetriever integration
│   ├── generation/              # Generative QA pipeline
│   │   ├── __init__.py
│   │   ├── llm.py               # Gemini 2.5 Flash chat model
│   │   ├── prompts.py           # Grounding and citation prompt templates
│   │   └── rag_chain.py         # LCEL chain and question answering logic
│   └── api/                     # FastAPI application layer
│       ├── __init__.py
│       ├── schemas.py           # Pydantic request & response schemas
│       └── routes.py            # API route endpoints (/ask, /ask/stream, /ingest, /health)
├── bm25_ai_docs.json            # Persisted BM25 vocabulary & term statistics
├── cli.py                       # Command-line utility for ingestion and queries
├── main.py                      # FastAPI application server entrypoint
├── pyproject.toml               # Project metadata and dependencies
├── requirements.txt             # Pip dependency specifications
└── README.md                    # Project documentation
```

---

## Installation & Setup

### 1. Prerequisites

- Python 3.10+ (or Python 3.13)
- Pinecone account & API key
- Google Cloud project with Vertex AI enabled (or Google AI API key)

### 2. Environment Variables

Create a `.env` file in the project root:

```env
# Pinecone Configuration
PINECONE_API=your_pinecone_api_key_here
PINECONE_INDEX_NAME=hybrid-rag
PINECONE_NAMESPACE=ai-docs
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# Google Vertex AI / GenAI Configuration
GOOGLE_PROJECT_ID=ai-api-2335
GOOGLE_LOCATION=us-central1
GOOGLE_REGION=global

# Optional: Direct Google AI Key fallback
# GOOGLE_API_KEY=your_google_api_key
```

### 3. Install Dependencies

Using `uv` (recommended):

```bash
uv sync
```

Or using standard `pip`:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Document Ingestion & Indexing

To ingest PDF research papers located in `./ai-docs/`:

### Option A: Using the CLI

```bash
# Check status / ingest if index is empty
python cli.py ingest

# Force reindexing (clears existing namespace and re-chunks)
python cli.py ingest --reindex
```

### Option B: Using the API

Send a `POST` request to `/ingest`:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"reindex": false}'
```

---

## Running the FastAPI Application

Start the server using `uvicorn` or the `main.py` entrypoint:

```bash
# Direct Python execution:
python main.py

# Or via uvicorn with hot-reload:
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

The server binds to `http://127.0.0.1:8000`. The interactive Swagger documentation is available at:
- **Interactive OpenAPI UI (Test directly in browser)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## API Endpoints Reference

### 1. Ask a RAG Question

Evaluates a question against the indexed documents, performing hybrid search and returning a grounded answer with citations. Supports both **POST** (with JSON body) and **GET** (with query parameters, easily tested directly in a browser).

- **URL**: `POST /ask` (or `GET /ask?question=...`)
- **Alternative URL**: `/api/v1/ask`

#### Parameters / Body

| Field | Type | Default | Description |
|---|---|---|---|
| `question` | `string` | *Required* | The question to ask over the documents. |
| `alpha` | `float` | `0.5` | Hybrid balance (0.0 = pure BM25, 1.0 = pure semantic, 0.5 = balanced). |
| `top_k` | `integer` | `5` | Number of context chunks to retrieve (1 to 20). |

#### Example 1: In the Web Browser (GET)
Paste directly into your browser address bar:
```
http://127.0.0.1:8000/ask?question=How+can+lung+cancer+be+detected+with+ML&alpha=0.5&top_k=3
```

#### Example 2: PowerShell (Windows)
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/ask" -Method Post -ContentType "application/json" -Body '{"question": "How can lung cancer be detected with ML?", "alpha": 0.5, "top_k": 3}' | ConvertTo-Json -Depth 5
```

#### Example 3: cURL (Bash / Linux / macOS)
```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How can lung cancer be detected with ML?",
    "alpha": 0.5,
    "top_k": 3
  }'
```

#### Example 4: Windows Command Prompt (CMD)
```cmd
curl.exe -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: application/json" -d "{\"question\": \"How can lung cancer be detected with ML?\", \"alpha\": 0.5, \"top_k\": 3}"
```

#### Example Response (`200 OK`)

```json
{
  "question": "How can lung cancer be detected with ML?",
  "answer": "Lung cancer can be detected with machine learning (ML) using deep learning architectures like Convolutional Neural Networks (CNNs) and VGG16 to analyze CT scan images. These models automatically learn features from imaging data to classify scans into normal, benign, and malignant categories.\n\nSources Cited:\nai-docs\\AI-Powered Lung Cancer Detection- Assessing VGG16 and CNN Architectures for CT Scan Image Classification.pdf | Page: 1",
  "alpha": 0.5,
  "top_k": 3,
  "sources": [
    {
      "document": "ai-docs\\AI-Powered Lung Cancer Detection- Assessing VGG16 and CNN Architectures for CT Scan Image Classification.pdf",
      "page": 1,
      "snippet": "cancer through CT scan image analysis using an openly accessible dataset for training and evaluation. The model is trained to classify CT scan images into three categories...",
      "score": 0.6276
    },
    {
      "document": "ai-docs\\Deep learning-based approach to diagnose lung cancer using CT-scan images.pdf",
      "page": 15,
      "snippet": "scan using deep cubical nodule transfer learning algorithm. Front Oncol Jul. 2021; 11:661244...",
      "score": 0.6158
    }
  ]
}
```

---

### 2. Stream RAG Answer

Streams plain-text tokens generated by Gemini 2.5 Flash in real time. Supports both **POST** and **GET**.

- **URL**: `POST /ask/stream` or `GET /ask/stream?question=...`
- **Content-Type**: `text/plain; charset=utf-8`

#### Browser / cURL Streaming (GET)
```bash
curl -N "http://127.0.0.1:8000/ask/stream?question=What+is+the+Scale-Aware+Curriculum+Learning+approach+for+lung+nodule+detection+with+YOLOv11?"
```

#### POST Streaming Request
```bash
curl -N -X POST http://127.0.0.1:8000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the Scale-Aware Curriculum Learning approach for lung nodule detection with YOLOv11?",
    "alpha": 0.5,
    "top_k": 4
  }'
```

---

### 3. Service Health & Vector Index Statistics

Returns the connectivity status and Pinecone vector statistics.

- **URL**: `GET /health` (or `GET /api/v1/health`)

#### Example Request

```bash
curl -X GET http://localhost:8000/health
```

#### Example Response (`200 OK`)

```json
{
  "status": "healthy",
  "index_name": "hybrid-rag",
  "metric": "dotproduct",
  "dimension": 768,
  "namespace": "ai-docs",
  "namespace_vectors": 1177,
  "total_index_vectors": 1199
}
```

---

### 4. Trigger Ingestion

Triggers batch processing of PDFs from the document directory.

- **URL**: `POST /ingest` (or `POST /api/v1/ingest`)

#### Example Request

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"reindex": false}'
```

---

## CLI Usage

The CLI enables executing RAG operations directly in the terminal:

```bash
# Ask a question (balanced hybrid search)
python cli.py ask "What deep learning approaches are discussed for satellite image time series analysis?"

# Ask a question using streaming output
python cli.py ask "How do VGG16 and CNN architectures compare for CT scan image classification?" --stream

# Use pure keyword search (alpha=0.0)
python cli.py ask "YOLOv11 lung nodule detection" --alpha 0.0

# Use pure dense semantic search (alpha=1.0)
python cli.py ask "Explain the methodology for land cover mapping" --alpha 1.0

# Ingest or re-index documents
python cli.py ingest --reindex
```

---

## Hybrid Search Parameter Tuning ($\alpha$)

The `alpha` parameter balances dense semantic similarity against sparse keyword frequency:

$$\text{Final Score} = \alpha \cdot \text{Dense Score} + (1 - \alpha) \cdot \text{Sparse Score}$$

| Alpha Value | Search Type | Best Used For |
|---|---|---|
| `alpha = 1.0` | **Dense Semantic Search** | High-level conceptual questions, paraphrased language, exploratory queries. |
| `alpha = 0.0` | **Sparse Keyword Search (BM25)** | Exact model names (`YOLOv11`, `VGG16`), medical acronyms (`NSCLC`), specific authors or identifiers. |
| `alpha = 0.5` | **Balanced Hybrid Search (Recommended)** | Standard domain queries that combine conceptual context with specific scientific keywords. |
