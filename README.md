# 🎓 Research Paper Assistant - RAG System Assessment

## 🎯 Objective
Build a production-ready RAG (Retrieval-Augmented Generation) service that helps researchers efficiently query and understand academic papers.

## 💡 The Problem
Researchers waste hours reading through multiple papers to find:
- Specific methodologies and approaches
- Key findings and results
- Dataset information and benchmarks
- Comparative analysis across papers
- Citations and references


## ⚙️ Setup Instructions

### Prerequisites
- Python 3.10+
- Docker (for Qdrant)
- 8GB RAM recommended

### Quick Start (5 minutes)


1. **Clone and enter the directory**

```bash
git clone https://github.com/rafid-2020331095/research-paper-rag-assessment.git
cd research-paper-rag-assessment
```

2. **Configure environment/**
create a .env and copy files from .env.example to .env
```bash
copy .env.example .env   # Windows

```

3. **Start all services with Docker Compose**

```bash
docker-compose up -d
```

4. **API documentation**
   test the apis on the following link
   Visit: [http://localhost:8000/docs](http://localhost:8000/docs)


## 📝 Implementation Summary

This PR implements an end-to-end FastAPI-based RAG system for academic papers with robust ingestion, section-aware chunking, vector storage in Qdrant, and grounded QA with citations. The flow:

- Upload one or more PDFs via `POST /api/papers/upload`. Each file is persisted under `uploaded_papers/`, Parsed with `PyMuPDF` and sectionized using heuristics in `PDFProcessor._identify_sections`, driven by canonical SECTION_PATTERNS for Abstract, Introduction, Methods, Results, Discussion, Conclusion, and References. Text is chunked per section into bigger, section-based chunks with a target of ~1200 tokens, a maximum of 1800 tokens, a minimum of 300 tokens, and ~60 tokens of overlap, preserving sentence boundaries and page numbers.
- Embeddings are generated in batches using Sentence Transformers `all-MiniLM-L6-v2` (dim=384). Chunks are stored to Qdrant with payload metadata: `paper_id`, `paper_title`, `content`, `section`, `page_number`, `chunk_index`. Vector ids returned by Qdrant are persisted alongside chunks in Postgres.
- Queries hit `POST /api/query?query=...` with optional `paper_ids` or `paper_filenames` for filtering. A query embedding is computed and used to retrieve top-k similar chunks from Qdrant, with an internal filter on `paper_id`. Retrieved chunks are formatted into a structured context block.
- The LLM (`LLM_TYPE` switchable; defaults to Ollama Cloud through `ollama` client; `deepseek-v3.1:671b`supported) receives a structured prompt that enforces grounding, citation formatting, and emits a confidence score. The answer is returned with citations built from the retrieved payloads and mapped to stored chunks where possible.
- Query history (text, response time, optional rating) is stored, retrievable via `GET /api/query/history`, and rateable via `POST /api/query/history/{id}/rate`.
- Analytics exposes `GET /api/analytics/popular`, clustering past queries (KMeans over query embeddings) to present popular semantic topics with example queries.

---

## 🛠️ Technology Choices

- **LLM**: Ollama Cloud (default) : model-`deepseek-v3.1:671b`
  - Why: Switchable via env, reliable generation with strict prompting; Ollama Cloud path simplifies local setup, DeepSeek available via API key.
- **Embeddings**: sentence-transformers `all-MiniLM-L6-v2` (384-dim)
  - Why: Strong speed/quality tradeoff for academic text; small footprint.
- **Vector DB**: Qdrant (cosine distance, size=384)
  - Why: Fast ANN search, easy metadata payloads and filters, simple local Docker.
- **Database**: PostgreSQL/NeonDB via SQLAlchemy Async + asyncpg
  - Why: Stores papers, chunks, queries, answers, citations; async fits FastAPI.

Key Libraries:
- FastAPI (async API, OpenAPI docs)
- qdrant-client (vector operations)
- Pymupdf (PDF extraction)

---

###FULL API DOCUMENTATION
---
| Method | Endpoint | Description | Key Parameters | Example Curl |
|--------|----------|-------------|----------------|--------------|
| POST | `/api/papers/upload` | Upload PDFs, parse, embed, store | `files` (array of files) | `curl -X POST ... -F "files=@paper_4.pdf"` |
| GET | `/api/papers` | List all uploaded papers | None | `curl -X GET "http://127.0.0.1:8000/api/papers"` |
| GET | `/api/papers/{paper_id}` | Get paper metadata by ID | `paper_id` (path) | `curl -X GET "http://127.0.0.1:8000/api/papers/52"` |
| GET | `/api/papers/{paper_id}/download` | Download PDF by ID | `paper_id` (path) | `curl -X GET "http://127.0.0.1:8000/api/papers/52/download"` |
| DELETE | `/api/papers` | Bulk delete papers & embeddings | `ids` (array<int>, query) | `curl -X DELETE "http://127.0.0.1:8000/api/papers?ids=48&ids=49&ids=50"` |
| POST | `/api/query` | RAG query with citations | `query` (string, required); `paper_ids`or `paper_filenames(string)` fillup anyone, `top_k` (optional) | `curl -X POST "http://127.0.0.1:8000/api/query?query=What%20methodology..."` |
| GET | `/api/query/history` | Get query history | `skip`, `limit` (query, optional) | `curl -X GET "http://127.0.0.1:8000/api/query/history?limit=10"` |
| POST | `/api/query/history/{query_id}/rate` | Rate a query (1-5 stars) | `query_id` (path); `rating` (query) | `curl -X POST "http://127.0.0.1:8000/api/query/history/52/rate?rating=4"` |
| GET | `/api/analytics/popular` | Popular topics via k-means | `k`, `max_samples` (query, optional) | `curl -X GET "http://127.0.0.1:8000/api/analytics/popular?k=6&max_samples=3"` |
| GET | `/` | Health check | None | `curl -X GET "http://127.0.0.1:8000/"` |

## 🏗️ Architecture Overview

Key Components:
1) API Layer (FastAPI)
   - Request validation, error handling, response formatting
2) Processing Pipeline
   - PDF text extraction → section-aware chunking → embeddings → metadata
3) Storage Layer
   - Qdrant (vectors), PostgreSQL/NeonDB (papers, queries, analytics)
4) RAG Pipeline
   - Retrieval (top-k) → context assembly → LLM generation with citations

## DIAGRAM
![ER Diagram](./architecture.png)
---

## 🎯 Design Decisions

### 1) Chunking Strategy
- Approach: Section-aware chunking with sentence boundaries; section-based chunks with a target of ~1200 tokens, a maximum of 1800 tokens, a minimum of 300 tokens, and ~60 tokens of overlap, preserving sentence boundaries and page numbers.
- Rationale: Preserves semantic and structural context, improving retrieval precision and citation fidelity.
- Trade-off: Slightly higher indexing time and storage for overlap.

### 2) Retrieval Method
- Approach: Cosine similarity search in Qdrant;filtering by `paper_id` or `paper_filenames`. Payload includes rich metadata for downstream citation.
- Rationale: Keeps retrieval simple and fast; payload richness enables grounded answers without extra DB round-trips.
- Trade-off: No learned re-ranking; relies on embedding quality and chunking.

### 3) Prompting
- Approach: Structured prompt instructing grounding only on provided context, mandatory citations with section/page, and emits a confidence line.
- Rationale: Improves instruction adherence and makes confidence explicit for consumers.

### 4) Persistence & History
- Papers, chunks, queries, answers, and citations are normalized in Postgres. Citations attempt FK alignment to `paper_chunks` via `(paper_id, chunk_index)`; otherwise fall back to title mapping.
- History endpoint aggregates `Query` and `QueryAnswer` for quick review; rating endpoint updates `user_rating`.

### 5) Analytics
- KMeans over embeddings of past queries exposes popular semantic topics with representative examples.

---

## 🧪 Testing


**Test Results**:
- [x] All 5 papers ingested successfully (avg: 12 seconds each)
- [x] All API endpoints return proper status codes
- [x] 18/20 test queries return relevant answers
- [x] Citations properly formatted in 100% of responses
- [x] Error handling works for edge cases


## ✨ Bonus Features Implemented

- [x] **Docker Compose** - One-command setup
- [ ] **Unit Tests** - 67% coverage, all core functions tested
- [ ] **Web UI** - Would add if more time
- [ ] **Multi-paper Compare** - Works with paper_ids filter
- [ ] **Caching** - Redis cache for embeddings (30% speedup)
- [x] **Analytics** - Track popular queries, avg response time
---


<!-- rag-venv\scripts\activate
ctrl+shift+p ->select python:interpretor ->select the virtual env
uvicorn src.main:app --reload
docker run -p 6333:6333 qdrant/qdrant
git status
git branch
git remote -v
git add .
git commit -m "message"
git push -u origin submission/rafid-adib

docker compose up --build -d
docker ps

docker compose logs -f
docker compose logs -f api
docker-compose up
docker-compose build -->