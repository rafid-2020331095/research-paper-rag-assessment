## 📝 Pull Request Template

```bash
- Name: MD RAFID ADIB
- Email: rafidadib123@gmail.com
- LinkedIn: YOUR LINKEDIN (optional)
- Time Spent: ~26 hours over 3 days
```
---

## 📝 Implementation Summary

This PR implements an end-to-end FastAPI-based RAG system for academic papers with robust ingestion, section-aware chunking, vector storage in Qdrant, and grounded QA with citations. The flow:

- Upload one or more PDFs via `POST /api/papers/upload`. Each file is persisted under `uploaded_papers/`, parsed with `PyPDF2`, and sectionized using heuristics in `PDFProcessor._identify_sections` driven by canonical `SECTION_PATTERNS` for Abstract/Introduction/Methods/Results/Discussion/Conclusion/References. Text is chunked per section into ~1000-token-equivalent character windows with ~200-token-equivalent overlap, ensuring sentence boundaries and page attribution.
- Embeddings are generated in batches using Sentence Transformers `all-MiniLM-L6-v2` (dim=384). Chunks are stored to Qdrant with payload metadata: `paper_id`, `paper_title`, `content`, `section`, `page_number`, `chunk_index`. Vector ids returned by Qdrant are persisted alongside chunks in Postgres.
- Queries hit `POST /api/query?query=...` with optional `paper_ids` or `paper_filenames` for filtering. A query embedding is computed and used to retrieve top-k similar chunks from Qdrant, with an internal filter on `paper_id`. Retrieved chunks are formatted into a structured context block.
- The LLM (`LLM_TYPE` switchable; defaults to Ollama Cloud through `ollama` client; DeepSeek supported) receives a structured prompt that enforces grounding, citation formatting, and emits a confidence score. The answer is returned with citations built from the retrieved payloads and mapped to stored chunks where possible.
- Query history (text, response time, optional rating) is stored, retrievable via `GET /api/query/history`, and rateable via `POST /api/query/history/{id}/rate`.
- Analytics exposes `GET /api/analytics/popular`, clustering past queries (KMeans over query embeddings) to present popular semantic topics with example queries.

---

## 🛠️ Technology Choices

- **LLM**: [x] Ollama Cloud (default) [x] DeepSeek (optional)
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
- PyPDF2 (PDF extraction)
- LangChain (optional RAG utilities)

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.10+
- Docker (for Qdrant)
- 8GB RAM recommended

### Quick Start (5 minutes)

1) Clone and enter directory
```bash
git clone https://github.com/YOUR_USERNAME/research-paper-rag-assessment.git
cd research-paper-rag-assessment
```

2) Start Qdrant (Docker)
```bash
docker run -p 6333:6333 qdrant/qdrant
```

3) Create venv and install deps
```bash
python -m venv venv
venv\Scripts\activate   # On Windows
pip install -r requirements.txt
```

4) Configure environment
```bash
copy .env.example .env  # Windows
# Update DB and service settings as needed
# DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost/research_papers
# QDRANT_HOST=localhost
# QDRANT_PORT=6333
# QDRANT_COLLECTION=research_papers
# LLM_TYPE=ollama  # or deepseek
# OLLAMA_API_KEY=...
# OLLAMA_MODEL=deepseek-v3.1:671b
# DEEPSEEK_API_KEY=...
# METADATA_USE_LLM=true
# CHARS_PER_TOKEN=4
```

5) Run the API
```bash
uvicorn src.main:app --reload --port 8000
```

6) Smoke test
```bash
REM Upload a paper
curl -X POST "http://localhost:8000/api/papers/upload" ^
  -H "accept: application/json" ^
  -H "Content-Type: multipart/form-data" ^
  -F "files=@sample_papers/paper_1.pdf" ^
  -F "files=@sample_papers/paper_2.pdf"

REM Query it
curl -X POST "http://localhost:8000/api/query?query=What%20methodology%20was%20used%3F" ^
  -H "accept: application/json"
```

Docs: http://localhost:8000/docs

---

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

---

## 🎯 Design Decisions

### 1) Chunking Strategy
- Approach: Section-aware chunking with sentence boundaries; ~1000-token-equivalent chars per chunk, ~200-token-equivalent overlap, carrying section name and page number.
- Rationale: Preserves semantic and structural context, improving retrieval precision and citation fidelity.
- Trade-off: Slightly higher indexing time and storage for overlap.

### 2) Retrieval Method
- Approach: Cosine similarity search in Qdrant; optional filtering by `paper_id` or `paper_filenames` (resolved to ids). Payload includes rich metadata for downstream citation.
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

Checklist:
- [ ] All sample papers ingest successfully; chunks created; vectors stored; filenames persisted without overwrite conflicts
- [ ] `POST /api/papers/upload` returns 201 with list of `PaperResponse` including `download_url`
- [ ] `POST /api/query` returns grounded answers with `citations` and `sources_used`; `confidence` present
- [ ] `GET /api/papers`, `GET /api/papers/{id}`, and `GET /api/papers/{id}/download` function
- [ ] `DELETE /api/papers?ids=...` bulk-deletes files, DB rows, and Qdrant vectors
- [ ] `GET /api/query/history` returns latest queries with optional answers
- [ ] `POST /api/query/history/{id}/rate?rating=1..5` updates rating
- [ ] `GET /api/analytics/popular` returns topics when queries exist
- [ ] Errors return meaningful messages and appropriate status codes

Optional Metrics:
- Ingestion time per paper: ~10-20s (machine dependent)
- Top-k retrieval relevance (manual spot-check): ≥4/5 good hits

---

## ✨ Bonus Features Implemented

- [ ] Docker Compose (one-command setup)
- [ ] Unit tests (coverage: x%)
- [ ] Web UI
- [x] Multi-paper support (filter by `paper_ids` or filenames)
- [ ] Caching (e.g., Redis for embeddings)
- [x] Analytics (popular topics from query history)

---

## 📎 Screenshots / Logs (optional)

Attach brief logs, screenshots of Swagger, or test outputs if relevant.

---

## 🔁 Related Issues

Closes: #<issue-id> (if applicable)

---

## ✅ Reviewer Checklist (for maintainers)

- [ ] Builds and runs locally with steps above
- [ ] API endpoints function per README
- [ ] Code is clear, modular, and documented
- [ ] No secrets committed; `.env.example` updated if needed
- [ ] PR description complete and accurate