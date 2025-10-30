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

**Your mission**: Build an intelligent assistant that does this in seconds.

---

## 🛠️ Required Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| **Vector DB** | Qdrant | Fast similarity search |
| **Database** | PostgreSQL/MySQL/MongoDB | Metadata & query history |
| **LLM** | Ollama OR DeepSeek | Answer generation |
| **Embeddings** | sentence-transformers | Text vectorization |
| **Backend** | Python + FastAPI/Flask Or Any Other Language | API service |

---

## 📋 Features to Implement

### ✅ Must-Have Features

#### 1. Document Ingestion System
```python
POST /api/papers/upload
```
- Accept PDF research papers
- Extract text with section awareness (Abstract, Intro, Methods, Results, Conclusion)
- Intelligent chunking (preserve semantic context)
- Generate embeddings
- Store vectors in Qdrant with metadata
- Save paper info in database

**Expected Behavior**:
- Handle multi-page PDFs
- Extract author names, title, year
- Store page numbers for citations
- Process 5 papers in < 2 minutes

#### 2. Intelligent Query System
```python
POST /api/query
{
  "question": "What methodology was used in the transformer paper?",
  "top_k": 5,
  "paper_ids": [1, 3]  // optional: limit to specific papers
}
```

**Response Format**:
```json
{
  "answer": "The transformer paper uses a self-attention mechanism...",
  "citations": [
    {
      "paper_title": "Attention is All You Need",
      "section": "Methodology",
      "page": 3,
      "relevance_score": 0.89
    }
  ],
  "sources_used": ["paper3_nlp_transformers.pdf"],
  "confidence": 0.85
}
```

#### 3. Paper Management
```python
GET    /api/papers              # List all papers
GET    /api/papers/{id}         # Get paper details
DELETE /api/papers/{id}         # Remove paper + vectors
GET    /api/papers/{id}/stats   # View/download stats
```

#### 4. Query History & Analytics
```python
GET /api/queries/history         # Recent queries
GET /api/analytics/popular       # Most queried topics
```

Store:
- Query text
- Papers referenced
- Response time
- User satisfaction (optional rating)

---

## 🏗️ System Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│      FastAPI Application        │
│  ┌───────────────────────────┐  │
│  │   Document Processor      │  │
│  │  - PDF extraction         │  │
│  │  - Chunking strategy      │  │
│  │  - Embedding generation   │  │
│  └───────────────────────────┘  │
│                                  │
│  ┌───────────────────────────┐  │
│  │   RAG Pipeline            │  │
│  │  - Query understanding    │  │
│  │  - Vector retrieval       │  │
│  │  - Context assembly       │  │
│  │  - LLM generation         │  │
│  └───────────────────────────┘  │
└────┬──────────────────┬─────────┘
     │                  │
     ▼                  ▼
┌─────────┐      ┌──────────────┐
│ Qdrant  │      │ PostgreSQL/  │
│ Vector  │      │ MySQL        │
│ Store   │      │ (Metadata)   │
└─────────┘      └──────────────┘
     │
     ▼
┌─────────────┐
│ Ollama/     │
│ DeepSeek    │
│ (LLM)       │
└─────────────┘
```

---

## 📊 Test Dataset

**5 Sample Papers Provided** (in `sample_papers/` directory):

1. `paper1_machine_learning.pdf` - Classic ML algorithms
2. `paper2_neural_networks.pdf` - Deep learning architectures
3. `paper3_nlp_transformers.pdf` - Transformer models
4. `paper4_computer_vision.pdf` - CNN and vision models
5. `paper5_reinforcement_learning.pdf` - RL algorithms

**20 Test Queries** provided in `test_queries.json` covering:
- Single-paper queries (easy)
- Multi-paper comparisons (medium)
- Abstract concept queries (hard)

---

## 🚀 Getting Started

### Prerequisites
```bash
# Python 3.10+
python --version

# Docker (for Qdrant)
docker --version

# Ollama (if using local LLM)
curl https://ollama.ai/install.sh | sh
```

### Quick Setup

1. **Fork this repository**
   ```bash
   # Click "Fork" button on GitHub
   ```

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/research-paper-rag-assessment.git
   cd research-paper-rag-assessment
   ```

3. **Create working branch**
   ```bash
   git checkout -b submission/YOUR_NAME
   ```

4. **Set up environment**
   ```bash
   # Start Qdrant
   docker run -p 6333:6333 qdrant/qdrant
   
   # Install Ollama and pull model
   ollama pull llama3
   # OR set up DeepSeek API key
   
   # Create Python virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

5. **Start building!** 🎉

---

## 📦 Submission Requirements

### Your Repo Structure Should Look Like:
```
your-fork/
├── src/
│   ├── main.py                 # FastAPI app
│   ├── models/                 # Data models
│   ├── services/
│   │   ├── pdf_processor.py
│   │   ├── embedding_service.py
│   │   ├── qdrant_client.py
│   │   └── rag_pipeline.py
│   ├── api/
│   │   └── routes.py
│   └── config.py
├── tests/                      # Unit tests (bonus)
├── requirements.txt
├── .env.example
├── docker-compose.yml          # (optional)
├── README.md                   # YOUR documentation
├── APPROACH.md                 # Design decisions
└── architecture.png            # System diagram
```

### Must Include:

1. **README.md** with:
   - Clear setup instructions (step-by-step)
   - How to run the application
   - API endpoint documentation
   - Example curl commands or Postman collection
   - Architecture explanation

2. **APPROACH.md** explaining:
   - Chunking strategy and why
   - Embedding model choice
   - Prompt engineering approach
   - Database schema design
   - Trade-offs and limitations

3. **Working Code** that:
   - Processes all 5 sample papers
   - Answers test queries accurately
   - Includes error handling
   - Has proper logging

4. **Configuration**:
   - `.env.example` (no secrets!)
   - `requirements.txt` (complete)

---

## ✅ Self-Check Before Submission

- [ ] Can upload and process PDFs
- [ ] Query endpoint returns relevant answers
- [ ] Citations include paper name + section/page
- [ ] All 5 papers successfully indexed
- [ ] Tested with queries from `test_queries.json`
- [ ] API returns proper error messages
- [ ] README has complete setup instructions
- [ ] No hardcoded paths or credentials
- [ ] Code is clean and commented
- [ ] Logs to console/file

---

## 🎯 Evaluation Criteria

| Category | Weight | Key Points |
|----------|--------|------------|
| **Functionality** | 35% | Features work, edge cases handled |
| **RAG Quality** | 25% | Relevant retrieval, accurate answers, citations |
| **Code Quality** | 20% | Clean, modular, error handling |
| **Documentation** | 10% | Clear setup, architecture explained |
| **API Design** | 10% | RESTful, proper validation |
| **Bonus** | +15% | Tests, Docker, UI, extras |

**Total**: 100 points + 15 bonus = 115 possible

### Scoring:
- **90+**: Exceptional - Strong hire ⭐
- **75-89**: Good - Hire with mentoring ✅
- **60-74**: Borderline - Discussion needed ⚠️
- **<60**: Does not meet requirements ❌

---

## 🏆 Bonus Features (Optional)

Impress us with:
- ✨ **Docker Compose** - One command setup
- 🧪 **Unit Tests** - >60% coverage
- 🎨 **Simple Web UI** - Upload & query interface
- 🔄 **Multi-paper Compare** - Side-by-side analysis
- ⚡ **Caching** - Speed up repeat queries
- 📊 **Analytics Dashboard** - Query insights
- 🔒 **Authentication** - Basic API keys
- 📝 **Export Results** - Save as PDF/Markdown

---

## ⏱️ Timeline

**Recommended**: 3 daus

- **Day 1**: Setup + PDF processing + Qdrant integration
- **Day 2**: Database schema + embeddings + basic API
- **Day 3**: RAG pipeline + LLM integration
- **Day 4**: Testing + documentation + refinement
- **Day 5**: Bonus features + final polish

**Submission Deadline**: [31st October 2025]

---

## 📨 How to Submit

1. **Push your code** to your fork
2. **Create Pull Request** to this repo's `master` branch
3. **Fill out PR template** completely
4. **Wait for review** (we'll respond within 3 business days)

Detailed submission guide: See [SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md)

Detailed Pull request guide: See [PULL_REQUEST_TEMPLATE.md]()

---

## ❓ Need Help?

**Technical Questions**:
- Open an issue with `question` label
- We respond within 24 hours (weekdays)

**Submission Issues**:
- Check [SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md)
- Check [PULL_REQUEST_TEMPLATE.md](PULL_REQUEST_TEMPLATE.md)
- Email: [ishmam.abid5422@gmail.com]

**Clarifications**:
- Don't assume - ask!
- We prefer over-communication

---

## 🎓 What Happens After Submission?

1. ✅ **Code Review** (2-3 days)
   - Automated tests run
   - Manual code review
   - Documentation check

2. 📞 **Technical Interview** (1 hour)
   - Discuss your solution
   - Architecture deep-dive
   - Potential improvements
   - Scaling scenarios

3. 🎉 **Decision** (within 1 week)

---

## 🌟 Tips for Success

**DO**:
- ✅ Start simple, then enhance
- ✅ Test with provided papers/queries
- ✅ Write clear documentation
- ✅ Handle errors gracefully
- ✅ Explain your decisions
- ✅ Ask questions if unclear

**DON'T**:
- ❌ Hardcode credentials
- ❌ Copy-paste without understanding
- ❌ Skip error handling
- ❌ Ignore the test dataset
- ❌ Submit without testing
- ❌ Overcomplicate unnecessarily

---

## 📚 Helpful Resources

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Ollama Documentation](https://ollama.ai/docs)
- [LangChain RAG Guide](https://python.langchain.com/docs/use_cases/question_answering/)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [Sentence Transformers](https://www.sbert.net/)

---

## 🤝 Good Luck!

We're excited to see your solution! This assessment reflects real work you'd do as a Junior AI Engineer on our team. Show us your problem-solving skills, code quality, and passion for AI.

Remember: We're not looking for perfection - we're looking for potential, clear thinking, and solid fundamentals.

**Questions? Open an issue!**
**Ready? Fork and start building!** 🚀

---
# Research Paper RAG System

A production-ready Retrieval-Augmented Generation (RAG) system for academic research papers.

## Features

- **Document Upload**: Upload PDF research papers, extract text with section awareness
- **Intelligent Querying**: Ask natural language questions about papers
- **Paper Management**: CRUD operations for managing papers
- **Query History**: Track and analyze query performance

## Tech Stack

- **FastAPI**: Async Python web framework
- **Qdrant**: Vector database for embeddings
- **NeonDB (PostgreSQL)**: Metadata and query history storage
- **Sentence-Transformers**: Text embedding generation
- **Ollama/DeepSeek**: LLM for answer generation

## Setup

1. **Clone the repository**

2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   - Copy `.env.example` to `.env` and update with your settings
   - Set database credentials, Qdrant URL, and LLM configuration

4. **Start Qdrant**
   - Using Docker: `docker run -p 6333:6333 qdrant/qdrant`
   - Or install locally: [Qdrant Installation](https://qdrant.tech/documentation/install/)

5. **Start Ollama (if using)**
   - Install from [Ollama.ai](https://ollama.ai/)
   - Pull the model: `ollama pull llama3`

6. **Run the application**
   ```
   uvicorn src.main:app --reload
   ```

7. **Access the API**
   - API documentation: http://localhost:8000/docs
   - Health check: http://localhost:8000/

## API Endpoints

### Paper Management

- `POST /api/papers/upload`: Upload a PDF research paper
- `GET /api/papers`: List all papers
- `GET /api/papers/{paper_id}`: Get paper details
- `DELETE /api/papers/{paper_id}`: Delete a paper

### Querying

- `POST /api/query`: Query papers with natural language
- `GET /api/query/history`: Get query history
- `POST /api/query/history/{query_id}/rate`: Rate a query response

## Example Usage

### Uploading a Paper

```bash
curl -X 'POST' \
  'http://localhost:8000/api/papers/upload' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@research_paper.pdf'
```

### Querying Papers

```bash
curl -X 'POST' \
  'http://localhost:8000/api/query?query=What%20are%20the%20main%20findings%20of%20the%20paper?' \
  -H 'accept: application/json'
```

### Query with Paper Filter

```bash
curl -X 'POST' \
  'http://localhost:8000/api/query?query=Explain%20the%20methodology&paper_ids=1&paper_ids=2' \
  -H 'accept: application/json'
```

## Architecture

- **PDF Processing Pipeline**: Extract text → Identify sections → Chunk text
- **Embedding Generation**: Convert text chunks to vector embeddings
- **Vector Search**: Find relevant chunks based on query similarity
- **RAG Pipeline**: Retrieve context → Generate answer with LLM
- **Async Database**: Efficient PostgreSQL integration with SQLAlchemy

uvicorn src.main:app --reload
