# Research Paper RAG System: Implementation Approach

This document outlines the key design decisions and implementation strategies for the Research Paper RAG system.

## Chunking Strategy

We implemented a section-aware chunking strategy that:

1. **Preserves Section Context**: Each chunk includes section information (Abstract, Introduction, Methods, Results, Discussion, Conclusion)
2. **Maintains Semantic Coherence**: Chunks are created with overlapping text to preserve context across chunk boundaries
3. **Balanced Size**: Chunks are sized to balance between:
   - Small enough for precise retrieval (300-500 tokens)
   - Large enough to contain complete thoughts

This approach ensures that when retrieving chunks for a query, we maintain the original paper's structure and context, making citations more accurate.

## Embedding Model Choice

We selected the `all-MiniLM-L6-v2` model from Sentence-Transformers for several reasons:

1. **Efficiency**: Produces 384-dimensional embeddings with excellent performance-to-size ratio
2. **Speed**: Fast inference suitable for production environments
3. **Semantic Understanding**: Well-suited for academic text with good performance on technical content
4. **Resource Requirements**: Lightweight enough to run on standard hardware

This model provides a good balance between quality and performance for academic paper retrieval.

## RAG Pipeline Design

Our RAG pipeline follows these steps:

1. **Query Understanding**: Convert user query to embedding vector
2. **Retrieval**: Find most similar chunks in Qdrant with optional paper filtering
3. **Context Assembly**: Format retrieved chunks with metadata (paper title, section, page)
4. **Prompt Engineering**: Construct a prompt that:
   - Instructs the LLM to answer based ONLY on provided context
   - Requires citations to specific papers, sections, and pages
   - Asks for a confidence score
5. **Answer Generation**: Use Ollama or DeepSeek to generate the final response

### Prompt Engineering Approach

Our prompt is designed to:

```
You are a research assistant helping with academic papers. Answer the question based ONLY on the provided context.
If you cannot answer the question based on the context, say "I don't have enough information to answer this question."

CONTEXT:
[Document details and content]

QUESTION:
[User query]

INSTRUCTIONS:
1. Answer the question concisely and accurately based only on the provided context.
2. Cite specific papers, sections, and page numbers in your answer.
3. Do not make up information or use knowledge outside the provided context.
4. If the context contains conflicting information, acknowledge this in your answer.
5. Format your answer in a clear, structured way.
6. At the end, include a confidence score between 0.0 and 1.0 on a new line in this format: "CONFIDENCE: 0.X"
```

This prompt structure ensures:
- Answers are grounded in the provided context
- Citations are specific and traceable
- The system acknowledges uncertainty when appropriate

## Database Schema Design

We designed the database schema to efficiently store:

1. **Papers**: Metadata about uploaded papers (title, authors, year)
2. **Paper Chunks**: Segments of papers with section and page information
3. **Queries**: User questions and response metrics
4. **Citations**: Tracking which papers were referenced in answers

The schema supports:
- Efficient paper retrieval and management
- Query history analysis
- Citation tracking for analytics

## Async Implementation

The entire system is built with asynchronous operations:

1. **FastAPI**: Async request handling
2. **Database**: AsyncPG with SQLAlchemy async ORM
3. **PDF Processing**: Async methods for text extraction and chunking
4. **Embedding Generation**: Batched async processing
5. **LLM Calls**: Async HTTP requests to Ollama/DeepSeek

This approach ensures high throughput and responsiveness, even under load.

## Trade-offs and Limitations

1. **PDF Extraction Quality**: PDF extraction can be imperfect, especially with complex layouts or equations
2. **Embedding Model Size**: Chose a smaller model for speed, sacrificing some semantic understanding
3. **Local LLM Limitations**: Ollama models may have lower quality than cloud alternatives
4. **Citation Precision**: Citations are based on chunk boundaries, which may not perfectly align with paper structure
5. **Query Complexity**: Very complex queries spanning multiple papers may require refinement

## Future Improvements

1. **Hybrid Search**: Combine vector search with keyword search for better retrieval
2. **Improved PDF Extraction**: Better handling of tables, figures, and equations
3. **Multi-stage Retrieval**: Implement re-ranking of initial search results
4. **User Feedback Loop**: Incorporate user ratings to improve retrieval and answer quality
5. **Caching**: Implement response caching for common queries