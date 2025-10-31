import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from src.models.database import get_db
from src.models.models import Query as QueryModel, QueryPaper, QueryResponse, QueryHistoryResponse, Paper, QueryAnswer, Citation, PaperChunk, CitationResponse
from src.services.rag_pipeline import get_rag_pipeline

router = APIRouter(prefix="/api/query", tags=["query"])
logger = logging.getLogger(__name__)

@router.post("", response_model=QueryResponse)
async def query_papers(
    query: str = Query(..., min_length=3),
    paper_ids: Optional[List[int]] = Query(None),
    paper_filenames: Optional[List[str]] = Query(None, description="Filter by uploaded filenames."),
    top_k: int = Query(5, ge=1, le=20),
    db: AsyncSession = Depends(get_db),
    rag_pipeline = Depends(get_rag_pipeline)
):
    """
    Query papers using RAG pipeline and return answer with citations.
    """
    try:
        # Resolve filenames to IDs if provided
        if paper_filenames:
            result = await db.execute(select(Paper).where(Paper.filename.in_(paper_filenames)))
            matched = result.scalars().all()
            resolved_ids = [p.id for p in matched]
            paper_ids = resolved_ids if paper_ids is None else list(set(paper_ids) | set(resolved_ids))

        # Process query through RAG pipeline
        response, response_time_ms = await rag_pipeline.process_query(
            query=query,
            top_k=top_k,
            paper_ids=paper_ids
        )
        
        # Store query in database
        query_record = QueryModel(
            text=query,
            response_time_ms=response_time_ms
        )
        db.add(query_record)
        await db.commit()
        await db.refresh(query_record)
        
        # Store model answer for history
        db.add(QueryAnswer(query_id=query_record.id, answer=response.answer))

        # Store paper references (map titles to IDs)
        if response.sources_used:
            for paper_title in response.sources_used:
                result = await db.execute(select(Paper).where(Paper.title == paper_title))
                paper_obj = result.scalars().first()
                if paper_obj:
                    query_paper = QueryPaper(
                        query_id=query_record.id,
                        paper_id=paper_obj.id
                    )
                    db.add(query_paper)
        # Store citations with proper FKs when possible
        if response.citations:
            for c in response.citations:
                # Defensive: convert to CitationResponse object if not already
                if isinstance(c, tuple):
                    c = CitationResponse(*c)
                elif isinstance(c, dict):
                    c = CitationResponse(**c)
                paper_id = None
                chunk_id = None
                if getattr(c, 'paper_id', None) is not None and getattr(c, 'chunk_index', None) is not None:
                    paper_id = c.paper_id
                    # Resolve chunk by (paper_id, chunk_index)
                    result = await db.execute(
                        select(PaperChunk).where(PaperChunk.paper_id == paper_id, PaperChunk.chunk_index == c.chunk_index)
                    )
                    chunk = result.scalars().first()
                    if chunk:
                        chunk_id = chunk.id
                # Fallback: map by title to paper_id
                if paper_id is None:
                    result = await db.execute(select(Paper).where(Paper.title == c.paper_title))
                    paper = result.scalars().first()
                    paper_id = paper.id if paper else None
                # Only persist citation if we have a valid chunk_id to satisfy FK
                if paper_id is not None and chunk_id is not None:
                    section_val = c.section[:100] if isinstance(c.section, str) else c.section
                    db.add(Citation(
                        query_id=query_record.id,
                        paper_id=paper_id,
                        chunk_id=chunk_id,
                        section=section_val,
                        page=c.page,
                        relevance_score=c.relevance_score,
                    ))

        await db.commit()
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )

@router.get("/history", response_model=List[QueryHistoryResponse])
async def get_query_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """
    Get query history with pagination.
    """
    query = select(QueryModel).offset(skip).limit(limit).order_by(QueryModel.created_at.desc())
    result = await db.execute(query)
    queries = result.scalars().all()
    
    history = []
    for q in queries:
        # Fetch answer if exists
        ans_result = await db.execute(select(QueryAnswer).where(QueryAnswer.query_id == q.id))
        qa = ans_result.scalars().first()
        history.append(QueryHistoryResponse(
            id=q.id,
            text=q.text,
            created_at=q.created_at,
            response_time_ms=q.response_time_ms,
            user_rating=q.user_rating,
            answer=qa.answer if qa else None
        ))
    return history

@router.post("/history/{query_id}/rate", status_code=status.HTTP_204_NO_CONTENT)
async def rate_query(
    query_id: int,
    rating: int = Query(..., ge=1, le=5),
    db: AsyncSession = Depends(get_db)
):
    """
    Rate a previous query from 1-5 stars.
    """
    query = select(QueryModel).where(QueryModel.id == query_id)
    result = await db.execute(query)
    query_record = result.scalars().first()
    
    if not query_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Query with ID {query_id} not found"
        )
    
    query_record.user_rating = rating
    await db.commit()
    
    return None