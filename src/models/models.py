import datetime
from typing import List, Optional
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field

from src.models.database import Base

# SQLAlchemy Models
class Paper(Base):
    """SQLAlchemy model for research papers."""
    __tablename__ = "papers"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    authors = Column(String(255), nullable=False)
    year = Column(Integer, nullable=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(255), nullable=True)
    uploaded_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    chunks = relationship("PaperChunk", back_populates="paper", cascade="all, delete-orphan")
    queries = relationship("Query", secondary="query_papers", back_populates="papers")

class PaperChunk(Base):
    """SQLAlchemy model for paper chunks (sections of text)."""
    __tablename__ = "paper_chunks"

    id = Column(Integer, primary_key=True, index=True)
    paper_id = Column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    content = Column(Text, nullable=False)
    section = Column(String(100), nullable=True)
    page_number = Column(Integer, nullable=True)
    chunk_index = Column(Integer, nullable=False)
    vector_id = Column(String(255), nullable=False)  # ID in Qdrant
    
    # Relationships
    paper = relationship("Paper", back_populates="chunks")

class Query(Base):
    """SQLAlchemy model for user queries."""
    __tablename__ = "queries"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    response_time_ms = Column(Float, nullable=True)
    user_rating = Column(Integer, nullable=True)
    
    # Relationships
    papers = relationship("Paper", secondary="query_papers", back_populates="queries")
    citations = relationship("Citation", back_populates="query", cascade="all, delete-orphan")

class QueryAnswer(Base):
    """Stores the generated answer text for a query."""
    __tablename__ = "query_answers"

    query_id = Column(Integer, ForeignKey("queries.id", ondelete="CASCADE"), primary_key=True)
    answer = Column(Text, nullable=False)

class QueryPaper(Base):
    """Association table for queries and papers."""
    __tablename__ = "query_papers"

    query_id = Column(Integer, ForeignKey("queries.id", ondelete="CASCADE"), primary_key=True)
    paper_id = Column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True)

class Citation(Base):
    """SQLAlchemy model for citations in query responses."""
    __tablename__ = "citations"

    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer, ForeignKey("queries.id", ondelete="CASCADE"), nullable=False)
    paper_id = Column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    chunk_id = Column(Integer, ForeignKey("paper_chunks.id", ondelete="CASCADE"), nullable=False)
    section = Column(String(100), nullable=True)
    page = Column(Integer, nullable=True)
    relevance_score = Column(Float, nullable=True)
    
    # Relationships
    query = relationship("Query", back_populates="citations")
    paper = relationship("Paper")
    chunk = relationship("PaperChunk")

# Pydantic Models for API
class PaperBase(BaseModel):
    title: str
    authors: str
    year: Optional[int] = None

class PaperCreate(PaperBase):
    pass

class PaperResponse(PaperBase):
    id: int
    filename: str
    uploaded_at: datetime.datetime

    class Config:
        orm_mode = True

class QueryBase(BaseModel):
    question: str
    top_k: int = 5
    paper_ids: Optional[List[int]] = None

class CitationResponse(BaseModel):
    paper_title: str
    section: Optional[str] = None
    page: Optional[int] = None
    relevance_score: float

class QueryResponse(BaseModel):
    answer: str
    citations: List[CitationResponse]
    sources_used: List[str]
    confidence: float

class QueryHistoryResponse(BaseModel):
    id: int
    text: str
    created_at: datetime.datetime
    response_time_ms: Optional[float] = None
    user_rating: Optional[int] = None
    answer: Optional[str] = None

    class Config:
        orm_mode = True