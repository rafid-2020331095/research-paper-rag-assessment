import os
import logging
import tempfile
from typing import List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import delete

from src.models.database import get_db
from src.models.models import Paper, PaperCreate, PaperResponse
from src.services.pdf_processor import PDFProcessor
from src.services.embedding_service import get_embedding_service
from src.services.qdrant_client import get_qdrant_service

router = APIRouter(prefix="/api/papers", tags=["papers"])
logger = logging.getLogger(__name__)

@router.post("/upload", response_model=List[PaperResponse], status_code=status.HTTP_201_CREATED)
async def upload_paper(
    files: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
    pdf_processor: PDFProcessor = Depends(PDFProcessor),
    embedding_service = Depends(get_embedding_service),
    qdrant_service = Depends(get_qdrant_service)
):
    """
    Upload one or more research paper PDFs, process them, and store in the database and vector store.
    """
    results = []
    # Determine persistent upload directory
    upload_dir = os.getenv("UPLOAD_DIR", os.path.join(os.getcwd(), "uploaded_papers"))
    os.makedirs(upload_dir, exist_ok=True)
    
    for file in files:
        temp_file_path = None
        
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Only PDF files are supported. Invalid file: {file.filename}"
            )
        
        try:
            # Save uploaded file to persistent location
            original_filename = file.filename
            persistent_path = os.path.join(upload_dir, original_filename)
            # Avoid overwriting existing files by adding a numeric suffix if needed
            if os.path.exists(persistent_path):
                name, ext = os.path.splitext(original_filename)
                counter = 1
                while True:
                    candidate = f"{name} ({counter}){ext}"
                    candidate_path = os.path.join(upload_dir, candidate)
                    if not os.path.exists(candidate_path):
                        persistent_path = candidate_path
                        original_filename = candidate
                        break
                    counter += 1

            content = await file.read()
            with open(persistent_path, "wb") as out_f:
                out_f.write(content)

            # Extract text and sections first
            full_text, sections = await pdf_processor.extract_text_from_pdf(persistent_path)

            # Extract metadata using both file_path and full_text
            metadata = await pdf_processor.extract_metadata(persistent_path, full_text)

            # Create paper record (ensure non-null file_path)
            paper = Paper(
                title=metadata.get('title', file.filename),
                authors=metadata.get('authors', ''),
                year=metadata.get('year'),
                filename=original_filename,
                file_path=persistent_path
            )
            
            db.add(paper)
            await db.commit()
            await db.refresh(paper)
            
            # Chunk text with sections
            chunks = await pdf_processor.chunk_text(full_text, sections)
            
            # Generate embeddings
            chunk_texts = [chunk['content'] for chunk in chunks]
            embeddings = await embedding_service.generate_embeddings(chunk_texts)

            # Prepare metadata list aligned with embeddings
            metadata_list = []
            for chunk in chunks:
                metadata_list.append({
                    "paper_id": paper.id,
                    "paper_title": paper.title,
                    "content": chunk['content'],
                    "section": chunk.get('section'),
                    "page_number": chunk.get('page_number'),
                    "chunk_index": chunk.get('chunk_index')
                })

            # Store in Qdrant in batch
            await qdrant_service.store_embeddings(embeddings=embeddings, metadata=metadata_list)
            
            logger.info(f"Successfully processed paper: {paper.title} with {len(chunks)} chunks")
            
            results.append(PaperResponse(
                id=paper.id,
                title=paper.title,
                authors=paper.authors,
                year=paper.year,
                filename=paper.filename,
                uploaded_at=paper.uploaded_at
            ))
        
        except Exception as e:
            logger.error(f"Error processing paper {file.filename}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing paper {file.filename}: {str(e)}"
            )
        finally:
            # No temp file cleanup needed; we persist the uploaded file
            pass
    
    return results

@router.get("", response_model=List[PaperResponse])
async def list_papers(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """
    List all papers with pagination.
    """
    query = select(Paper).offset(skip).limit(limit).order_by(Paper.uploaded_at.desc())
    result = await db.execute(query)
    papers = result.scalars().all()
    
    return [
        PaperResponse(
            id=paper.id,
            title=paper.title,
            authors=paper.authors,
            year=paper.year,
            filename=paper.filename,
            uploaded_at=paper.uploaded_at
        ) for paper in papers
    ]

@router.get("/{paper_id}", response_model=PaperResponse)
async def get_paper(
    paper_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific paper by ID.
    """
    query = select(Paper).where(Paper.id == paper_id)
    result = await db.execute(query)
    paper = result.scalars().first()
    
    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper with ID {paper_id} not found"
        )
    
    return PaperResponse(
        id=paper.id,
        title=paper.title,
        authors=paper.authors,
        year=paper.year,
        filename=paper.filename,
        uploaded_at=paper.uploaded_at
    )

@router.delete("/{paper_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_paper(
    paper_id: int,
    db: AsyncSession = Depends(get_db),
    qdrant_service = Depends(get_qdrant_service)
):
    """
    Delete a paper and its associated embeddings.
    """
    # Check if paper exists
    query = select(Paper).where(Paper.id == paper_id)
    result = await db.execute(query)
    paper = result.scalars().first()
    
    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper with ID {paper_id} not found"
        )
    
    # Delete from database
    await db.execute(delete(Paper).where(Paper.id == paper_id))
    await db.commit()
    
    # Delete from vector store
    await qdrant_service.delete_by_paper_id(paper_id)
    
    return None