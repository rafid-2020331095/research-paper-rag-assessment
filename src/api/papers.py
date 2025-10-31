import os
import logging
import tempfile
import re
import unicodedata
from typing import List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import delete

from src.models.database import get_db
from src.models.models import Paper, PaperCreate, PaperResponse, PaperChunk
from src.services.pdf_processor import PDFProcessor
from src.services.embedding_service import get_embedding_service
from src.services.qdrant_client import get_qdrant_service
from src.services.metadata_llm import get_llm_metadata_extractor

router = APIRouter(prefix="/api/papers", tags=["papers"])
logger = logging.getLogger(__name__)


def _sanitize_text(value: str) -> str:
    """Make text safe for Postgres TEXT: remove null bytes and invalid unicode/control chars."""
    if not isinstance(value, str):
        return ""
    # Replace null bytes and normalize
    text = value.replace("\x00", " ")
    # Remove most control characters except newline, tab, carriage return
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", " ", text)
    # Normalize unicode and drop undecodable bytes
    text = unicodedata.normalize("NFC", text)
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    return text

@router.post("/upload", response_model=List[PaperResponse], status_code=status.HTTP_201_CREATED)
async def upload_paper(
    files: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
    pdf_processor: PDFProcessor = Depends(PDFProcessor),
    embedding_service = Depends(get_embedding_service),
    qdrant_service = Depends(get_qdrant_service),
    llm_metadata_extractor = Depends(get_llm_metadata_extractor)
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

            # Extract metadata using LLM (with fallback inside)
            metadata = await pdf_processor.extract_metadata(persistent_path, full_text, llm_extractor=llm_metadata_extractor)

            # Create paper record (ensure non-null file_path)
            raw_title = metadata.get('title', file.filename)
            raw_authors = metadata.get('authors', '')
            title_val = (raw_title if isinstance(raw_title, str) else str(raw_title))[:255]
            authors_val = (raw_authors if isinstance(raw_authors, str) else str(raw_authors))[:255]
            paper = Paper(
                title=title_val,
                authors=authors_val,
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

            # Store in Qdrant in batch and capture point IDs
            point_ids = await qdrant_service.store_embeddings(embeddings=embeddings, metadata=metadata_list)

            # Persist chunks in relational DB aligned to Qdrant IDs
            for idx, chunk in enumerate(chunks):
                vector_id = point_ids[idx] if idx < len(point_ids) else ""
                section_val = chunk.get('section')
                if isinstance(section_val, str):
                    section_val = section_val[:100]
                content_val = _sanitize_text(chunk.get('content', ''))
                db.add(PaperChunk(
                    paper_id=paper.id,
                    content=content_val,
                    section=section_val,
                    page_number=chunk.get('page_number'),
                    chunk_index=chunk.get('chunk_index'),
                    vector_id=vector_id
                ))
            await db.commit()
            
            logger.info(f"Successfully processed paper: {paper.title} with {len(chunks)} chunks")
            
            results.append(PaperResponse(
                id=paper.id,
                title=paper.title,
                authors=paper.authors,
                year=paper.year,
                filename=paper.filename,
                uploaded_at=paper.uploaded_at,
                file_path=paper.file_path,
                download_url=f"/api/papers/{paper.id}/download"
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
            uploaded_at=paper.uploaded_at,
            file_path=paper.file_path,
            download_url=f"/api/papers/{paper.id}/download"
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
        uploaded_at=paper.uploaded_at,
        file_path=paper.file_path,
        download_url=f"/api/papers/{paper.id}/download"
    )

# @router.delete("/{paper_id}", status_code=status.HTTP_204_NO_CONTENT)
# async def delete_paper(
#     paper_id: int,
#     db: AsyncSession = Depends(get_db),
#     qdrant_service = Depends(get_qdrant_service)
# ):
#     """
#     Delete a paper and its associated embeddings.
#     """
#     # Check if paper exists
#     query = select(Paper).where(Paper.id == paper_id)
#     result = await db.execute(query)
#     paper = result.scalars().first()
    
#     if not paper:
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND,
#             detail=f"Paper with ID {paper_id} not found"
#         )
    
#     # Delete file from filesystem if present
#     try:
#         if paper.file_path and os.path.exists(paper.file_path):
#             os.remove(paper.file_path)
#     except Exception as fe:
#         logger.warning(f"Failed to remove file for paper_id={paper_id}: {fe}")

#     # Delete from database
#     await db.execute(delete(Paper).where(Paper.id == paper_id))
#     await db.commit()
    
#     # Delete from vector store
#     await qdrant_service.delete_by_paper_id(paper_id)
    
#     return None


@router.get("/{paper_id}/download")
async def download_paper(
    paper_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Stream-download the stored PDF file for a paper by ID.
    """
    query = select(Paper).where(Paper.id == paper_id)
    result = await db.execute(query)
    paper = result.scalars().first()
    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper with ID {paper_id} not found"
        )
    if not paper.file_path or not os.path.exists(paper.file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found on server"
        )
    return FileResponse(path=paper.file_path, filename=paper.filename, media_type="application/pdf")

@router.delete("", status_code=status.HTTP_200_OK)
async def bulk_delete_papers(
    ids: List[int] = Query(..., description="Repeat the query param for multiple IDs: ?ids=1&ids=2"),
    db: AsyncSession = Depends(get_db),
    qdrant_service = Depends(get_qdrant_service)
):
    """
    Delete multiple papers and their associated embeddings and files.
    Example: DELETE /api/papers?ids=1&ids=2&ids=3
    """
    if not ids:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No paper IDs provided")

    # Fetch all requested papers
    result = await db.execute(select(Paper).where(Paper.id.in_(ids)))
    papers = result.scalars().all()

    found_ids = {p.id for p in papers}
    not_found_ids = [pid for pid in ids if pid not in found_ids]

    # Remove files and qdrant vectors per paper
    for p in papers:
        try:
            if p.file_path and os.path.exists(p.file_path):
                os.remove(p.file_path)
        except Exception as fe:
            logger.warning(f"Failed to remove file for paper_id={p.id}: {fe}")
        try:
            await qdrant_service.delete_by_paper_id(p.id)
        except Exception as qe:
            logger.warning(f"Failed to delete vectors for paper_id={p.id}: {qe}")

    # Bulk delete from DB
    await db.execute(delete(Paper).where(Paper.id.in_(list(found_ids))))
    await db.commit()

    return {"deleted_ids": list(found_ids), "not_found_ids": not_found_ids}
