import os
import re
import logging
import PyPDF2
from typing import List, Dict, Tuple, Optional
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)

# Section headers commonly found in research papers
SECTION_PATTERNS = {
    "abstract": r"abstract",
    "introduction": r"introduction|^1\.?\s+intro",
    "methodology": r"methodology|method|approach|^3\.?\s+",
    "results": r"results|evaluation|^4\.?\s+",
    "discussion": r"discussion|^5\.?\s+",
    "conclusion": r"conclusion|^6\.?\s+",
    "references": r"references|bibliography"
}

class PDFProcessor:
    """Service for processing PDF research papers."""
    
    @staticmethod
    async def extract_text_from_pdf(file_path: str) -> Tuple[str, Dict[str, List[Tuple[int, str]]]]:
        """
        Extract text from PDF with section awareness.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple containing:
                - Full text of the PDF
                - Dictionary mapping section names to list of (page_num, text) tuples
        """
        # Use asyncio to run CPU-bound PDF processing in a thread pool
        return await asyncio.to_thread(PDFProcessor._extract_text_sync, file_path)
    
    @staticmethod
    def _extract_text_sync(file_path: str) -> Tuple[str, Dict[str, List[Tuple[int, str]]]]:
        """Synchronous implementation of text extraction from PDF."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                num_pages = len(reader.pages)
                
                # Extract text from each page
                pages_text = []
                for i in range(num_pages):
                    page = reader.pages[i]
                    text = page.extract_text()
                    pages_text.append((i + 1, text))  # Store page number (1-indexed) with text
                
                # Identify sections in the document
                sections = PDFProcessor._identify_sections(pages_text)
                
                # Combine all text
                full_text = "\n".join([text for _, text in pages_text])
                
                return full_text, sections
                
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
    
    @staticmethod
    def _identify_sections(pages_text: List[Tuple[int, str]]) -> Dict[str, List[Tuple[int, str]]]:
        """
        Identify sections in the document based on common section headers.
        
        Args:
            pages_text: List of (page_num, text) tuples
            
        Returns:
            Dictionary mapping section names to list of (page_num, text) tuples
        """
        sections = {
            "abstract": [],
            "introduction": [],
            "methodology": [],
            "results": [],
            "discussion": [],
            "conclusion": [],
            "references": [],
            "other": []
        }
        
        current_section = "other"
        
        for page_num, text in pages_text:
            # Check if this page starts a new section
            for section, pattern in SECTION_PATTERNS.items():
                if re.search(pattern, text.lower()[:100]):
                    current_section = section
                    break
            
            # Add text to current section
            sections[current_section].append((page_num, text))
        
        return sections
    
    @staticmethod
    async def chunk_text(text: str, sections: Dict[str, List[Tuple[int, str]]], chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """
        Chunk text into smaller pieces for embedding, preserving section context.
        
        Args:
            text: Full text to chunk
            sections: Dictionary mapping section names to list of (page_num, text) tuples
            chunk_size: Target size of each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of dictionaries with chunk information
        """
        chunks = []
        
        # Process each section separately to maintain context
        for section_name, section_pages in sections.items():
            if not section_pages:
                continue
                
            # Combine all text from this section
            section_text = "\n".join([text for _, text in section_pages])
            
            # Get page numbers for this section
            page_numbers = [page_num for page_num, _ in section_pages]
            min_page = min(page_numbers)
            max_page = max(page_numbers)
            
            # Chunk the section text
            if len(section_text) <= chunk_size:
                # If section is smaller than chunk size, keep it as one chunk
                chunks.append({
                    "content": section_text,
                    "section": section_name,
                    "page_number": min_page,
                    "chunk_index": 0
                })
            else:
                # Split into overlapping chunks
                for i in range(0, len(section_text), chunk_size - overlap):
                    chunk = section_text[i:i + chunk_size]
                    if len(chunk) < 100:  # Skip very small chunks
                        continue
                        
                    # Estimate page number based on position in text
                    position_ratio = i / len(section_text)
                    estimated_page = min_page + int(position_ratio * (max_page - min_page))
                    
                    chunks.append({
                        "content": chunk,
                        "section": section_name,
                        "page_number": estimated_page,
                        "chunk_index": len(chunks)
                    })
        
        return chunks
    
    @staticmethod
    async def extract_metadata(file_path: str, full_text: str) -> Dict:
        """
        Extract metadata from the PDF file and its content.
        
        Args:
            file_path: Path to the PDF file
            full_text: Extracted text from the PDF
            
        Returns:
            Dictionary containing metadata (title, authors, year)
        """
        metadata = {
            "title": "",
            "authors": "",
            "year": None,
            "filename": Path(file_path).name
        }
        
        # Extract title (usually at the beginning of the document)
        title_match = re.search(r'^(.+?)\n', full_text)
        if title_match:
            metadata["title"] = title_match.group(1).strip()
        
        # Extract authors (usually after title)
        authors_match = re.search(r'(?:^|\n)(?:by\s+)?([A-Z][a-z]+(?: [A-Z][a-z]+)*(?:,? (?:and )?[A-Z][a-z]+(?: [A-Z][a-z]+)*)*)', full_text[:500])
        if authors_match:
            metadata["authors"] = authors_match.group(1).strip()
        
        # Extract year (look for a 4-digit number that could be a year)
        year_match = re.search(r'(?:^|\D)(19\d{2}|20\d{2})(?:\D|$)', full_text[:1000])
        if year_match:
            try:
                metadata["year"] = int(year_match.group(1))
            except ValueError:
                pass
        
        return metadata