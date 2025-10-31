import os
import re
import logging
import PyPDF2
from typing import List, Dict, Tuple, Optional
import asyncio
from pathlib import Path
import math

logger = logging.getLogger(__name__)

# Canonical section patterns to map common headers to standard names
SECTION_PATTERNS = {
    "Abstract": r"^\s*abstract\b",
    "Introduction": r"^\s*introduction\b|^\s*1\.?\s+intro",
    "Methods": r"^\s*methods?\b|^\s*methodology\b|^\s*approach\b|^\s*3\.?\s+",
    "Results": r"^\s*results?\b|^\s*evaluation\b|^\s*4\.?\s+",
    "Discussion": r"^\s*discussion\b|^\s*5\.?\s+",
    "Conclusion": r"^\s*conclusions?\b|^\s*6\.?\s+",
    "References": r"^\s*references\b|^\s*bibliography\b"
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
        # We build dynamic sections keyed by header name; no 'other' bucket.
        dynamic_sections: Dict[str, List[Tuple[int, str]]] = {}
        current_header: Optional[str] = None

        for page_num, text in pages_text:
            header_for_page: Optional[str] = None

            # 1) Try canonical mappings first (case-insensitive, look at the first ~200 chars of page)
            head_slice = (text or "")[:200].lower()
            for canonical, pattern in SECTION_PATTERNS.items():
                if re.search(pattern, head_slice):
                    header_for_page = canonical
                    break

            # 2) If not matched, heuristically pick a header as the first short, title-like line
            if header_for_page is None:
                for line in (text or "").splitlines():
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue
                    # Skip very long lines and typical paragraph-like lines
                    if len(line_stripped) > 120 or len(line_stripped.split()) < 2:
                        continue
                    # Prefer lines with Title Case signal
                    has_mixed = any(c.islower() for c in line_stripped) and any(c.isupper() for c in line_stripped)
                    if has_mixed and line_stripped[0].isalpha():
                        header_for_page = line_stripped
                        break

            # 3) If still none, reuse last seen header to avoid 'other'
            if header_for_page is None:
                header_for_page = current_header or "Front Matter"

            current_header = header_for_page
            dynamic_sections.setdefault(header_for_page, []).append((page_num, text))

        return dynamic_sections
    
    @staticmethod
    async def chunk_text(text: str, sections: Dict[str, List[Tuple[int, str]]], chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """
        Chunk text into smaller pieces for embedding, preserving section context.
        
        Args:
            text: Full text to chunk
            sections: Dictionary mapping section names to list of (page_num, text) tuples
            chunk_size: Target size of each chunk (tokens)
            overlap: Number of tokens to overlap between chunks
            
        Returns:
            List of dictionaries with chunk information
        """
        # Configurable conversion from tokens to approximate characters (avoid tokenizer dependency)
        chars_per_token = float(os.getenv("CHARS_PER_TOKEN", "4"))
        max_chars = int(chunk_size * chars_per_token)
        overlap_chars = int(overlap * chars_per_token)

        def split_into_sentences(t: str) -> List[str]:
            # Lightweight sentence splitter that avoids mid-word splits
            if not t:
                return []
            # Normalize whitespace
            cleaned = re.sub(r"\s+", " ", t).strip()
            # Split on . ! ? followed by space and capital letter or end
            parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", cleaned)
            # Fallback if overly coarse
            if len(parts) <= 1:
                parts = cleaned.split(". ")
                parts = [p + ("." if i < len(parts) - 1 and not p.endswith(".") else "") for i, p in enumerate(parts)]
            return [p.strip() for p in parts if p and len(p.strip()) > 0]

        chunks: List[Dict] = []
        chunk_counter = 0

        # Process each section separately to maintain context
        for section_name, section_pages in sections.items():
            if not section_pages:
                continue

            # Build sentence list with page attribution
            sentences_with_pages: List[Tuple[int, str]] = []
            for page_num, page_text in section_pages:
                for sent in split_into_sentences(page_text or ""):
                    # skip tiny artifacts
                    if len(sent) < 20:
                        continue
                    sentences_with_pages.append((page_num, sent))

            if not sentences_with_pages:
                continue

            # Pack sentences into chunks up to max_chars, with overlap at sentence boundaries
            i = 0
            while i < len(sentences_with_pages):
                cur_chars = 0
                cur_sents: List[Tuple[int, str]] = []
                while i < len(sentences_with_pages) and cur_chars + len(sentences_with_pages[i][1]) + 1 <= max_chars:
                    cur_sents.append(sentences_with_pages[i])
                    cur_chars += len(sentences_with_pages[i][1]) + 1
                    i += 1

                if not cur_sents:
                    # Ensure progress even if a single sentence is very long
                    page_n, sent_text = sentences_with_pages[i]
                    cur_sents = [(page_n, sent_text[:max_chars])]
                    i += 1

                # Determine page number as first sentence page in the chunk
                page_number = cur_sents[0][0]
                content = " ".join([s for _, s in cur_sents])

                chunks.append({
                    "content": content,
                    "section": section_name,
                    "page_number": page_number,
                    "chunk_index": chunk_counter
                })
                chunk_counter += 1

                # Overlap: move back based on overlap_chars in sentence units
                if i < len(sentences_with_pages) and overlap_chars > 0:
                    back_chars = 0
                    back_idx = len(cur_sents) - 1
                    while back_idx > 0 and back_chars < overlap_chars:
                        back_chars += len(cur_sents[back_idx][1]) + 1
                        back_idx -= 1
                    # Rewind global index by the number of overlapped sentences
                    rewind_by = len(cur_sents) - 1 - back_idx
                    i = max(0, i - rewind_by)

        return chunks
    
    @staticmethod
    async def extract_metadata(file_path: str, full_text: str, llm_extractor: Optional[object] = None) -> Dict:
        """
        Extract metadata from the PDF file and its content.
        
        Args:
            file_path: Path to the PDF file
            full_text: Extracted text from the PDF
            llm_extractor: Optional LLM-based extractor dependency
            
        Returns:
            Dictionary containing metadata (title, authors, year)
        """
        metadata = {
            "title": "",
            "authors": "",
            "year": None,
            "filename": Path(file_path).name
        }
        
        # Prefer LLM-based extraction if available and enabled
        use_llm = os.getenv("METADATA_USE_LLM", "true").lower() in ("1", "true", "yes")
        if use_llm and llm_extractor is not None:
            # Use only the first ~2 pages worth of text for speed/quality
            first_block = full_text[:4000]
            try:
                llm_result = await llm_extractor.extract(filename=metadata["filename"], text_sample=first_block)
                if isinstance(llm_result, dict):
                    title = (llm_result.get("title") or "").strip()
                    authors = (llm_result.get("authors") or "").strip()
                    year = llm_result.get("year")
                    if title:
                        metadata["title"] = title
                    if authors:
                        metadata["authors"] = authors
                    if isinstance(year, int):
                        metadata["year"] = year
                    # If LLM produced reasonable fields, return early
                    if metadata["title"] or metadata["authors"] or metadata["year"] is not None:
                        return metadata
            except Exception as e:
                logger.warning(f"LLM metadata extraction failed, falling back to regex: {e}")

        # Fallback: lightweight regex heuristics
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