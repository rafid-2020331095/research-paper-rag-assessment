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
# Broadened to better match research paper structure and numbered headings.
SECTION_PATTERNS = {
    "Title": r"^\s*title\b",
    "Abstract": r"^\s*abstract\b",
    "Keywords": r"^\s*keywords?\b",
    "Introduction": r"^\s*introduction\b|^\s*background\b|^\s*motivation\b|^\s*1\.?\s+",
    "Related Work": r"^\s*related\s+work\b|^\s*literature\s+review\b|^\s*prior\s+work\b|^\s*2\.?\s+",
    "Preliminaries": r"^\s*preliminaries\b|^\s*problem\s+formulation\b|^\s*notation\b",
    "Dataset": r"^\s*dataset\b|^\s*data\s+collection\b",
    "Method": r"^\s*method(?:s|ology)?\b|^\s*approach\b|^\s*model\b|^\s*materials?\s+and\s+methods\b|^\s*3\.?\s+",
    "Implementation": r"^\s*implementation\b|^\s*training\b|^\s*experimental\s+setup\b",
    "Experiments": r"^\s*experiments?\b|^\s*evaluation\b|^\s*4\.?\s+",
    "Results": r"^\s*results?\b|^\s*analysis\b",
    "Discussion": r"^\s*discussion\b",
    "Ablation": r"^\s*ablation\b",
    "Limitations": r"^\s*limitations?\b",
    "Conclusion": r"^\s*conclusions?\b|^\s*summary\b",
    "Future Work": r"^\s*future\s+work\b",
    "Acknowledgments": r"^\s*acknowledg?e?ments\b",
    "Appendix": r"^\s*appendix\b|^\s*supplementary\b",
    "References": r"^\s*references\b|^\s*bibliography\b",
}

# Generic numbered heading pattern (e.g., 1, 1.1, 2.3.4 followed by a title)
NUMBERED_HEADING_PATTERN = r"^\s*\d+(?:\.\d+)*\s+[A-Z][^\n]{0,120}$"

# Roman numeral heading pattern (e.g., I. Introduction, II Related Work)
ROMAN_HEADING_PATTERN = r"^\s*(?:[IVXLCDM]+\.?)[\s-]+[A-Z][^\n]{0,120}$"

# Map various raw header strings to canonical section names
CANONICAL_MAP = {
    "abstract": "Abstract",
    "keywords": "Keywords",
    "introduction": "Introduction",
    "background": "Introduction",
    "motivation": "Introduction",
    "related work": "Related Work",
    "literature review": "Related Work",
    "prior work": "Related Work",
    "preliminaries": "Preliminaries",
    "problem formulation": "Preliminaries",
    "notation": "Preliminaries",
    "dataset": "Dataset",
    "data collection": "Dataset",
    "method": "Method",
    "methods": "Method",
    "methodology": "Method",
    "approach": "Method",
    "model": "Method",
    "materials and methods": "Method",
    "implementation": "Implementation",
    "training": "Implementation",
    "experimental setup": "Implementation",
    "experiments": "Experiments",
    "evaluation": "Experiments",
    "results": "Results",
    "analysis": "Results",
    "discussion": "Discussion",
    "ablation": "Ablation",
    "limitations": "Limitations",
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
    "summary": "Conclusion",
    "future work": "Future Work",
    "acknowledgments": "Acknowledgments",
    "acknowledgements": "Acknowledgments",
    "appendix": "Appendix",
    "supplementary": "Appendix",
    "references": "References",
    "bibliography": "References",
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
        """Synchronous implementation of text extraction from PDF.
        Prefer PyMuPDF (fitz) for more robust extraction; fallback to PyPDF2.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        try:
            # Try PyMuPDF
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(file_path)
                pages_text = []
                for i, page in enumerate(doc):
                    text = page.get_text("text")
                    pages_text.append((i + 1, text))
            except Exception:
                # Fallback to PyPDF2
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    num_pages = len(reader.pages)
                    pages_text = []
                    for i in range(num_pages):
                        page = reader.pages[i]
                        text = page.extract_text()
                        pages_text.append((i + 1, text))

            # Identify sections in the document (mid-page aware)
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

        def normalize_header(raw: str) -> str:
            # Remove numbering/roman numerals and map to canonical names
            s = re.sub(r"^\s*\d+(?:\.\d+)*\s+", "", raw.strip(), flags=re.IGNORECASE)
            s = re.sub(r"^\s*(?:[IVXLCDM]+\.?)[\s-]+", "", s, flags=re.IGNORECASE)
            key = s.lower()
            # Try exact canonical map
            for k, v in CANONICAL_MAP.items():
                if key.startswith(k):
                    return v
            # Fallback to capitalized original trimmed
            return s if s else "Front Matter"

        def is_header_line(line: str) -> Optional[str]:
            if not line:
                return None
            # Direct canonical patterns anywhere in line (case-insensitive)
            for canonical, pattern in SECTION_PATTERNS.items():
                if re.search(pattern, line.lower()):
                    return canonical
            # Numbered or Roman headings at line start
            if re.match(NUMBERED_HEADING_PATTERN, line) or re.match(ROMAN_HEADING_PATTERN, line, flags=re.IGNORECASE):
                stripped = re.sub(r"^\s*(?:\d+(?:\.\d+)*|[IVXLCDM]+\.?)[\s-]+", "", line).strip()
                return normalize_header(stripped)
            # Title-like line heuristics
            if 3 <= len(line.split()) <= 12 and len(line) <= 120:
                has_mixed = any(c.islower() for c in line) and any(c.isupper() for c in line)
                if has_mixed and line[0].isalpha():
                    return normalize_header(line)
            return None

        for page_num, text in pages_text:
            page_text = text or ""
            lines = page_text.splitlines()
            buffer: List[str] = []
            header_for_segment = current_header or "Front Matter"

            def flush_segment():
                if buffer:
                    segment_text = "\n".join(buffer).strip()
                    if segment_text:
                        dynamic_sections.setdefault(header_for_segment, []).append((page_num, segment_text))

            for raw in lines:
                line = raw.strip()
                maybe_header = is_header_line(line)
                if maybe_header:
                    # Close previous segment under previous header
                    flush_segment()
                    header_for_segment = maybe_header
                    current_header = maybe_header
                    buffer = []
                    continue
                buffer.append(raw)

            # Flush remaining content for the page
            flush_segment()

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
        # Token-based sizing (larger, fewer chunks) with environment overrides
        # Defaults favor large chunks based on sections.
        target_tokens = int(os.getenv("TARGET_CHUNK_TOKENS", "1200"))  # preferred size per chunk
        max_tokens = int(os.getenv("MAX_CHUNK_TOKENS", "1800"))        # hard upper bound per chunk
        min_tokens = int(os.getenv("MIN_CHUNK_TOKENS", "300"))         # avoid tiny fragments
        overlap_tokens = int(os.getenv("OVERLAP_TOKENS", "60"))        # token overlap between chunks
        # Coalescing configuration to cap total number of chunks
        max_total_chunks = int(os.getenv("MAX_TOTAL_CHUNKS", "50"))
        super_max_tokens = int(os.getenv("SUPER_MAX_TOKENS", "2400"))   # when merging, allow bigger chunks

        # Backward-compat support if callers pass character-based params
        chars_per_token = float(os.getenv("CHARS_PER_TOKEN", "4"))
        max_chars = int(chunk_size * chars_per_token)  # unused in token mode but left for BC
        overlap_chars = int(overlap * chars_per_token)

        def normalize_paragraphs(t: str) -> List[str]:
            """Split text into paragraphs, preserving paragraph boundaries and fixing hyphenation."""
            if not t:
                return []
            # Normalize newlines and remove control chars
            cleaned = t.replace("\r\n", "\n").replace("\r", "\n")
            cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", " ", cleaned)
            # Fix hyphenation at line endings: "hy-\nphen" -> "hyphen"
            cleaned = re.sub(r"-\n", "", cleaned)
            # Collapse single newlines within paragraphs to spaces, preserve double newlines as paragraph breaks
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
            paragraphs = [p.strip() for p in re.split(r"\n\n+", cleaned) if p.strip()]
            # Within each paragraph, collapse remaining newlines to spaces
            paragraphs = [re.sub(r"\n+", " ", p) for p in paragraphs]
            # Drop tiny artifacts
            return [p for p in paragraphs if len(p) > 30]

        chunks: List[Dict] = []
        chunk_counter = 0

        def estimate_tokens(s: str) -> int:
            if not s:
                return 0
            # Normalize spaces
            s_norm = re.sub(r"\s+", " ", s.strip())
            char_count = len(s_norm)
            words = s_norm.split()
            if not words:
                return 0
            # Base char-per-token for academic text
            base_cpt = 4.2
            complex_words = sum(1 for w in words if len(w) > 10)
            technical_terms = sum(1 for w in words if any(c.isupper() for c in w[1:]))
            if complex_words > len(words) * 0.2:
                base_cpt = 3.8
            if technical_terms > len(words) * 0.1:
                base_cpt = 3.5
            punctuation = sum(1 for c in s_norm if c in '.,;:!?()[]{}')
            est = char_count / base_cpt + punctuation * 0.3
            return max(int(round(est)), len(words))

        def split_into_sentences_for_overlap(t: str) -> List[str]:
            if not t:
                return []
            parts = re.split(r"(?<=[.!?])\s+", t.strip())
            return [p.strip() for p in parts if p and len(p.strip()) > 0]

        # Process each section separately to maintain context and avoid breaking paragraphs
        for section_name, section_pages in sections.items():
            if not section_pages:
                continue

            # Build paragraph list with page attribution
            paragraphs_with_pages: List[Tuple[int, str]] = []
            for page_num, page_text in section_pages:
                for para in normalize_paragraphs(page_text or ""):
                    paragraphs_with_pages.append((page_num, para))

            if not paragraphs_with_pages:
                continue

            # Pack paragraphs into larger chunks by token targets, never cross section boundaries
            i = 0
            while i < len(paragraphs_with_pages):
                cur_paras: List[Tuple[int, str]] = []
                cur_tokens = 0

                # Grow chunk up to target_tokens, but don't exceed max_tokens
                while i < len(paragraphs_with_pages):
                    para_text = paragraphs_with_pages[i][1]
                    add_tokens = estimate_tokens(para_text) + (2 if cur_paras else 0)
                    if cur_tokens + add_tokens > max_tokens and cur_paras:
                        break
                    cur_paras.append(paragraphs_with_pages[i])
                    cur_tokens += add_tokens
                    i += 1

                # If no paragraph could be added (single huge paragraph), hard cut on punctuation
                if not cur_paras:
                    page_n, para_text = paragraphs_with_pages[i]
                    s = para_text.strip()
                    # Try to cut around max_tokens by character heuristic
                    approx_chars = int(max_tokens * chars_per_token)
                    slice_text = s[:approx_chars]
                    cut = max(slice_text.rfind('. '), slice_text.rfind('; '), slice_text.rfind('! '), slice_text.rfind('? '))
                    if cut < int(approx_chars * 0.6):
                        cut = approx_chars
                    cur_paras = [(page_n, slice_text[:cut].rstrip())]
                    i += 1

                # Compose chunk content
                page_number = cur_paras[0][0]
                content_core = "\n\n".join([p for _, p in cur_paras])

                # Token-aware overlap: prepend last sentences from previous chunk and/or keep for next
                # For simplicity, include only previous overlap sentences into current content
                if overlap_tokens > 0 and chunks:
                    prev_content = chunks[-1]["content"]
                    prev_sents = split_into_sentences_for_overlap(prev_content)
                    tail = " ".join(prev_sents[-2:]) if len(prev_sents) >= 2 else (prev_sents[-1] if prev_sents else "")
                    if tail:
                        tail_tokens = estimate_tokens(tail)
                        if tail_tokens > overlap_tokens:
                            # Trim overlap to approximate token limit
                            words = tail.split()
                            keep = max(1, int(overlap_tokens // 1.3))
                            tail = " ".join(words[:keep])
                        content = f"{tail}\n\n{content_core}"
                    else:
                        content = content_core
                else:
                    content = content_core

                # Ensure min_tokens threshold; if too small and there is a next paragraph, try to include it
                total_tokens = estimate_tokens(content)
                if total_tokens < min_tokens and i < len(paragraphs_with_pages):
                    para_text = paragraphs_with_pages[i][1]
                    if total_tokens + estimate_tokens(para_text) <= max_tokens:
                        content = f"{content}\n\n{para_text}"
                        i += 1

                chunks.append({
                    "content": content,
                    "section": section_name,
                    "page_number": page_number,
                    "chunk_index": chunk_counter
                })
                chunk_counter += 1

        # If too many chunks overall, coalesce adjacent chunks while preserving section grouping
        def estimate_tokens_global(s: str) -> int:
            return estimate_tokens(s)

        if len(chunks) > max_total_chunks:
            merged: List[Dict] = []
            idx = 0
            while idx < len(chunks):
                cur = chunks[idx]
                if not merged:
                    merged.append({
                        "content": cur["content"],
                        "section": cur["section"],
                        "page_number": cur["page_number"],
                        "chunk_index": len(merged)
                    })
                else:
                    prev = merged[-1]
                    # Merge only if same section and combined tokens under super_max_tokens
                    if prev["section"] == cur["section"]:
                        combined = f"{prev['content']}\n\n{cur['content']}"
                        if estimate_tokens_global(combined) <= super_max_tokens or len(chunks) - idx + len(merged) > max_total_chunks:
                            prev["content"] = combined
                        else:
                            merged.append({
                                "content": cur["content"],
                                "section": cur["section"],
                                "page_number": cur["page_number"],
                                "chunk_index": len(merged)
                            })
                    else:
                        merged.append({
                            "content": cur["content"],
                            "section": cur["section"],
                            "page_number": cur["page_number"],
                            "chunk_index": len(merged)
                        })
                idx += 1

            # If still over limit, perform a second pass merging across sections conservatively
            chunks = merged
            if len(chunks) > max_total_chunks:
                merged2: List[Dict] = []
                for c in chunks:
                    if not merged2:
                        merged2.append({**c, "chunk_index": 0})
                        continue
                    prev = merged2[-1]
                    combined = f"{prev['content']}\n\n{c['content']}"
                    if estimate_tokens_global(combined) <= super_max_tokens * 1.5 or len(merged2) >= max_total_chunks:
                        prev["content"] = combined
                    else:
                        c2 = {**c}
                        c2["chunk_index"] = len(merged2)
                        merged2.append(c2)
                chunks = merged2

            # Re-index chunk_index after merges
            for i, ch in enumerate(chunks):
                ch["chunk_index"] = i

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
                    title = (llm_result.get("title") or "").strip()[:250]
                    authors = (llm_result.get("authors") or "").strip()[:250]
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