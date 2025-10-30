import os
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import json

import httpx
from dotenv import load_dotenv
import google.generativeai as genai

from src.services.embedding_service import get_embedding_service
from src.services.qdrant_client import get_qdrant_service
from src.models.models import QueryResponse, CitationResponse

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for answering queries about research papers."""
    
    def __init__(self):
        """Initialize the RAG pipeline."""
        self.llm_type = os.getenv("LLM_TYPE", "gemini")  # "gemini", "ollama" or "deepseek"
        
        # Gemini configuration
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        
        # Legacy configurations (kept for backward compatibility)
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.deepseek_api_url = "https://api.deepseek.com/v1/chat/completions"
        
        # Initialize Gemini if API key is available
        if self.gemini_api_key and self.llm_type.lower() == "gemini":
            genai.configure(api_key=self.gemini_api_key)
        
    async def process_query(self, 
                          query: str, 
                          top_k: int = 5, 
                          paper_ids: Optional[List[int]] = None) -> Tuple[QueryResponse, float]:
        """
        Process a query using the RAG pipeline.
        
        Args:
            query: The query text
            top_k: Number of relevant chunks to retrieve
            paper_ids: Optional list of paper IDs to search within
            
        Returns:
            Tuple containing:
                - QueryResponse object with answer, citations, sources, and confidence
                - Response time in milliseconds
        """
        start_time = time.time()
        
        # Get services
        embedding_service = await get_embedding_service()
        qdrant_service = await get_qdrant_service()
        
        # Generate query embedding
        query_embedding = await embedding_service.generate_query_embedding(query)
        
        # Retrieve relevant chunks
        search_results = await qdrant_service.search_similar(
            query_vector=query_embedding,
            top_k=top_k,
            paper_ids=paper_ids
        )
        
        if not search_results:
            # No relevant chunks found
            return QueryResponse(
                answer="I couldn't find any relevant information to answer your question.",
                citations=[],
                sources_used=[],
                confidence=0.0
            ), 0.0
        
        # Prepare context from retrieved chunks
        context = self._prepare_context(search_results)
        
        # Generate answer using LLM
        answer, confidence = await self._generate_answer(query, context)
        
        # Extract citations from search results
        citations = self._extract_citations(search_results)
        
        # Get unique source papers
        sources_used = list(set([citation.paper_title for citation in citations]))
        
        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000
        
        return QueryResponse(
            answer=answer,
            citations=citations,
            sources_used=sources_used,
            confidence=confidence
        ), response_time_ms
    
    def _prepare_context(self, search_results: List[Dict[str, Any]]) -> str:
        """
        Prepare context from search results for the LLM.
        
        Args:
            search_results: List of search results from Qdrant
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, result in enumerate(search_results):
            payload = result["payload"]
            context_parts.append(
                f"[Document {i+1}] {payload.get('paper_title', 'Unknown Paper')}\n"
                f"Section: {payload.get('section', 'Unknown Section')}\n"
                f"Page: {payload.get('page_number', 'Unknown')}\n"
                f"Content: {payload.get('content', '')}\n"
            )
        
        return "\n\n".join(context_parts)
    
    async def _generate_answer(self, query: str, context: str) -> Tuple[str, float]:
        """
        Generate an answer using the LLM based on the query and context.
        
        Args:
            query: The query text
            context: The context from retrieved documents
            
        Returns:
            Tuple containing:
                - Generated answer
                - Confidence score
        """
        # Construct prompt
        prompt = self._construct_prompt(query, context)
        
        # Call appropriate LLM based on configuration
        if self.llm_type.lower() == "gemini":
            return await self._call_gemini(prompt)
        elif self.llm_type.lower() == "ollama":
            return await self._call_ollama(prompt)
        elif self.llm_type.lower() == "deepseek":
            return await self._call_deepseek(prompt)
        else:
            raise ValueError(f"Unsupported LLM type: {self.llm_type}")
    
    def _construct_prompt(self, query: str, context: str) -> str:
        """
        Construct a prompt for the LLM.
        
        Args:
            query: The query text
            context: The context from retrieved documents
            
        Returns:
            Formatted prompt string
        """
        return f"""You are a research assistant helping with academic papers. Answer the question based ONLY on the provided context. 
If you cannot answer the question based on the context, say "I don't have enough information to answer this question."

CONTEXT:
{context}

QUESTION:
{query}

INSTRUCTIONS:
1. Answer the question concisely and accurately based only on the provided context.
2. Cite specific papers, sections, and page numbers in your answer.
3. Do not make up information or use knowledge outside the provided context.
4. If the context contains conflicting information, acknowledge this in your answer.
5. Format your answer in a clear, structured way.
6. At the end, include a confidence score between 0.0 and 1.0 on a new line in this format: "CONFIDENCE: 0.X"

ANSWER:
"""
    
    async def _call_ollama(self, prompt: str) -> Tuple[str, float]:
        """
        Call Ollama LLM API.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            Tuple containing:
                - Generated text
                - Confidence score
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=60.0
                )
                
                response.raise_for_status()
                result = response.json()
                
                # Extract answer and confidence
                answer_text = result.get("response", "")
                confidence = self._extract_confidence(answer_text)
                
                # Remove confidence line from answer if present
                answer_text = answer_text.replace(f"CONFIDENCE: {confidence}", "").strip()
                
                return answer_text, confidence
                
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            return "I encountered an error while generating the answer.", 0.0
    
    async def _call_gemini(self, prompt: str) -> Tuple[str, float]:
        """
        Call Gemini API.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            Tuple containing:
                - Generated text
                - Confidence score
        """
        if not self.gemini_api_key:
            logger.error("Gemini API key not provided")
            return "Gemini API key not configured.", 0.0
            
        try:
            # Create Gemini client
            genai.configure(api_key=self.gemini_api_key)
            model = genai.GenerativeModel(self.gemini_model)
            
            # Run in asyncio executor to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: model.generate_content(prompt)
            )
            
            # Extract answer text
            answer_text = response.text
            confidence = self._extract_confidence(answer_text)
            
            # Remove confidence line from answer if present
            answer_text = answer_text.replace(f"CONFIDENCE: {confidence}", "").strip()
            
            return answer_text, confidence
                
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return "I encountered an error while generating the answer.", 0.0
    
    async def _call_deepseek(self, prompt: str) -> Tuple[str, float]:
        """
        Call DeepSeek LLM API.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            Tuple containing:
                - Generated text
                - Confidence score
        """
        if not self.deepseek_api_key:
            logger.error("DeepSeek API key not provided")
            return "DeepSeek API key not configured.", 0.0
            
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.deepseek_api_url,
                    headers={
                        "Authorization": f"Bearer {self.deepseek_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "deepseek-chat",
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 2000
                    },
                    timeout=60.0
                )
                
                response.raise_for_status()
                result = response.json()
                
                # Extract answer and confidence
                answer_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                confidence = self._extract_confidence(answer_text)
                
                # Remove confidence line from answer if present
                answer_text = answer_text.replace(f"CONFIDENCE: {confidence}", "").strip()
                
                return answer_text, confidence
                
        except Exception as e:
            logger.error(f"Error calling DeepSeek API: {e}")
            return "I encountered an error while generating the answer.", 0.0
    
    def _extract_confidence(self, text: str) -> float:
        """
        Extract confidence score from LLM output.
        
        Args:
            text: The LLM output text
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        try:
            # Look for confidence pattern
            import re
            confidence_match = re.search(r"CONFIDENCE:\s*(0\.\d+|1\.0)", text)
            if confidence_match:
                return float(confidence_match.group(1))
            
            # Default confidence if not found
            return 0.7
        except Exception:
            return 0.5
    
    def _extract_citations(self, search_results: List[Dict[str, Any]]) -> List[CitationResponse]:
        """
        Extract citations from search results.
        
        Args:
            search_results: List of search results from Qdrant
            
        Returns:
            List of CitationResponse objects
        """
        citations = []
        
        for result in search_results:
            payload = result["payload"]
            citation = CitationResponse(
                paper_title=payload.get("paper_title", "Unknown Paper"),
                section=payload.get("section", None),
                page=payload.get("page_number", None),
                relevance_score=result["score"]
            )
            citations.append(citation)
        
        return citations

# Create a singleton instance
rag_pipeline = RAGPipeline()

async def get_rag_pipeline() -> RAGPipeline:
    """Dependency for getting the RAG pipeline."""
    return rag_pipeline