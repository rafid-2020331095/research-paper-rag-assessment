import os
import logging
from typing import List, Dict, Any
import asyncio

from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating embeddings from text using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding service with the specified model.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = "all-MiniLM-L6-v2"
        self.model = None
    
    async def initialize(self):
        """Initialize the embedding model asynchronously."""
        if self.model is None:
            # Load model in a separate thread to avoid blocking
            self.model = await asyncio.to_thread(SentenceTransformer, self.model_name)
            logger.info(f"Initialized embedding model: {self.model_name}")
    
    async def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if self.model is None:
            await self.initialize()
        logger.info(f"Generating embeddings for {len(texts)} texts using model {self.model_name}")
        
        # Process in batches to avoid memory issues with large documents
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # Run embedding generation in a thread pool
            batch_embeddings = await asyncio.to_thread(self.model.encode, batch)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    async def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query string.
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector for the query
        """
        if self.model is None:
            await self.initialize()
        logger.info(f"Generating embedding for query: {query}")
        
        # Run embedding generation in a thread pool
        embedding = await asyncio.to_thread(self.model.encode, query)
        return embedding

# Create a singleton instance
embedding_service = EmbeddingService()

async def get_embedding_service() -> EmbeddingService:
    """Dependency for getting the embedding service."""
    await embedding_service.initialize()
    return embedding_service