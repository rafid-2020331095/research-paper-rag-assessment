import os
import logging
import uuid
from typing import List, Dict, Any, Optional, Union
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class QdrantService:
    """Service for interacting with Qdrant vector database."""
    
    def __init__(self):
        """Initialize the Qdrant client."""
        self.host = os.getenv("QDRANT_HOST", "localhost")
        self.port = int(os.getenv("QDRANT_PORT", "6333"))
        self.collection_name = os.getenv("QDRANT_COLLECTION", "research_papers")
        self.vector_size = 384  # Default for all-MiniLM-L6-v2
        self.client = None
    
    async def initialize(self):
        """Initialize the Qdrant client and create collection if it doesn't exist."""
        if self.client is None:
            self.client = QdrantClient(host=self.host, port=self.port)
            logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
            
            # Check if collection exists, create if not
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
                )
                logger.info(f"Created collection: {self.collection_name}")
    
    async def store_embeddings(self, 
                              embeddings: List[np.ndarray], 
                              metadata: List[Dict[str, Any]]) -> List[str]:
        """
        Store embeddings in Qdrant with associated metadata.
        
        Args:
            embeddings: List of embedding vectors
            metadata: List of metadata dictionaries for each embedding
            
        Returns:
            List of IDs for the stored points
        """
        if self.client is None:
            await self.initialize()
        
        # Generate unique IDs for each point
        ids = [str(uuid.uuid4()) for _ in range(len(embeddings))]
        
        # Create points
        points = [
            PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload=meta
            )
            for point_id, embedding, meta in zip(ids, embeddings, metadata)
        ]
        
        # Upload in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        
        logger.info(f"Stored {len(embeddings)} embeddings in Qdrant")
        return ids
    
    async def search_similar(self, 
                           query_vector: np.ndarray, 
                           top_k: int = 5,
                           paper_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in Qdrant.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            paper_ids: Optional list of paper IDs to filter by
            
        Returns:
            List of dictionaries with search results
        """
        if self.client is None:
            await self.initialize()
        
        # Create filter if paper_ids is provided
        query_filter = None
        if paper_ids:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="paper_id",
                        match=models.MatchAny(any=paper_ids)
                    )
                ]
            )
        
        # Perform search (use query_filter per newer SDK)
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
            query_filter=query_filter
        )
        
        # Format results
        results = []
        for result in search_results:
            results.append({
                "id": result.id,
                "score": result.score,
                "payload": result.payload
            })
        
        return results
    
    async def delete_by_paper_id(self, paper_id: int) -> int:
        """
        Delete all vectors associated with a specific paper.
        
        Args:
            paper_id: ID of the paper to delete vectors for
            
        Returns:
            Number of deleted points
        """
        if self.client is None:
            await self.initialize()
        
        # Create filter for the paper_id
        delete_filter = Filter(
            must=[
                FieldCondition(
                    key="paper_id",
                    match=MatchValue(value=paper_id)
                )
            ]
        )
        
        # Delete points
        result = self.client.delete(
            collection_name=self.collection_name,
            points_selector=delete_filter
        )
        
        logger.info(f"Deleted vectors for paper_id={paper_id}")
        return result.status

# Create a singleton instance
qdrant_service = QdrantService()

async def get_qdrant_service() -> QdrantService:
    """Dependency for getting the Qdrant service."""
    await qdrant_service.initialize()
    return qdrant_service