import logging
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import numpy as np
from typing import List, Dict, Any
from src.models.database import get_db
from src.models.models import Query as QueryModel
from src.services.embedding_service import get_embedding_service
from sklearn.cluster import KMeans

router = APIRouter(prefix="/api/analytics", tags=["analytics"])
logger = logging.getLogger(__name__)

@router.get("/popular")
async def get_popular_topics(
    k: int = Query(7, ge=2, le=20, description="Number of topics/clusters") ,
    max_samples: int = Query(3, ge=1, le=10, description="Example queries per topic"),
    db: AsyncSession = Depends(get_db),
    embedding_service = Depends(get_embedding_service),
):
    """
    Find the most common semantic topics in user queries using k-means over query embeddings.
    Returns a list of topics with representative query and popularity.
    """
    # Fetch all queries
    sql = select(QueryModel.text).order_by(QueryModel.id.desc()).limit(500)
    result = await db.execute(sql)
    queries: List[str] = [row[0] for row in result.all() if row[0]]
    if not queries:
        return []
    # Generate embeddings
    try:
        embeddings = await embedding_service.generate_embeddings(queries)
        X = np.stack(embeddings)
    except Exception as e:
        logger.error(f"Embedding failure: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate embeddings.")
    # Cluster
    n_clusters = min(k, len(embeddings))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = kmeans.fit_predict(X)
    clusters: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(idx)
    # For each cluster, find center-point query, select sample queries
    topics = []
    for label, idxs in clusters.items():
        inds = np.array(idxs)
        emb_cluster = X[inds]
        ctr = kmeans.cluster_centers_[label]
        dists = np.linalg.norm(emb_cluster - ctr, axis=1)
        rep_idx = inds[np.argmin(dists)]
        topic_info = {
            "topic_repr": queries[rep_idx],
            "num_queries": len(idxs),
            "sample_queries": [queries[i] for i in inds[:max_samples]]
        }
        topics.append(topic_info)
    # Sort by number of queries descending
    topics.sort(key=lambda x: x["num_queries"], reverse=True)
    return topics
