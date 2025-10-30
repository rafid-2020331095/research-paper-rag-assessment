from fastapi import APIRouter
from src.api import papers, query

# Main API router
api_router = APIRouter()

# Include all sub-routers
api_router.include_router(papers.router)
api_router.include_router(query.router)