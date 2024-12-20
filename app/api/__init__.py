from fastapi import APIRouter

from app.api.api_v1.endpoint.recommendation import recommendation

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(
    recommendation.router, prefix="/recommendation", tags=["recommendation"]
)
