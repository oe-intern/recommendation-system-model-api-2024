from fastapi import (
    APIRouter,
    Depends,
    File,
    UploadFile,
    Form,
    Query,
    Path,
    Body,
)
from redis import Redis

from app.storage.redis import get_redis
from app.core.recommendation.recommendation_service import recommendation_service

router = APIRouter()


@router.get("/job/{job_id}", summary="Get a job by id.")
async def get_me(
    redis: Redis = Depends(get_redis),
    job_id: str = Path(
        ...,
        description="The id of the job.",
        example="d2f60111-aec6-4c58-83a7-24f0edb7ac5f",
    ),
):
    """
    Get the job by id.

    This endpoint allows getting the job by id.

    Returns:
    - status_code (200): The job has been found successfully.
    - status_code (404): The job is not found.

    """
    return await recommendation_service.get(redis, {"job_id": job_id})


@router.post("/pre-recommendation", summary="Handle pre-recommendation.")
async def pre_recommdation(
    redis: Redis = Depends(get_redis),
    data: dict = Body(
        ...,
        example={
            "number_of_items": 6,
            "products": [],
            "type_scores": [],
            "product_scores": [],
            "total": 0,
        },
    ),
):
    """
    Handle pre-recommendation.

    This endpoint allows handling pre-recommendation.

    Parameters:
    - number_of_items (int): The number of items.
    - products (List[object]): The list of products.
    - type_scores (List[object]): The list of type scores.
    - total (int): The total.
    - product_scores (List[object]): The list of product scores.

    Returns:
    - status_code (200): The pre-recommendation has been handled successfully.
    - status_code (400): The request is invalid.

    """

    return await recommendation_service.pre_recommendation(redis, data)

@router.post("/recommendation", summary="Handle recommendation.")
async def recommdation(
    redis: Redis = Depends(get_redis),
    data: dict = Body(
        ...,
        example={
            "number_of_items": 6,
            "products": [],
            "type_scores": [],
            "total": 0,
            "product_scores": [],
        },
    ),
):
    """
    Handle recommendation.

    This endpoint allows handling recommendation.

    Parameters:
    - number_of_items (int): The number of items.
    - products (List[object]): The list of products.
    - type_scores (List[object]): The list of type scores.
    - total (int): The total.
    - product_scores (List[object]): The list of product scores.

    Returns:
    - status_code (200): The recommendation has been handled successfully.
    - status_code (400): The request is invalid.

    """

    return await recommendation_service.recommendation(redis, data)

@router.post("/product", summary="Handle recommendation.")
async def recommdation(
    redis: Redis = Depends(get_redis),
    data: dict = Body(
        ...,
        example={
            "number_of_items": 6,
            "products": [],
            "product_id": "grid://....",
        },
    ),
):
    """
    Handle recommendation.

    This endpoint allows handling recommendation.

    Parameters:
    - number_of_items (int): The number of items.
    - products (List[object]): The list of products.
    - type_scores (List[object]): The list of type scores.
    - total (int): The total.
    - product_scores (List[object]): The list of product scores.

    Returns:
    - status_code (200): The recommendation has been handled successfully.
    - status_code (400): The request is invalid.

    """

    return await recommendation_service.product_recommendation(redis, data)