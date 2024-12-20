from fastapi import status
from fastapi import status
from redis import Redis
from fastapi import UploadFile
from typing import List, Tuple
from datetime import datetime
from celery.result import AsyncResult

from app.storage.s3 import s3_service
from app.hepler.common import CommonHelper
from app.core import constant
from app.storage.cache.job_cache_service import job_cache_service
from app.schema.job import Job
from app.schema.recommendation import RecommendationRequest, PreRecommendationRequest, NewProductRecommendationRequest
from app.schema.job import GetJobRequest
from app.hepler.enum import JobStatus
from app.common.exception import CustomException
from app.common.response import CustomResponse
from app.core.recommendation.recommendation_model import pre_recommend, recommend, new_product_recommend
from app.tasks.recommendation import recommend_task
from app.hepler.enum import JobStatus

class RecommendationService:
    async def get(self, redis, data: dict):
        request_data = GetJobRequest(**data)

        job: Job = job_cache_service.get_cache_job(redis, job_id=request_data.job_id)
        if not job:
            raise CustomException(
                status_code=status.HTTP_404_NOT_FOUND,
                msg="Job not found",
            )
        
        return CustomResponse(data=job)

    async def pre_recommendation(self, redis, data: dict):
        request_data = PreRecommendationRequest(**data)
        result = pre_recommend(redis, request_data)

        return CustomResponse(data=result)

    async def recommendation(self, redis, data: dict):
        request_data = RecommendationRequest(**data)

        task = recommend_task.apply_async(
            args=[request_data.model_dump(), "recommendation"],
        )

        response = {
            "job_id": task.id,
            "status": JobStatus.PENDING.value,
        }

        return CustomResponse(data=response)
    
    async def product_recommendation(self, redis, data: dict):
        request_data = NewProductRecommendationRequest(**data)
        result = new_product_recommend(redis, request_data)

        return CustomResponse(data=result)

recommendation_service = RecommendationService()
