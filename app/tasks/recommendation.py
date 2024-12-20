from app.core.celery_app import celery_app
from sqlalchemy.orm import Session
from datetime import timedelta, datetime
from time import sleep
from redis import Redis
import torch
import json

from app.storage.redis import get_redis, redis_dependency
from app.storage.cache.job_cache_service import job_cache_service
from app.hepler.common import CommonHelper
from app.hepler.enum import JobStatus
from app.core.recommendation.recommendation_model import recommend
from app.schema.job import Job
from app.schema.recommendation import RecommendationRequest
from app.storage.s3 import s3_service


@celery_app.task(bind=True, name="recommend_task", max_retries=3)
def recommend_task(self, data: dict, name: str) -> str:
    try:
        if redis_dependency.redis is None:
            redis_dependency.init()
        redis: Redis = redis_dependency.redis
        job = Job(
            job_id=self.request.id,
            created_at=datetime.now(),
            status=JobStatus.PENDING,
        )
        print("STARTING JOB")
        job_cache_service.cache_job(
            redis,
            job_id=self.request.id,
            job=job,
        )
        print("JOB CACHED")
        print("STARTING RECOMMENDATION")
        result = recommend(redis, data)
        result_json = json.dumps(result)
        print("RECOMMENDATION DONE")
        url = s3_service.update_json_data(
            key=self.request.id,
            data=result_json,
        )

        job_cache_service.update_cache_job(
            redis,
            job_id=self.request.id,
            job=Job(
                job_id=self.request.id,
                created_at=job.created_at,
                status=JobStatus.SUCCESS,
                result_url=url,
            ),
        )

        return f"Task {name} executed successfully"
    except Exception as exc:
        raise self.retry(exc=exc)
