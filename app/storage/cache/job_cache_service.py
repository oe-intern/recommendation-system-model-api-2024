from redis import Redis
import json

from app.storage.base_cache import BaseCache
from app.schema.job import Job


class JobCacheService(BaseCache):
    def __init__(self):
        super().__init__("job_cache_", 86400)
        self.image_url_message_key = "image_url_message"

    def cache_job(
        self,
        redis: Redis,
        *,
        job_id: str,
        job: Job,
        expire_time: int = None,
    ) -> None:
        value = json.dumps(job.model_dump(), default=str)
        self.set(
            redis,
            job_id,
            value,
            expire_time,
        )

    def get_cache_job(self, redis: Redis, *, job_id: str) -> Job:
        response = self.get(
            redis,
            job_id,
        )
        return Job(**json.loads(response)) if response else None

    def delete_cache_job(
        self,
        redis: Redis,
        *,
        job_id: str,
    ):
        self.delete(redis, job_id)

    def update_cache_job(
        self,
        redis: Redis,
        *,
        job_id: str,
        job: Job,
        expire_time: int = None,
    ):
        value = json.dumps(job.model_dump(), default=str)
        self.set(
            redis,
            job_id,
            value,
            expire_time,
        )


job_cache_service = JobCacheService()
