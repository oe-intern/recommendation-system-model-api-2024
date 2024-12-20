from pydantic import BaseModel, Field, validator, ConfigDict
from typing import Optional
from datetime import datetime

from app.hepler.enum import JobStatus


class Job(BaseModel):
    status: JobStatus = JobStatus.PENDING
    created_at: datetime 
    result_url: Optional[str] = None
    job_id: str

    model_config = ConfigDict(from_attribute=True, extra="ignore")

class GetJobRequest(BaseModel):
    job_id: str

    model_config = ConfigDict(from_attribute=True, extra="ignore")