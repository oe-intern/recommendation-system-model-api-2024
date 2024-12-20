from celery import Celery
from celery.schedules import crontab

from app.core.config import settings

import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)

celery_app = Celery(
    "worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.config_from_object("app.core.celery_config")

celery_app.autodiscover_tasks(["app.tasks.recommendation"], force=True)