from redis import Redis, ConnectionPool
import json
from typing import Any, Optional, Annotated
from fastapi import Depends

from app.core.config import settings
from typing import Set, Any, Optional
from app.core.loggers import get_logger

logger = get_logger(__name__)


class RedisBackend:
    def __init__(
        self,
        host: str,
        port: int,
        password: str,
        db: int,
        expire: int,
    ):
        self.expire = expire
        self.connection_pool = ConnectionPool(
            host=host,
            port=port,
            password=password,
            db=db,
            max_connections=settings.REDIS_MAX_CONNECTIONS or 10,
        )
        self.connection = Redis(connection_pool=self.connection_pool)

    def get(self, key: str) -> Any:
        """Get Value from Key"""
        response = self.connection.get(
            key,
        )
        return response

    def set(self, key: str, value: str, expire: int = None):
        """Set Value to Key"""
        self.connection.set(key, value, expire or self.expire)

    def keys(self, pattern: str) -> Set[str]:
        """Get Keys by Pattern"""
        return self.connection.keys(pattern)

    def set_list(self, key: str, value: list, expire: int = None):
        """Set Value to Key"""
        for v in value:
            self.connection.rpush(key, json.dumps(v))
        self.connection.expire(key, expire or self.expire)

    def get_list(self, key: str) -> list:
        """Get Value from Key"""
        return [json.loads(v) for v in self.connection.lrange(key, 0, -1)]

    def set_dict(self, key: str, value: dict, expire: int = None):
        """Set Value to Key"""
        self.connection.hmset(key, value)
        self.connection.expire(key, expire or self.expire)

    def get_dict(self, key: str) -> dict:
        """Get Value from Key"""
        return self.connection.hgetall(key)

    def delete(self, key: str):
        """Delete Key"""
        self.connection.delete(key)


class RedisDependency:
    redis: Optional[RedisBackend] = None

    def __call__(self):
        return self.redis

    def init(self):
        try:
            self.redis = RedisBackend(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                db=settings.REDIS_DB,
                expire=settings.REDIS_EXPIRE,
            )
            self.redis.connection.ping()
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")

    def close(self):
        if self.redis:
            self.redis.connection.close()

    def get_redis(self):
        return self.redis.connection


redis_dependency = RedisDependency()


def get_redis() -> Redis:
    if redis_dependency.redis is None:
        redis_dependency.init()
    return redis_dependency.get_redis()
