from redis import Redis
import json
from typing import Any, Set

from app.storage.redis import redis_dependency


class BaseCache:
    def __init__(self, key_prefix: str, expire: int):
        self.key_prefix = key_prefix
        self.expire = expire

    def get(self, redis: Redis, key: str) -> Any:
        """Get Value from Key"""
        print("key", self.key_prefix + key)
        response = redis.get(
            self.key_prefix + key,
        )
        return response

    def set(self, redis: Redis, key: str, value: Any, expire: int = None):
        """Set Value to Key"""
        redis.set(self.key_prefix + key, value, expire or self.expire)

    def keys(self, redis: Redis, pattern: str) -> Set[str]:
        """Get Keys by Pattern"""
        return redis.keys(self.key_prefix + pattern)

    def set_list(self, redis: Redis, key: str, value: list, expire: int = None):
        """Set Value to Key"""
        for v in value:
            redis.rpush(self.key_prefix + key, json.dumps(v))
        redis.expire(self.key_prefix + key, expire or self.expire)

    def get_list(self, redis: Redis, key: str) -> list:
        """Get Value from Key"""
        return [json.loads(v) for v in redis.lrange(self.key_prefix + key, 0, -1)]

    def add_to_list(self, redis: Redis, key: str, value: Any):
        """Add Value to List"""
        redis.rpush(self.key_prefix + key, json.dumps(value))

    def remove_from_list(self, redis: Redis, key: str, value: Any):
        """Remove Value from List"""
        redis.lrem(self.key_prefix + key, 0, json.dumps(value))

    def set_dict(self, redis: Redis, key: str, value: dict, expire: int = None):
        """Set Value to Key"""
        redis.hmset(self.key_prefix + key, value)
        redis.expire(self.key_prefix + key, expire or self.expire)

    def get_dict(self, redis: Redis, key: str) -> dict:
        """Get Value from Key"""
        return redis.hgetall(self.key_prefix + key)

    def set_number(self, redis: Redis, key: str, value: int, expire: int = None):
        """Set Value to Key"""
        redis.set(self.key_prefix + key, value, expire or self.expire)

    def get_number(self, redis: Redis, key: str) -> int:
        """Get Value from Key"""
        response = redis.get(self.key_prefix + key)
        return int(response) if response else None

    def incr(self, redis: Redis, key: str, amount: int = 1):
        """Increase Value"""
        redis.incr(self.key_prefix + key, amount)

    def decr(self, redis: Redis, key: str, amount: int = 1):
        """Decrease Value"""
        redis.decr(self.key_prefix + key, amount)

    def delete(self, redis: Redis, key: str):
        """Delete Key"""
        redis.delete(self.key_prefix + key)
