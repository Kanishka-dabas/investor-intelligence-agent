import hashlib
import json

import redis
from loguru import logger

from src.config import settings

EMBEDDING_CACHE_TTL = 60 * 60 * 24 * 7   # 7 days — embeddings for the same text rarely change
RESPONSE_CACHE_TTL = 60 * 60             # 1 hour — LLM answers should not go stale too long


class RedisCache:
    """
    Redis-based cache for embeddings and LLM responses, keyed by a hash
    of the input text/query. Uses separate logical DBs so caching and
    Celery don't collide.
    """

    def __init__(self):
        self.client = redis.Redis(
            host=settings.redis.host,
            port=settings.redis.port,
            db=settings.redis.cache_db,
            decode_responses=True,
        )

    def _make_key(self, prefix: str, text: str) -> str:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{prefix}:{text_hash}"

    def get_embedding(self, text: str) -> list[float] | None:
        key = self._make_key("embedding", text)
        cached = self.client.get(key)
        if cached:
            logger.debug(f"Embedding cache hit for key {key}")
            return json.loads(cached)
        return None

    def set_embedding(self, text: str, embedding: list[float]) -> None:
        key = self._make_key("embedding", text)
        self.client.setex(key, EMBEDDING_CACHE_TTL, json.dumps(embedding))

    def get_response(self, query: str) -> str | None:
        key = self._make_key("response", query)
        cached = self.client.get(key)
        if cached:
            logger.debug(f"Response cache hit for key {key}")
            return cached
        return None

    def set_response(self, query: str, response: str) -> None:
        key = self._make_key("response", query)
        self.client.setex(key, RESPONSE_CACHE_TTL, response)