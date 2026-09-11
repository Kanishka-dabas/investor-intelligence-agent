from functools import lru_cache

from src.llm.gateway import LLMGateway
from src.retrieval.hybrid_retriever import HybridRetriever
from src.caching.redis_cache import RedisCache
from src.graph.build_graph import build_graph
from src.config import settings


@lru_cache
def get_gateway() -> LLMGateway:
    return LLMGateway()


@lru_cache
def get_cache() -> RedisCache:
    return RedisCache()


@lru_cache
def get_retriever() -> HybridRetriever:
    return HybridRetriever(
        host=settings.vectordb.host,
        port=settings.vectordb.port,
        collection_name=settings.vectordb.collection_name,
        cache=get_cache(),
    )


@lru_cache
def get_graph_app():
    return build_graph(get_gateway(), get_retriever(), get_cache())