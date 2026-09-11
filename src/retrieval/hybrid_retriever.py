from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding
from src.caching.redis_cache import RedisCache
from loguru import logger

DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL_NAME = "Qdrant/bm25"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class HybridRetriever:
    """
    Hybrid retrieval: dense (MiniLM) + sparse (BM25) search fused via
    Qdrant's native RRF, followed by cross-encoder reranking.
    """

    def __init__(self, host: str, port: int, collection_name: str, cache: RedisCache | None = None):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.dense_model = SentenceTransformer(DENSE_MODEL_NAME)
        self.sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
        self.reranker = CrossEncoder(RERANKER_MODEL_NAME)
        self.cache = cache

    def retrieve(self, query: str, top_k: int = 5, prefetch_limit: int = 20) -> list[dict]:
        """
        Runs hybrid search (dense + sparse, RRF-fused) then reranks
        the fused candidates with a cross-encoder.

        Returns top_k results as list of {text, score, metadata}.
        """
        dense_vector = None
        if self.cache:
            dense_vector = self.cache.get_embedding(query)

        if dense_vector is None:
            dense_vector = self.dense_model.encode(query).tolist()
            if self.cache:
                self.cache.set_embedding(query, dense_vector)
        sparse_vector = list(self.sparse_model.embed([query]))[0]

        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=prefetch_limit,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vector.indices.tolist(),
                        values=sparse_vector.values.tolist(),
                    ),
                    using="sparse",
                    limit=prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=prefetch_limit,
        )

        candidates = [
            {"text": point.payload.get("text", ""), "metadata": point.payload}
            for point in results.points
        ]

        if not candidates:
            logger.warning(f"No candidates found for query: {query}")
            return []

        pairs = [[query, c["text"]] for c in candidates]
        rerank_scores = self.reranker.predict(pairs)

        for candidate, score in zip(candidates, rerank_scores):
            candidate["score"] = float(score)

        reranked = sorted(candidates, key=lambda c: c["score"], reverse=True)
        return reranked[:top_k]