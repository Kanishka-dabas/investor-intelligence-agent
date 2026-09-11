import uuid

from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer
from loguru import logger

DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DENSE_VECTOR_SIZE = 384
SPARSE_MODEL_NAME = "Qdrant/bm25"


class QdrantVectorStore:
    """
    Wraps Qdrant collection setup and chunk ingestion with named
    dense (MiniLM) + sparse (BM25) vectors for hybrid search.
    """

    def __init__(self, host: str, port: int, collection_name: str):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.dense_model = SentenceTransformer(DENSE_MODEL_NAME)
        self.sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)

    def create_collection(self, recreate: bool = False) -> None:
        """Creates the collection with named dense + sparse vectors if it doesn't exist."""
        exists = self.client.collection_exists(self.collection_name)

        if exists and not recreate:
            logger.info(f"Collection '{self.collection_name}' already exists, skipping creation.")
            return

        if exists and recreate:
            logger.warning(f"Recreating collection '{self.collection_name}' — existing data will be lost.")
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=DENSE_VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(),
            },
        )
        logger.info(f"Collection '{self.collection_name}' created.")

    def upsert_chunks(self, chunks: list[str], metadatas: list[dict], start_id: int = 0) -> int:
        """
        Embeds and upserts a list of text chunks with their metadata.
        Returns the number of points upserted.

        Uses a content-based UUID (derived from source filename + chunk
        index + chunk text) instead of a small sequential int, so chunks
        from different files never collide on ID and re-ingesting the
        same file overwrites its own points instead of someone else's.
        """
        dense_vectors = self.dense_model.encode(chunks, show_progress_bar=False)
        sparse_vectors = list(self.sparse_model.embed(chunks))

        points = []
        for i, (chunk, meta, dense_vec, sparse_vec) in enumerate(
            zip(chunks, metadatas, dense_vectors, sparse_vectors)
        ):
            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_DNS,
                    f"{meta.get('source', 'unknown')}_{i}_{chunk[:50]}",
                )
            )
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_vec.tolist(),
                        "sparse": models.SparseVector(
                            indices=sparse_vec.indices.tolist(),
                            values=sparse_vec.values.tolist(),
                        ),
                    },
                    payload={"text": chunk, **meta},
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(f"Upserted {len(points)} chunks into '{self.collection_name}'.")
        return len(points)

    def count(self) -> int:
        return self.client.count(self.collection_name).count  