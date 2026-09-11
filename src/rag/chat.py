from src.llm.gateway import LLMGateway
from src.retrieval.hybrid_retriever import HybridRetriever
from src.caching.redis_cache import RedisCache
from loguru import logger

CHAT_SYSTEM_PROMPT = """You are a financial analyst assistant answering questions about
company financial reports. Answer ONLY using the provided context below.

IMPORTANT: Financial tables may have ambiguous or misaligned column headers (e.g. figures
labeled "Three Months" vs "Nine Months" may be unclear due to formatting issues). Before
citing a number, verify which time period it actually corresponds to by cross-checking
against other context (e.g. a matching Revenue or Net Sales figure for the same period).
If you cannot confidently determine which period a number belongs to, say so explicitly
rather than guessing.

If the context does not contain enough information to answer confidently, say so explicitly
rather than guessing. Cite specific numbers from the context when relevant."""


def build_context(retrieved_chunks: list[dict]) -> str:
    """Formats retrieved chunks into a single context block for the prompt."""
    parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        source = chunk["metadata"].get("source", "unknown")
        parts.append(f"[Chunk {i} | source: {source}]\n{chunk['text']}")
    return "\n\n".join(parts)


def answer_question(
    query: str,
    retriever: HybridRetriever,
    gateway: LLMGateway,
    cache: RedisCache,
    top_k: int = 10,
) -> str:
    """
    Retrieves relevant chunks for the query and generates a grounded
    answer using the LLM gateway. Checks the response cache first to
    avoid redundant LLM calls for repeated questions.
    """
    cached_answer = cache.get_response(query)
    if cached_answer:
        logger.info(f"Returning cached answer for: {query}")
        return cached_answer

    retrieved_chunks = retriever.retrieve(query, top_k=top_k)

    if not retrieved_chunks:
        logger.warning(f"No relevant context found for query: {query}")
        return "I couldn't find relevant information in the ingested reports to answer this question."

    context = build_context(retrieved_chunks)
    prompt = f"{CHAT_SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}"

    answer = gateway.generate(prompt)
    cache.set_response(query, answer)
    return answer