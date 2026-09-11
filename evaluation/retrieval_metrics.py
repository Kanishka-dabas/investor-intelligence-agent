def hit_rate_at_k(retrieved_sources: list[str], relevant_companies: list[str], k: int) -> bool:
    """
    Returns True if at least one of the top-k retrieved chunks comes from
    a source associated with any of the relevant companies for this question.
    """
    top_k_sources = retrieved_sources[:k]
    return any(
        any(company.split()[0].lower() in source.lower() for company in relevant_companies)
        for source in top_k_sources
    )


def precision_at_k(retrieved_sources: list[str], relevant_companies: list[str], k: int) -> float:
    """
    Fraction of the top-k retrieved chunks that come from a relevant source.
    """
    if not relevant_companies:
        return None  # not applicable (e.g. out-of-scope questions)

    top_k_sources = retrieved_sources[:k]
    if not top_k_sources:
        return 0.0

    relevant_count = sum(
        1 for source in top_k_sources
        if any(company.split()[0].lower() in source.lower() for company in relevant_companies)
    )
    return relevant_count / len(top_k_sources)


def mrr(retrieved_sources: list[str], relevant_companies: list[str]) -> float:
    """
    Mean Reciprocal Rank: 1 / (rank of first relevant chunk), or 0 if none found.
    """
    if not relevant_companies:
        return None

    for rank, source in enumerate(retrieved_sources, start=1):
        if any(company.split()[0].lower() in source.lower() for company in relevant_companies):
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(retrieved_sources: list[str], relevant_companies: list[str]) -> dict:
    """
    Runs all retrieval metrics for a single question's retrieved chunks.
    """
    return {
        "hit_rate_at_3": hit_rate_at_k(retrieved_sources, relevant_companies, 3),
        "hit_rate_at_5": hit_rate_at_k(retrieved_sources, relevant_companies, 5),
        "precision_at_3": precision_at_k(retrieved_sources, relevant_companies, 3),
        "precision_at_5": precision_at_k(retrieved_sources, relevant_companies, 5),
        "mrr": mrr(retrieved_sources, relevant_companies),
    }