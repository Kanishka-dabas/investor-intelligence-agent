import re
from pathlib import Path

from loguru import logger

FINANCIAL_KEYWORDS = [
    "balance sheet",
    "income statement",
    "cash flow",
    "revenue",
    "net income",
    "shareholders equity",
    "shareholders' equity",
    "annual report",
    "quarterly report",
    "10-k",
    "10-q",
    "fiscal year",
    "consolidated statements",
    "earnings per share",
    "operating expenses",
    "total assets",
    "total liabilities",
]

ACCEPT_THRESHOLD = 5   # keyword matches >= this -> confidently financial
REJECT_THRESHOLD = 1   # keyword matches <= this -> confidently NOT financial


def count_keyword_matches(text: str) -> int:
    """Count how many distinct financial keywords appear in the text."""
    normalized_text = " ".join(text.lower().split())
    return sum(1 for kw in FINANCIAL_KEYWORDS if kw in normalized_text)


def heuristic_check(text: str) -> str:
    """
    Returns "accept", "reject", or "ambiguous" based on keyword density.
    """
    match_count = count_keyword_matches(text)
    logger.info(f"Financial keyword matches: {match_count}")

    if match_count >= ACCEPT_THRESHOLD:
        return "accept"
    if match_count <= REJECT_THRESHOLD:
        return "reject"
    return "ambiguous"


def is_financial_document(markdown_text: str) -> bool:
    """
    Validates whether a document is a financial report before it enters
    the expensive ingestion pipeline (embedding, KPI extraction, etc.)

    Uses a fast keyword heuristic first; only falls back to an LLM call
    for ambiguous cases, to avoid unnecessary API cost/latency.
    """
    verdict = heuristic_check(markdown_text)

    if verdict == "accept":
        logger.info("Document accepted (heuristic: clearly financial)")
        return True
    if verdict == "reject":
        logger.info("Document rejected (heuristic: clearly not financial)")
        return False

    # Ambiguous -> fall back to LLM check (placeholder for now)
    logger.info("Ambiguous case, LLM check needed (not yet implemented)")
    return _llm_fallback_check(markdown_text)


def _llm_fallback_check(markdown_text: str) -> bool:
    """
    Placeholder — will call the LLM gateway (src/llm/) once that module
    is built, to classify ambiguous documents.
    """
    raise NotImplementedError(
        "LLM fallback for ambiguous documents will be wired up once "
        "src/llm/ (LLM gateway) is built."
    )


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    markdown_dir = repo_root / "data" / "markdown"

    for md_file in sorted(markdown_dir.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        try:
            result = is_financial_document(text)
            logger.info(f"{md_file.name}: {'VALID' if result else 'REJECTED'}")
        except NotImplementedError as e:
            logger.warning(f"{md_file.name}: {e}")