import re
from loguru import logger

from src.llm.gateway import LLMGateway

# Common prompt-injection patterns — fast, free, catches obvious attempts
INJECTION_PATTERNS = [
    r"ignore (all |the )?(previous|above|prior) instructions",
    r"disregard (all |the )?(previous|above|prior) instructions",
    r"you are now",
    r"forget (everything|all) (you were|i) told",
    r"system prompt",
    r"act as (a|an)(?! financial)",
    r"jailbreak",
    r"reveal your (instructions|prompt|system message)",
    r"pretend (you are|to be)",
]

# Keywords that indicate an on-topic financial query
FINANCIAL_TOPIC_KEYWORDS = [
    "revenue", "net income", "profit", "loss", "earnings", "balance sheet",
    "cash flow", "assets", "liabilities", "equity", "financial", "fiscal",
    "quarter", "annual report", "kpi", "growth", "risk factor", "stock",
    "share", "dividend", "expense", "cost", "margin", "company", "report",
]


def detect_prompt_injection(query: str) -> bool:
    """
    Fast regex-based check for common prompt-injection attempts.
    Returns True if injection is suspected.
    """
    lower_query = query.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lower_query):
            logger.warning(f"Prompt injection pattern matched: '{pattern}' in query.")
            return True
    return False


def is_on_topic(query: str) -> bool:
    """
    Fast heuristic: does the query contain at least one financial keyword?
    This is intentionally permissive (low false-reject rate) — ambiguous
    queries are allowed through rather than blocked, since blocking a
    legitimate question is worse than answering an off-topic one grounded
    in "no relevant context found".
    """
    lower_query = query.lower()
    return any(keyword in lower_query for keyword in FINANCIAL_TOPIC_KEYWORDS)


def validate_query(query: str, gateway: LLMGateway | None = None) -> tuple[bool, str]:
    """
    Runs guardrail checks on an incoming query before it reaches the
    RAG/MCP pipeline.

    Returns (is_valid, reason). If is_valid is False, the caller should
    reject the query and return `reason` to the user instead of processing it.
    """
    if detect_prompt_injection(query):
        return False, "This query appears to contain an instruction-injection attempt and was blocked."

    if not is_on_topic(query):
        # Not blocking outright — just logging. RAG's own "no context found"
        # response already handles genuinely off-topic questions gracefully.
        logger.info(f"Query has no obvious financial keywords: '{query}'")

    return True, ""