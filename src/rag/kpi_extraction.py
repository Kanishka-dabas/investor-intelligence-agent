from src.llm.gateway import LLMGateway
from src.rag.schemas import FinancialKPI
from loguru import logger

EXTRACTION_SYSTEM_PROMPT = """You are a financial analyst extracting structured data from
company financial reports. Extract only information explicitly present in the text.
If a field is not clearly stated, leave it as null (for numbers) or an empty list
(for risk_factors/growth_drivers). Do not estimate or infer missing numbers.

IMPORTANT: Financial statement tables often show multiple columns for different periods
side by side — e.g. "Three Months Ended" vs "Six Months Ended" (or "Nine Months Ended"),
and current year vs prior year. Always extract the figure for the SINGLE MOST RECENT
QUARTER (the latest "Three Months Ended" column), NOT the cumulative six-month/nine-month
figure, and NOT the prior-year comparison column. Double-check which column header each
number falls under before extracting it."""


def extract_kpis(report_text: str, source_file: str, gateway: LLMGateway) -> FinancialKPI:
    """
    Extracts structured financial KPIs from report text using the LLM
    gateway's structured output capability.
    """
    chat_model = gateway.get_chat_model("groq")
    structured_model = chat_model.with_structured_output(FinancialKPI, method="json_schema")

    prompt = f"{EXTRACTION_SYSTEM_PROMPT}\n\nReport text:\n{report_text}"

    try:
        result: FinancialKPI = structured_model.invoke(prompt)
        result.source_file = source_file
        return result
    except Exception as e:
        logger.error(f"KPI extraction failed for {source_file}: {e}")
        raise


def find_financial_statements_section(text: str, window_size: int = 12000) -> str:
    """
    Locates the ACTUAL financial statements table in a long report (not
    just a Table-of-Contents mention of it) by searching for section
    headers followed by dollar-amount numbers within a short window.
    """
    import re

    markers = [
        "condensed consolidated statements of operations",
        "consolidated statements of operations",
        "condensed consolidated statements of income",
        "consolidated statements of income",
        "condensed consolidated balance sheet",
        "consolidated balance sheet",
    ]
    lower_text = text.lower()

    for marker in markers:
        for match in re.finditer(re.escape(marker), lower_text):
            idx = match.start()
            window_ahead = text[idx: idx + 800]
            if re.search(r"\$\s?[\d,]+", window_ahead):
                start = max(0, idx - 200)
                end = min(len(text), idx + window_size)
                return text[start:end]

    # Fallback: no marker matched near numbers, take first window_size chars
    return text[:window_size]