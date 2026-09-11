from src.ingestion.validation import count_keyword_matches, heuristic_check


def test_count_keyword_matches_finds_financial_terms():
    text = "The balance sheet shows total assets and net income for this quarter."
    count = count_keyword_matches(text)
    assert count >= 2


def test_count_keyword_matches_zero_for_unrelated_text():
    text = "The weather today is sunny with a chance of rain in the afternoon."
    count = count_keyword_matches(text)
    assert count == 0


def test_heuristic_check_accepts_clearly_financial_document():
    text = """
    CONSOLIDATED BALANCE SHEET
    Revenue: $100 million
    Net income: $20 million
    Cash flow from operations
    Annual report fiscal year 2026
    Total assets and shareholders equity
    """
    result = heuristic_check(text)
    assert result == "accept"


def test_heuristic_check_rejects_clearly_non_financial_document():
    text = "This is a recipe for chocolate chip cookies. Mix flour, sugar, and butter."
    result = heuristic_check(text)
    assert result == "reject"


def test_heuristic_check_ambiguous_for_borderline_text():
    text = "The company reported revenue growth and discussed its annual report briefly."
    result = heuristic_check(text)
    assert result in ("accept", "ambiguous")