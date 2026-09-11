import pytest
from src.ingestion.chunking import extract_tables, is_valid_chunk
from langchain_core.documents import Document


def test_extract_tables_finds_markdown_table():
    text = """Some intro text.

|Header 1|Header 2|
|---|---|
|Value 1|Value 2|

More text after."""
    text_without_tables, tables = extract_tables(text)

    assert len(tables) == 1
    assert "[TABLE_PLACEHOLDER]" in text_without_tables
    assert "Value 1" in tables[0]


def test_extract_tables_no_table_present():
    text = "Just plain text with no tables at all."
    text_without_tables, tables = extract_tables(text)

    assert len(tables) == 0
    assert text_without_tables == text


def test_is_valid_chunk_rejects_short_text():
    chunk = Document(page_content="Too short")
    assert is_valid_chunk(chunk) is False


def test_is_valid_chunk_accepts_normal_financial_text():
    chunk = Document(
        page_content="Revenue increased by 18% year over year driven by strong "
        "growth in cloud services and continued demand across all business segments."
    )
    assert is_valid_chunk(chunk) is True


def test_is_valid_chunk_rejects_agm_notice_boilerplate():
    chunk = Document(
        page_content="Notice is hereby given that the Annual General Meeting "
        "of shareholders will be held at the registered office of the company."
    )
    assert is_valid_chunk(chunk) is False


def test_is_valid_chunk_rejects_high_contact_info_density():
    chunk = Document(
        page_content="www.example.com\ncontact@example.com\nTel. +1-234-567\n"
        "Fax +1-234-568\nwww.another.com"
    )
    assert is_valid_chunk(chunk) is False