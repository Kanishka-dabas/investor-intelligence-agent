import requests
import json

from src.config import settings


API_BASE_URL = "http://127.0.0.1:8000"
BEARER_TOKEN = settings.security.bearer_token.get_secret_value()


def stream_chat(query: str):
    """
    Calls /chat/stream and yields (event_type, content) tuples —
    event_type is "status" (agent progress) or "chunk" (answer text).
    """
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json",
    }
    with requests.post(
        f"{API_BASE_URL}/chat/stream",
        headers=headers,
        json={"query": query},
        stream=True,
    ) as response:
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if data.get("type") == "status":
                    yield "status", data["message"]
                elif data.get("type") == "chunk":
                    yield "chunk", data["content"]


def get_kpis_from_db():
    from src.database.connection import SessionLocal
    from src.database.models import FinancialKPIRecord
    from src.rag.kpi_extraction import find_financial_statements_section

    db = SessionLocal()
    records = db.query(FinancialKPIRecord).all()
    db.close()
    return records


def ingest_uploaded_pdf(file_path: str, filename: str):
    """
    Runs the full ingestion pipeline on an uploaded PDF: convert to
    markdown, chunk, validate, embed into Qdrant, extract KPIs into Postgres.
    """
    from src.ingestion.pdf_to_markdown import PDFToMarkdownConverter
    from src.ingestion.chunking import chunk_markdown
    from src.ingestion.validation import is_financial_document
    from src.retrieval.vector_store import QdrantVectorStore
    from src.rag.kpi_extraction import extract_kpis
    from src.llm.gateway import LLMGateway
    from src.database.connection import SessionLocal
    from src.database.kpi_storage import save_kpi
    from src.database.ingestion_tracker import compute_file_hash, is_already_ingested, mark_as_ingested
    from pathlib import Path

    db = SessionLocal()
    file_hash = compute_file_hash(file_path)

    if is_already_ingested(db, filename, file_hash):
        db.close()
        return {"status": "skipped", "message": f"{filename} already ingested."}

    converter = PDFToMarkdownConverter()
    md_path = converter.convert_pdf(file_path, output_dir="data/markdown")

    md_text = Path(md_path).read_text(encoding="utf-8")
    if not is_financial_document(md_text):
        db.close()
        return {"status": "rejected", "message": f"{filename} does not appear to be a financial report."}

    chunks = chunk_markdown(md_path)
    for c in chunks:
        c.metadata["source"] = filename

    vs = QdrantVectorStore(host="localhost", port=6333, collection_name="financial_reports")
    vs.upsert_chunks([c.page_content for c in chunks], [c.metadata for c in chunks])

    from src.rag.kpi_extraction import find_financial_statements_section

    layout_text = converter.extract_layout_text(file_path)
    targeted_text = find_financial_statements_section(layout_text, window_size=6000)

    gateway = LLMGateway()
    kpi_result = extract_kpis(targeted_text, filename, gateway)
    save_kpi(db, kpi_result)

    mark_as_ingested(db, filename, file_hash, chunk_count=len(chunks))
    db.close()

    return {"status": "success", "message": f"{filename} ingested: {len(chunks)} chunks, KPIs extracted."}