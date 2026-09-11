import hashlib
from pathlib import Path

from sqlalchemy.orm import Session
from loguru import logger

from src.database.models import IngestedDocument


def compute_file_hash(file_path: str) -> str:
    """Computes SHA256 hash of a file's content."""
    content = Path(file_path).read_bytes()
    return hashlib.sha256(content).hexdigest()


def is_already_ingested(db: Session, source_file: str, file_hash: str) -> bool:
    """
    Checks if a file (by name + content hash) has already been ingested.
    Returns True only if both the filename AND its content hash match a
    prior record — so an updated file (same name, changed content) is
    correctly treated as needing re-ingestion.
    """
    existing = db.query(IngestedDocument).filter_by(source_file=source_file).first()
    if existing is None:
        return False
    if existing.file_hash != file_hash:
        logger.info(f"{source_file} content changed since last ingestion, will re-ingest.")
        return False
    logger.info(f"{source_file} already ingested (unchanged), skipping.")
    return True


def mark_as_ingested(db: Session, source_file: str, file_hash: str, chunk_count: int) -> None:
    """Records a file as ingested, or updates the record if it already existed (re-ingestion case)."""
    existing = db.query(IngestedDocument).filter_by(source_file=source_file).first()
    if existing:
        existing.file_hash = file_hash
        existing.chunk_count = chunk_count
    else:
        db.add(IngestedDocument(source_file=source_file, file_hash=file_hash, chunk_count=chunk_count))
    db.commit()