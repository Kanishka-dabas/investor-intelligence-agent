from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.orm import declarative_base
from datetime import datetime, timezone

Base = declarative_base()


class FinancialKPIRecord(Base):
    """Stores structured KPIs extracted from each financial report."""
    __tablename__ = "financial_kpis"

    id = Column(Integer, primary_key=True, autoincrement=True)
    company_name = Column(String, nullable=False)
    fiscal_period = Column(String, nullable=False)
    revenue = Column(Float, nullable=True)
    revenue_unit = Column(String, nullable=True)
    net_income = Column(Float, nullable=True)
    net_income_unit = Column(String, nullable=True)
    risk_factors = Column(JSON, default=list)
    growth_drivers = Column(JSON, default=list)
    source_file = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class IngestedDocument(Base):
    """
    Tracks which source files have already been ingested, so re-running
    the ingestion pipeline (e.g. on container restart) skips duplicates.
    """
    __tablename__ = "ingested_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_file = Column(String, nullable=False, unique=True)
    file_hash = Column(String, nullable=False)
    chunk_count = Column(Integer, nullable=False)
    ingested_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))