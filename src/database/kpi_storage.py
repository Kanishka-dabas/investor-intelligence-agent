from sqlalchemy.orm import Session

from src.database.models import FinancialKPIRecord
from src.rag.schemas import FinancialKPI


def save_kpi(db: Session, kpi: FinancialKPI) -> FinancialKPIRecord:
    """Persists an extracted FinancialKPI (Pydantic) into Postgres."""
    record = FinancialKPIRecord(
        company_name=kpi.company_name,
        fiscal_period=kpi.fiscal_period,
        revenue=kpi.revenue,
        revenue_unit=kpi.revenue_unit,
        net_income=kpi.net_income,
        net_income_unit=kpi.net_income_unit,
        risk_factors=kpi.risk_factors,
        growth_drivers=kpi.growth_drivers,
        source_file=kpi.source_file,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record