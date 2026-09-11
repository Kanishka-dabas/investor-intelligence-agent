from pydantic import BaseModel, Field
from typing import Literal


class FinancialKPI(BaseModel):
    """Structured financial KPIs extracted from a report."""

    company_name: str = Field(description="Name of the company")
    fiscal_period: str = Field(description="Fiscal period, e.g. 'Q3 FY2026' or 'FY2025'")
    revenue: float | None = Field(default=None, description="Total revenue, in the report's stated currency/units")
    revenue_unit: str | None = Field(default=None, description="Unit of revenue, e.g. 'USD millions'")
    net_income: float | None = Field(default=None, description="Net income, in the report's stated currency/units")
    net_income_unit: str | None = Field(default=None, description="Unit of net income")
    risk_factors: list[str] = Field(default_factory=list, description="Key risk factors mentioned in the report")
    growth_drivers: list[str] = Field(default_factory=list, description="Key growth drivers mentioned in the report")
    source_file: str | None = Field(default=None, description="Source markdown/PDF filename this was extracted from")