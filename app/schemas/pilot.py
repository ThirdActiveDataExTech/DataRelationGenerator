"""실증 기관 스키마."""

from pydantic import BaseModel, Field

from app.schemas.explore import DatasetCard


class PilotOrg(BaseModel):
    """실증 기관 — 기관 보유 데이터(1st 파티)는 provider 로 식별한다."""

    org_id: str = Field(description="기관 식별자 (URL 경로용)")
    name: str = Field(description="기관 표시명")
    providers: list[str] = Field(description="이 기관 보유 데이터로 보는 provider 값")
    summary: str = Field(description="실증 화면 안내 문구")


class PilotOrgSummary(PilotOrg):
    """기관 + 보유 데이터 목록."""

    datasets: list[DatasetCard] = Field(default_factory=list)
