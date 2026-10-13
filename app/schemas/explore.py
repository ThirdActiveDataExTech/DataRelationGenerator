"""데이터 찾기 화면 스키마 — 분야 탐색, 검색 결과 카드, 상세, 함께 쓸 데이터."""

from typing import Any

from pydantic import BaseModel, Field


class TaxonomyNode(BaseModel):
    """분류 노드 (대분류 또는 중분류)와 데이터셋 수."""

    name: str
    count: int = 0
    children: list["TaxonomyNode"] = Field(default_factory=list)


class DatasetCard(BaseModel):
    """검색 결과·목록 카드."""

    table_name: str
    display_name: str = ""
    description: str = ""
    table_summary: str = ""
    provider: str = ""
    party: str = ""
    source: str = Field(default="", description="원천 포털 코드 (DATA_GO_KR / KOSIS / HIRA / FIRST_PARTY / OTHER)")
    categories: list[str] = Field(default_factory=list, description="'대분류 > 중분류' 1~3개")
    record_count: int = 0
    column_count: int = 0
    landing_page: str = ""
    keywords: list[str] = Field(default_factory=list)
    relation_count: int = Field(default=0, description="확정된 연계 관계 수")
    similarity: float | None = Field(default=None, description="검색어 의미 유사도 (검색어가 있을 때)")
    created_at: str = ""


class SearchPage(BaseModel):
    """검색 결과 한 페이지."""

    total: int
    items: list[DatasetCard]


class DatasetDetail(DatasetCard):
    """상세 — 카드 + 컬럼·샘플."""

    column_names: list[str] = Field(default_factory=list)
    sample_records: list[dict[str, Any]] = Field(default_factory=list)


class LinkableItem(BaseModel):
    """연계 가능한 데이터 1건."""

    table_name: str
    display_name: str = ""
    provider: str = ""
    description: str = ""
    landing_page: str = ""
    relation_type: str = Field(default="", description="관계 유형 원문 (판정 전 후보는 빈 값)")
    sub_label: str = Field(default="", description="관계 유형 하위라벨")
    score: float = Field(description="관계 신뢰도 (판정 전 후보는 의미 유사도)")
    join_keys: list[str] = Field(default_factory=list, description="결합 컬럼 (기준 컬럼 = 상대 컬럼)")
    reason: str = Field(default="", description="두 데이터를 함께 쓰는 상황 설명 (없으면 관계 판정 이유)")


class LinkableGroup(BaseModel):
    """관계 유형별 묶음 — 화면 배치 순서대로 정렬된다."""

    relation_type: str
    label: str
    description: str
    items: list[LinkableItem] = Field(default_factory=list)


class LinkableResult(BaseModel):
    """기준 데이터 1건과 함께 쓸 수 있는 데이터."""

    anchor: DatasetCard
    groups: list[LinkableGroup]
    candidates: list[LinkableItem] = Field(
        default_factory=list, description="관계 판정 전 의미 유사 후보 — 관계 생성 실행 대상"
    )
    graph: dict[str, Any] = Field(default_factory=dict, description="기준 1-hop 부분 그래프")
