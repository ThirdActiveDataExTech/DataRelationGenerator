"""Dataset(테이블) 관련 스키마 정의."""

import uuid
from enum import Enum
from typing import Any, Literal

from openai import pydantic_function_tool
from pydantic import BaseModel, Field


class Level1RelationType(str, Enum):
    """Level 1 (데이터셋 간) 관계 유형 — "최소 필수" 3종 (§7.4, §10.1).

    - JOINABLE: 공통 키/식별자로 직접 결합 가능 (주로 rule/Tier1)
    - SEMANTICALLY_RELATED: 같은 주제·도메인을 다루어 함께 분석할 가치 (embedding/llm)
    - COMPLEMENTARY: 한쪽이 다른 쪽을 보강·확장 (llm)

    도메인별 세분 유형은 relation_type 문자열의 하위 라벨로 확장 가능(개방형 스키마).
    """

    JOINABLE = "JOINABLE"
    SEMANTICALLY_RELATED = "SEMANTICALLY_RELATED"
    COMPLEMENTARY = "COMPLEMENTARY"


def canonicalize_level1_type(raw_type: str, has_common_columns: bool) -> str:
    """LLM/규칙이 산출한 자유 텍스트 관계 유형을 Level 1 표준 3종으로 정규화.

    원본 라벨은 하위 라벨로 보존한다 (예: "COMPLEMENTARY:병원기본-병원장비").

    Args:
        raw_type: 원본 관계 유형 텍스트
        has_common_columns: Tier1 공통 컬럼 존재 여부 (조인 가능성 신호)

    Returns:
        "<표준유형>" 또는 "<표준유형>:<원본라벨>"
    """
    text = (raw_type or "").lower()
    if has_common_columns or any(k in text for k in ("join", "조인", "fk", "공통", "식별자", "key")):
        canonical = Level1RelationType.JOINABLE
    elif any(k in text for k in ("complement", "보완", "보강", "확장", "extend")):
        canonical = Level1RelationType.COMPLEMENTARY
    else:
        canonical = Level1RelationType.SEMANTICALLY_RELATED

    raw_clean = (raw_type or "").strip()
    if raw_clean and raw_clean.upper() not in {t.value for t in Level1RelationType}:
        return f"{canonical.value}:{raw_clean}"
    return canonical.value


class DatasetProfile(BaseModel):
    """테이블 레벨 메타데이터 프로파일."""

    dataset_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="OpenSearch 문서 ID (UUID)")
    table_name: str = Field(description="테이블명 (식별자)")
    domain: str = Field(description="데이터 도메인 (medical, broadcast, general 등)")
    record_count: int = Field(description="레코드 수")
    column_count: int = Field(description="컬럼 수")
    column_names: list[str] = Field(description="컬럼명 목록")
    sample_records: list[dict[str, Any]] = Field(description="샘플 레코드 (최대 10개)")
    description: str = Field(default="", description="테이블 짧은 설명")
    table_summary: str = Field(default="", description="테이블 상세 요약 (임베딩 소스)")
    table_embedding: list[float] = Field(default_factory=list, description="table_summary 임베딩 벡터")
    keywords: list[str] = Field(default_factory=list, description="LLM이 추출한 도메인 개념 키워드")
    keywords_embedding: list[float] = Field(default_factory=list, description="keywords 평균 임베딩 벡터")
    created_at: str = Field(default="", description="생성 시각 (ISO 8601)")
    raw_storage_key: str = Field(default="", description="원본 업로드 파일 보관 키 (재처리용, RAW_STORAGE 백엔드 기준)")


class DatasetRelation(BaseModel):
    """테이블 간 관계 정보."""

    table_a: str = Field(description="출발 테이블명")
    table_b: str = Field(description="도착 테이블명")
    score: float = Field(ge=0.0, le=1.0, description="관계 점수")
    method: str = Field(description="생성 방법 (rule, embedding, llm)")
    relation_type: str = Field(description="관계 유형 설명 (예: FK, 진단명-건강정보)")


class DatasetIngestRequest(BaseModel):
    """테이블 메타데이터 적재 요청."""

    table_name: str = Field(description="테이블명")
    domain: str = Field(description="도메인 (medical, broadcast, general 등)")
    description: str = Field(default="", description="짧은 설명")
    table_summary: str = Field(default="", description="상세 요약 (비어있으면 description 사용)")


class DatasetSearchRequest(BaseModel):
    """테이블 KNN 검색 요청."""

    query: str = Field(description="검색 쿼리 텍스트")
    domain: str | None = Field(default=None, description="도메인 필터 (None이면 전체)")
    top_k: int = Field(default=5, ge=1, le=20, description="반환할 최대 결과 수")


class DatasetSearchResult(BaseModel):
    """테이블 KNN 검색 결과 단건."""

    table_name: str
    domain: str
    description: str
    table_summary: str
    score: float


# ── Tier 1 ──────────────────────────────────────────────────────────────────

class ColumnMatchInfo(BaseModel):
    """Tier 1 컬럼 매칭 결과."""

    column_a: str = Field(description="테이블 A의 컬럼명")
    column_b: str = Field(description="테이블 B의 컬럼명")
    match_type: Literal["exact_name", "fk", "code_dict"] = Field(description="매칭 방식")


class Tier1Result(BaseModel):
    """Tier 1 자동 확정 관계."""

    table_a: str
    table_b: str
    relation_type: str = Field(description="FK / 동일컬럼 / 코드사전")
    matched_columns: list[ColumnMatchInfo]
    score: float = Field(default=1.0)
    method: Literal["rule"] = "rule"


# ── Tier 2 ──────────────────────────────────────────────────────────────────

class Tier2Candidate(BaseModel):
    """Tier 2 KNN 유사도 후보."""

    table_a: str
    table_b: str
    score: float = Field(ge=0.0, le=1.0, description="table_embedding 코사인 유사도")
    method: Literal["embedding"] = "embedding"


# ── Tier 3 ──────────────────────────────────────────────────────────────────

EvidenceType = Literal[
    "shared_key", "value_overlap", "same_entity", "parallel_schema", "usage_dependency", "none"
]


class TableRelationVerification(BaseModel):
    """LLM 관계 검증 결과 (tool call 스키마)."""

    related: bool = Field(description="두 테이블이 의미적으로 연관되어 있는가")
    evidence_type: EvidenceType = Field(
        description="관계 근거 유형 — 근거가 테이블 데이터 안에 존재해야 함 (related=false면 none)"
    )
    relation_type: str = Field(description="관계 유형 설명 (예: 진단명-건강정보, 환자-병원)")
    confidence: float = Field(ge=0.0, le=1.0, description="확신도")
    reason: str = Field(description="판단 근거 (2~3문장)")


table_relation_verification_tool = pydantic_function_tool(
    TableRelationVerification,
    name="verify_table_relation",
    description="두 테이블 간 의미적 관계 여부를 판단합니다.",
)


class TableMetadataExtraction(BaseModel):
    """LLM 테이블 메타데이터 추출 결과 (tool call 스키마)."""

    description: str = Field(description="테이블 목적과 내용을 1~2문장으로 요약한 설명")
    keywords: list[str] = Field(description="테이블을 대표하는 도메인 개념 키워드 5~10개 (한국어)")


table_metadata_extraction_tool = pydantic_function_tool(
    TableMetadataExtraction,
    name="extract_table_metadata",
    description="테이블 메타데이터에서 설명과 키워드를 추출합니다.",
)


class Tier3Result(BaseModel):
    """Tier 3 LLM 검증 통과 결과."""

    table_a: str
    table_b: str
    relation_type: str
    evidence_type: str = Field(default="", description="LLM이 분류한 관계 근거 유형")
    confidence: float
    reason: str
    embedding_score: float = Field(description="Tier 2 유사도 (참고용)")
    method: Literal["llm"] = "llm"


# ── 전체 파이프라인 결과 ──────────────────────────────────────────────────────

class RelationPipelineResult(BaseModel):
    """3-Tier 관계 생성 파이프라인 전체 결과."""

    table_a: str
    table_b: str
    tier1: list[Tier1Result] = Field(default_factory=list)
    tier2_candidates: list[Tier2Candidate] = Field(default_factory=list)
    tier3_verified: list[Tier3Result] = Field(default_factory=list)
    saved_relations: list[DatasetRelation] = Field(default_factory=list, description="Neptune에 저장된 관계")


class RunRelationPipelineRequest(BaseModel):
    """관계 생성 파이프라인 실행 요청."""

    table_names: list[str] = Field(description="대상 테이블명 목록 (비어있으면 전체)")
    tier2_threshold: float = Field(default=0.4, ge=0.0, le=1.0, description="Tier 2 유사도 임계값")
    tier3_confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Tier 3 확신도 임계값")
