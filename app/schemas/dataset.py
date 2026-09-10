"""Dataset(테이블) 관련 스키마 정의."""

import uuid
from enum import Enum
from typing import Any, Literal

from app.schemas.tool_utils import pydantic_function_tool
from pydantic import BaseModel, Field


class Level1RelationType(str, Enum):
    """데이터셋 간 관계 유형 4종 (관계 유형 정의서 0항).

    - DETAIL_FOR: 같은 개체에 앵커가 갖지 않은 속성을 공급
    - STATISTIC_FOR: 다른 입도의 집계·지수로 앵커 값에 맥락을 제공
    - ADJACENT_TO: 같은 맥락의 다른 개체 (대체재·다음 단계·주제 인접)
    - SAME_KIND: 같은 표준을 따르는 다른 지역·유형·판본

    하위라벨은 relation_type 문자열에 `:` 로 이어 붙인다 (개방형 스키마).
    """

    DETAIL_FOR = "DETAIL_FOR"
    STATISTIC_FOR = "STATISTIC_FOR"
    ADJACENT_TO = "ADJACENT_TO"
    SAME_KIND = "SAME_KIND"


# 컬럼 자카드가 이 값 이상이면 같은 종류로 본다.
# ponytail: 파일럿 6종 표본 추정값. 표준 준수 쌍으로 재측정 예정 (정의서 9항 1번)
SAME_KIND_JACCARD_THRESHOLD = 0.6

# 4종 이전에 저장된 값 → 현행 유형. 원본은 하위라벨로 보존한다.
_LEGACY_TYPE_MAP = {
    "JOINABLE": Level1RelationType.DETAIL_FOR,
    "COMPLEMENTARY": Level1RelationType.DETAIL_FOR,
    "SEMANTICALLY_RELATED": Level1RelationType.ADJACENT_TO,
}

_AGGREGATE_HINTS = ("통계", "집계", "지표", "현황 통계", "statistic", "aggregate")


def column_jaccard(columns_a: list[str], columns_b: list[str]) -> float:
    """두 컬럼 목록의 자카드 유사도. 공백·대소문자 차이는 무시한다."""
    norm = lambda cols: {c.strip().lower().replace(" ", "") for c in cols if c and c.strip()}
    a, b = norm(columns_a), norm(columns_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# 집계표는 시점을 행의 구분자로 갖고 수치 측정값을 나른다. 개체 목록은 시점을
# 개체의 사건 일자(인허가일자 등)로만 가지므로 구분된다.
# ponytail: dataset_metadata.is_aggregate 가 채워지면 이 추정을 대체한다 (정의서 9항)
_PERIOD_COLUMNS = ("연도", "년도", "기준연월", "기준년월", "기준시점", "시점", "기간", "year", "month")
_MEASURE_TAILS = ("수", "건수", "명", "율", "률", "액", "합계", "평균", "count", "total", "sum", "avg", "rate")


def looks_aggregate(columns: list[str]) -> bool:
    """컬럼 구성으로 집계표인지 추정한다.

    시점 구분 컬럼과 수치 측정 컬럼을 함께 가지면 집계표로 본다. 개체 목록은
    시점을 행 구분자로 쓰지 않으므로 이 조합이 나오지 않는다.
    """
    norm = [c.strip().lower().replace(" ", "") for c in columns if c and c.strip()]
    has_period = any(tok in c for c in norm for tok in _PERIOD_COLUMNS)
    has_measure = any(c.endswith(tail) for c in norm for tail in _MEASURE_TAILS)
    return has_period and has_measure


def migrate_stored_type(stored: str) -> str:
    """저장된 관계 유형을 현행 4종으로 옮긴다 (조회 시점).

    정규화는 쓰기 시점에만 걸리므로 그 이전에 적재된 관계는 구 유형을 그대로
    갖고 있다. 조회 응답이 구 유형을 노출하지 않도록 앞 토큰만 바꾸고
    설명 하위라벨은 보존한다. 백필 없이 구·신 데이터를 함께 다룬다.
    """
    head, _, sub = (stored or "").partition(":")
    migrated = _LEGACY_TYPE_MAP.get(head.strip().upper())
    if migrated is None:
        return stored
    return f"{migrated.value}:{sub}" if sub else migrated.value


def _as_level1_type(value: str | None) -> "Level1RelationType | None":
    """LLM이 준 유형 문자열을 enum 으로. 알 수 없는 값은 무시한다."""
    try:
        return Level1RelationType(str(value).strip().upper())
    except ValueError:
        return None


def canonicalize_level1_type(
    raw_type: str,
    has_common_columns: bool,
    *,
    jaccard: float | None = None,
    conforms_to_match: bool = False,
    is_aggregate: bool = False,
    llm_type: str | None = None,
) -> str:
    """관계 유형을 4종으로 정규화. 원본 라벨은 하위라벨로 보존한다.

    판정 순서 (정의서 5항) — 앞선 조건이 성립하면 뒤는 보지 않는다.

    1. 같은 표준을 따르거나 컬럼 자카드가 임계 이상 → SAME_KIND
    2. 상대가 집계표 → STATISTIC_FOR
    3. LLM 판정 (구조 신호가 결론을 내지 못한 경우)
    4. 공통 컬럼이 있다 (같은 개체) → DETAIL_FOR
    5. 그 외 → ADJACENT_TO

    구조 신호(1·2)는 확인 가능하므로 LLM 판정보다 우선한다. 카탈로그 메타데이터만
    있어 컬럼이 비면 구조 신호가 아예 없는데, 그때 LLM 판정이 유일한 근거가 된다.

    Args:
        raw_type: LLM/규칙이 산출한 원본 관계 유형 텍스트
        has_common_columns: 공통 컬럼 존재 여부 (Tier1/Tier1.5)
        jaccard: 컬럼 자카드. None 이면 SAME_KIND 판정을 건너뛴다
        conforms_to_match: 양측이 같은 개방표준을 따르는지
        is_aggregate: 상대가 집계표인지. dataset_metadata 확보 후 채워진다
        llm_type: LLM이 판정한 4종 유형. 구프롬프트 응답이면 빈 값

    Returns:
        "<유형>" 또는 "<유형>:<원본라벨>"
    """
    text = (raw_type or "").lower()
    legacy = _LEGACY_TYPE_MAP.get((raw_type or "").strip().upper())

    judged = _as_level1_type(llm_type)

    if conforms_to_match or (jaccard is not None and jaccard >= SAME_KIND_JACCARD_THRESHOLD):
        canonical = Level1RelationType.SAME_KIND
    elif is_aggregate:
        canonical = Level1RelationType.STATISTIC_FOR
    elif legacy is not None:
        # 4종 이전 값은 텍스트 키워드보다 매핑을 우선한다
        canonical = legacy
    elif judged is not None:
        canonical = judged
    elif any(k in text for k in _AGGREGATE_HINTS):
        # 구프롬프트 응답 대비 — 라벨 문구로 집계를 추정한다
        canonical = Level1RelationType.STATISTIC_FOR
    elif has_common_columns:
        canonical = Level1RelationType.DETAIL_FOR
    else:
        canonical = Level1RelationType.ADJACENT_TO

    raw_clean = (raw_type or "").strip()
    if raw_clean and raw_clean.upper() not in {t.value for t in Level1RelationType}:
        return f"{canonical.value}:{raw_clean}"
    return canonical.value


class DatasetProfile(BaseModel):
    """테이블 레벨 메타데이터 프로파일."""

    dataset_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="OpenSearch 문서 ID (UUID)")
    table_name: str = Field(description="테이블명 (원문 식별용)")
    display_name: str = Field(default="", description="한국어 표시명 (LLM 자동 생성, 비어있으면 table_name 사용)")
    catalog_identifier: str = Field(
        default="",
        description="통합 카탈로그 식별자 (① catalog_entry.identifier, DCAT dct:identifier). 연동 시 필수 키",
    )
    party: Literal["", "1st", "3rd"] = Field(
        default="", description="데이터 파티 구분 — 1st(기관 보유 데이터) / 3rd(외부 개방 데이터). 빈 값은 미지정"
    )
    provider: str = Field(default="", description="제공 기관/발행처 (DCAT dct:publisher)")
    landing_page: str = Field(
        default="", description="원본 제공처 URL (DCAT dcat:landingPage/accessURL) — 사용자 시나리오 '수집' 단계 연결용"
    )
    domain: str = Field(description="데이터 도메인 (medical, broadcast, general 등)")
    domain_confidence: float | None = Field(
        default=None, description="도메인 자동 분류 신뢰도 (None = 사용자/카탈로그 지정)"
    )
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
    catalog_identifier_a: str = Field(default="", description="출발 데이터셋의 통합 카탈로그 식별자 (①연동 키)")
    catalog_identifier_b: str = Field(default="", description="도착 데이터셋의 통합 카탈로그 식별자 (①연동 키)")
    score: float = Field(ge=0.0, le=1.0, description="관계 점수")
    method: str = Field(description="생성 방법 (rule, embedding, llm)")
    relation_type: str = Field(description="관계 유형 설명 (예: FK, 진단명-건강정보)")
    evidence: dict[str, Any] = Field(
        default_factory=dict,
        description="생성 근거 (LLM reason, 매칭 컬럼, 임베딩 점수 등) — ① 환류 시 검토 근거로 사용",
    )


class DatasetIngestRequest(BaseModel):
    """테이블 메타데이터 적재 요청."""

    table_name: str = Field(description="테이블명")
    domain: str = Field(default="", description="도메인 (medical, broadcast 등). 빈 값이면 특징 기반 자동 분류")
    catalog_identifier: str = Field(default="", description="통합 카탈로그 식별자 (catalog_entry.identifier)")
    description: str = Field(default="", description="짧은 설명")
    table_summary: str = Field(default="", description="상세 요약 (비어있으면 description 사용)")
    party: Literal["", "1st", "3rd"] = Field(default="", description="데이터 파티 구분 (1st/3rd)")
    provider: str = Field(default="", description="제공 기관/발행처")
    landing_page: str = Field(default="", description="원본 제공처 URL")


class IngestFromCatalogRequest(BaseModel):
    """① 통합 카탈로그 기반 등록 요청 (catalog_entry_id 또는 catalog_identifier 중 하나 필수)."""

    catalog_entry_id: int | None = Field(default=None, description="카탈로그 엔트리 숫자 PK (직접 조회)")
    catalog_identifier: str = Field(default="", description="DCAT 고유 식별자 (identifier)")
    table_name: str = Field(default="", description="테이블명 재정의 (기본: 카탈로그 title)")
    domain: str = Field(default="", description="도메인 재정의 (생략 시 특징 기반 자동 분류)")
    party: Literal["", "1st", "3rd"] = Field(
        default="3rd", description="데이터 파티 구분 — 카탈로그 개방 데이터는 기본 3rd"
    )
    records: list[dict] | None = Field(
        default=None,
        description="선택: 실데이터 레코드. 있으면 컬럼 프로파일링 포함, 없으면 메타데이터-only 등록",
    )


class RelationDecisionRequest(BaseModel):
    """① 승인 워크플로우의 승인/반려 결정 콜백 요청 (내부 API).

    승인 SSOT는 ①(dataset_relation.status)이며, ②는 결정을 그래프에 반영만 한다.
    """

    source_identifier: str = Field(description="출발 데이터셋의 카탈로그 식별자")
    target_identifier: str = Field(description="도착 데이터셋의 카탈로그 식별자")
    relation_type: str = Field(description="관계 유형")
    score: float = Field(ge=0.0, le=1.0, description="관계 점수")
    method: str = Field(default="llm", description="생성 방법")
    evidence: dict[str, Any] = Field(default_factory=dict, description="생성 근거")
    approved: bool = Field(description="승인 여부 (True: 그래프 반영, False: no-op)")


class DatasetSearchRequest(BaseModel):
    """테이블 KNN 검색 요청."""

    query: str = Field(description="검색 쿼리 텍스트")
    domain: str | None = Field(default=None, description="도메인 필터 (None이면 전체)")
    party: Literal["", "1st", "3rd"] = Field(default="", description="파티 필터 (빈 값이면 전체)")
    provider: str = Field(default="", description="제공 기관 필터 (빈 값이면 전체)")
    top_k: int = Field(default=5, ge=1, le=20, description="반환할 최대 결과 수")


class DatasetSearchResult(BaseModel):
    """테이블 KNN 검색 결과 단건."""

    table_name: str
    display_name: str = ""
    domain: str
    description: str
    table_summary: str
    score: float
    party: str = ""
    provider: str = ""
    landing_page: str = ""
    match_type: Literal["knn", "relation"] = Field(
        default="knn", description="매칭 근거 — knn(임베딩 유사 후보) / relation(기존 확정 관계)"
    )
    relation_type: str = Field(default="", description="match_type=relation 일 때 관계 유형")


# ── Tier 1 ──────────────────────────────────────────────────────────────────

class ColumnMatchInfo(BaseModel):
    """Tier 1/1.5 컬럼 매칭 결과."""

    column_a: str = Field(description="테이블 A의 컬럼명")
    column_b: str = Field(description="테이블 B의 컬럼명")
    match_type: Literal["exact_name", "fk", "code_dict", "ai_correlation"] = Field(
        description="매칭 방식 (ai_correlation: ① 컬럼 연관성 분석 서비스의 의미적 매칭)"
    )
    correlation: float | None = Field(default=None, description="AI 매칭 연관성 점수 (ai_correlation 전용)")


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
    """Tier 2 연관성 후보 (임베딩 + PPMI 결합)."""

    table_a: str
    table_b: str
    score: float = Field(ge=0.0, le=1.0, description="결합 연관성 점수 (임베딩 + PPMI 가중 결합)")
    method: Literal["embedding"] = "embedding"
    embedding_score: float | None = Field(default=None, description="결합 전 임베딩 코사인 유사도")
    ppmi_score: float | None = Field(
        default=None, description="KV 그래프 PPMI 연관성 수치 (KV 특징 없으면 None)"
    )
    shared_features: list[str] = Field(
        default_factory=list, description="두 데이터셋이 공유하는 상위 KV 특징 (PPMI 근거)"
    )


# ── Tier 3 ──────────────────────────────────────────────────────────────────

TableNature = Literal["record", "knowledge_content"]


class TableRelationVerification(BaseModel):
    """LLM 관계 검증 결과 (tool call 스키마).

    유형은 4종 하나만 쓴다. 근거는 별도 분류를 받지 않고 **인용**으로 받아
    코드가 실제 컬럼·샘플에 대조 검증한다 (자기신고 분류는 검증이 불가능했다).
    """

    related: bool = Field(description="두 테이블이 의미적으로 연관되어 있는가")
    table_a_nature: TableNature = Field(
        description="테이블 A의 성격 (record: 실세계 개체·사건의 기록 / knowledge_content: 지식·콘텐츠 자체가 행의 대상)"
    )
    table_b_nature: TableNature = Field(description="테이블 B의 성격")
    level1_type: Level1RelationType = Field(
        description=(
            "관계 유형 4종 — DETAIL_FOR(같은 개체에 앵커가 없는 속성을 공급) / "
            "STATISTIC_FOR(다른 입도의 집계·지수로 맥락 제공) / "
            "SAME_KIND(같은 표준·구조의 다른 지역·판본) / "
            "ADJACENT_TO(같은 맥락의 다른 개체)"
        )
    )
    evidence_columns: list[str] = Field(
        default_factory=list,
        description="근거로 삼은 컬럼명 — 양쪽 테이블에 모두 존재하는 이름만. 없으면 빈 배열",
    )
    evidence_values: list[str] = Field(
        default_factory=list,
        description="근거로 삼은 값 — 양쪽 샘플 레코드에 모두 등장하는 값만. 없으면 빈 배열",
    )
    relation_type: str = Field(description="관계를 한 줄로 설명 (예: 어린이집-교직원 1:N 참조)")
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
    display_name: str = Field(
        default="",
        description="테이블의 한국어 표시명 — 사람이 한눈에 알아볼 간결한 명사구 (예: patient_info → '환자 기본정보'). 테이블명이 이미 자연스러운 한국어면 그대로 사용",
    )
    domain: str = Field(
        default="",
        description="데이터 도메인 분류 (한국어 명사, 예: 보건의료, 교통및물류, 환경). 기존 도메인 목록이 주어지면 그중 가장 적합한 것을 우선 선택하고, 정말 맞는 것이 없을 때만 새 도메인을 제안",
    )


table_metadata_extraction_tool = pydantic_function_tool(
    TableMetadataExtraction,
    name="extract_table_metadata",
    description="테이블 메타데이터에서 설명과 키워드를 추출합니다.",
)


class DomainSelection(BaseModel):
    """도메인 선택 결과 — 기존 도메인 우선, 신설은 최후 수단."""

    domain: str = Field(
        description="제시된 기존 도메인 중 하나 (우선). 정말 해당 없을 때만 대분류 수준의 새 도메인 명사"
    )


domain_selection_tool = pydantic_function_tool(
    DomainSelection,
    name="select_domain",
    description="데이터셋 메타데이터를 도메인으로 분류합니다. 기존 도메인 재사용을 우선하고 신설은 지양합니다.",
)


class DatasetSearchIntent(BaseModel):
    """자연어 질의 → 데이터셋 검색 의도 (임베딩 검색용 구문)."""

    search_query: str = Field(description="데이터셋 요약문과 유사도 비교할 서술형 명사구 (한국어)")
    keywords: list[str] = Field(default_factory=list, description="핵심 주제 키워드 (보조 신호)")


class DatasetSearchAnswer(BaseModel):
    """검색 결과와 관계 구조를 근거로 한 자연어 답변."""

    answer: str = Field(
        description="질의에 대한 답변. 검색된 데이터셋과 그 관계만 근거로 삼는다 (한국어 3~5문장)"
    )
    suggested_next: list[str] = Field(
        default_factory=list,
        description="이어서 볼 만한 데이터셋명 — 반드시 제시된 목록 안에서만 고른다",
    )


dataset_search_answer_tool = pydantic_function_tool(
    DatasetSearchAnswer,
    name="answer_dataset_search",
    description="검색된 데이터셋과 관계 구조를 근거로 질의에 답합니다.",
)


dataset_search_intent_tool = pydantic_function_tool(
    DatasetSearchIntent,
    name="extract_dataset_search_intent",
    description="자연어 질의에서 데이터셋 검색 의도를 추출합니다.",
)


class Tier3Result(BaseModel):
    """Tier 3 LLM 검증 통과 결과."""

    table_a: str
    table_b: str
    relation_type: str
    level1_type: str = Field(default="", description="LLM이 판정한 4종 유형 (구프롬프트 응답이면 빈 값)")
    evidence_columns: list[str] = Field(default_factory=list, description="LLM이 인용한 근거 컬럼")
    evidence_values: list[str] = Field(default_factory=list, description="LLM이 인용한 근거 값")
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
    tier15_matches: list[ColumnMatchInfo] = Field(
        default_factory=list, description="Tier 1.5 — ① 컬럼 연관성 분석 서비스의 의미적 컬럼 매칭"
    )
    tier2_candidates: list[Tier2Candidate] = Field(default_factory=list)
    tier3_verified: list[Tier3Result] = Field(default_factory=list)
    saved_relations: list[DatasetRelation] = Field(default_factory=list, description="Neptune에 저장된 관계")
    pending_relations: list[DatasetRelation] = Field(
        default_factory=list, description="① 승인 대기로 제안된 저신뢰 관계 (그래프 미반영)"
    )


class RunRelationPipelineRequest(BaseModel):
    """관계 생성 파이프라인 실행 요청."""

    table_names: list[str] = Field(default_factory=list, description="대상 테이블명 목록 (비어있으면 전체)")
    tier2_threshold: float = Field(default=0.4, ge=0.0, le=1.0, description="Tier 2 유사도 임계값")
    tier3_confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Tier 3 확신도 임계값")
    refresh_table_names: list[str] = Field(
        default_factory=list,
        description="메타데이터가 변동된 테이블 — 이 테이블이 낀 쌍은 기존 관계가 있어도 재평가 (증분 갱신)",
    )
    seed_table_names: list[str] = Field(
        default_factory=list,
        description="기준(1st 파티) 테이블 — 지정 시 이 테이블이 낀 쌍만 평가 (기준 데이터셋 연계 탐색)",
    )
