"""Key-Value 기반 그래프 모델 정의."""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ValueType(str, Enum):
    """값 타입."""

    TEXT = "text"
    NUMBER = "number"
    DATE = "date"
    BOOLEAN = "boolean"


class NodeLabel(str, Enum):
    """노드 레이블."""

    DOCUMENT = "Document"
    KEY = "Key"
    VALUE = "Value"
    NORMALIZED_VALUE = "NormalizedValue"


class RelationType(str, Enum):
    """관계 타입."""

    HAS_KEY = "HAS_KEY"
    HAS_VALUE = "HAS_VALUE"
    EXTRACTED_FROM = "EXTRACTED_FROM"
    SAME_AS = "SAME_AS"


# ===== 노드 모델 =====


class DocumentNode(BaseModel):
    """문서 노드 모델."""

    id: str = Field(..., description="문서 고유 ID")
    source_file: str = Field(..., description="원본 파일명")
    record_index: int = Field(..., description="파일 내 레코드 인덱스")
    raw_text: str = Field(..., description="원본 JSON 텍스트")


class KeyNode(BaseModel):
    """키 노드 모델."""

    name: str = Field(..., description="한글 필드명 (예: 병원명, 병상수)")
    original_field: str = Field(..., description="원본 필드명 (예: BIZPLC_NM, SICKBD_CNT)")
    description: Optional[str] = Field(None, description="필드 설명")


class ValueNode(BaseModel):
    """값 노드 모델."""

    value: str = Field(..., description="값 (문자열로 통일)")
    value_type: ValueType = Field(default=ValueType.TEXT, description="값 타입")
    original_value: Optional[str] = Field(None, description="원본 값 (타입 변환 전)")


class NormalizedValueNode(BaseModel):
    """정규화 값 노드 모델."""

    normalized_value: str = Field(..., description="정규화된 대표 값")
    key_name: str = Field(..., description="해당 Key명 (예: 병원명, 영업상태)")
    count: int = Field(default=0, description="연결된 Value 수")


# ===== 관계 모델 =====


class HasKeyRelation(BaseModel):
    """HAS_KEY 관계."""

    from_node: str = Field(..., description="Document ID")
    to_node: str = Field(..., description="Key name")


class HasValueRelation(BaseModel):
    """HAS_VALUE 관계."""

    from_node: str = Field(..., description="Key name")
    to_node: str = Field(..., description="Value")
    key_name: str = Field(..., description="Key명 (검색용)")


class ExtractedFromRelation(BaseModel):
    """EXTRACTED_FROM 관계."""

    from_node: str = Field(..., description="Value")
    to_node: str = Field(..., description="Document ID")


class SameAsRelation(BaseModel):
    """SAME_AS 관계."""

    from_node: str = Field(..., description="Value")
    to_node: str = Field(..., description="NormalizedValue")
    key_name: str = Field(..., description="Key명")


# ===== 그래프 구축 결과 =====


class GraphBuildResult(BaseModel):
    """그래프 구축 결과."""

    document_id: str = Field(..., description="생성된 문서 ID")
    keys_count: int = Field(default=0, description="생성된 Key 수")
    values_count: int = Field(default=0, description="생성된 Value 수")
    normalized_count: int = Field(default=0, description="정규화된 Value 수")
    success: bool = Field(default=True, description="성공 여부")
    message: Optional[str] = Field(None, description="메시지")


class RdfGraphBuildResult(BaseModel):
    """RDF 그래프 구축 결과."""

    document_id: str = Field(..., description="생성된 문서 ID")
    keys_count: int = Field(default=0, description="생성된 Key 수")
    values_count: int = Field(default=0, description="생성된 Value 수")
    json_ld: Optional[Any] = Field(None, description="JSON-LD 딕셔너리")
    success: bool = Field(default=True, description="성공 여부")
    message: Optional[str] = Field(None, description="메시지")


# ===== 검색 모델 =====


class KeyValueFilter(BaseModel):
    """Key-Value 기반 필터."""

    key_name: str = Field(..., description="Key명 (예: 병상수, 영업상태)")
    operator: Literal["=", ">", "<", ">=", "<=", "contains", "in"] = Field(
        default="=", description="비교 연산자"
    )
    value: Any = Field(..., description="비교 값")


class GraphSearchQuery(BaseModel):
    """그래프 검색 쿼리."""

    filters: List[KeyValueFilter] = Field(default_factory=list, description="필터 목록")
    use_normalized: bool = Field(
        default=True, description="정규화된 값 기반 검색 여부"
    )
    limit: int = Field(default=10, description="결과 제한")


class SearchResult(BaseModel):
    """검색 결과."""

    document_id: str = Field(..., description="문서 ID")
    matched_values: Dict[str, str] = Field(
        default_factory=dict, description="매칭된 Key-Value"
    )
    score: float = Field(default=0.0, description="매칭 점수")
    connection_path: Optional[str] = Field(None, description="연결 경로 (설명용)")
