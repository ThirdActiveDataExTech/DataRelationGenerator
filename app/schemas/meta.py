from typing import Dict, List, Literal, Union, Any

from openai import pydantic_function_tool
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, Text, DateTime
from sqlalchemy.orm import declarative_base


DocumentRelationType = Literal[
    "SIMILAR_TO",  # 유사한 내용/주제
    "CONTRADICTS",  # 상반된 견해
    "EXTENDS",  # 확장/발전
    "PREREQUISITE_OF",  # 선행 지식 필요
    "IMPLEMENTS",  # 이론→실제 구현
    "CO_OCCURS"  # 그래프 구조상 자주 함께 나타남
]


class DocumentRelationInfo(BaseModel):
    """Document 간 관계 정보"""
    target_doc_id: str = Field(..., description="관계 대상 문서 ID")
    relation_type: DocumentRelationType = Field(..., description="관계 타입")
    reason: str = Field(..., description="관계가 성립하는 이유")
    confidence: float = Field(..., ge=0.0, le=1.0, description="관계 확신도 (0.0 ~ 1.0)")


class DocumentRelationResult(BaseModel):
    """Document 관계 추론 결과"""
    relations: List[DocumentRelationInfo] = Field(
        default_factory=list,
        description="추출된 Document 간 관계 목록. 관계가 없으면 빈 배열 반환"
    )


Base = declarative_base()


class DataInfo(Base):
    __tablename__ = 'data_info'
    id = Column(Integer, primary_key=True, index=True)
    origin_text = Column(Text, nullable=False)
    tags = Column(Text, nullable=False, default="[]")
    description = Column(Text, nullable=False, default="")
    extract_time = Column(DateTime, nullable=True)


# Pydantic 모델 기반 OpenAI Tool 정의
document_relation_tool = pydantic_function_tool(
    DocumentRelationResult,
    name="infer_document_relations",
    description="새 문서와 기존 유사 문서들 간의 의미적 관계를 추론합니다."
)


class DocumentRelevance(BaseModel):
    """문서 적합성 평가 결과"""
    is_relevant: bool = Field(..., description="문서가 쿼리 의도에 적합한지 여부 (true/false)")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="적합도 점수 (0.0~1.0)")
    reason: str = Field(..., description="판단 근거 (구체적으로 작성)")


document_relevance_tool = pydantic_function_tool(
    DocumentRelevance,
    name="verify_document_relevance",
    description="검색된 문서가 사용자 쿼리의 의도를 만족하는지 평가합니다."
)


class KeyValueExtractionResult(BaseModel):
    """Key-Value 추출 결과"""
    key_values: dict[str, List[str]] = Field(
        ...,
        description="추출된 Key-Value 쌍. Key는 원본 필드명(예: 'BIZPLC_NM'), Value는 문자열 배열"
    )


key_value_extraction_tool = pydantic_function_tool(
    KeyValueExtractionResult,
    name="extract_key_value_pairs",
    description="데이터에서 모든 필드명(Key)과 그 값(Value)을 추출합니다."
)


class ClusterNormalizationResult(BaseModel):
    """클러스터 정규화 결과"""
    normalized_value: str = Field(
        ...,
        description="클러스터를 대표하는 정규화된 값. 원본 값 중 하나를 선택하거나 적절한 메타 개념을 생성."
    )


cluster_normalization_tool = pydantic_function_tool(
    ClusterNormalizationResult,
    name="normalize_cluster",
    description="유사한 값들의 클러스터를 분석하여 적절한 정규화 값을 생성합니다."
)


class QueryKeyInferenceResult(BaseModel):
    """쿼리에 필요한 Key 추론 결과"""
    keys: List[str] = Field(
        default_factory=list,
        description="검색에 필요한 Key 이름 목록 (예: ['병원명', '지역'])"
    )
    search_terms: Dict[str, Union[str, Dict[str, Any]]] = Field(
        default_factory=dict,
        description=(
            "각 Key에서 검색할 값. "
            "텍스트 검색: {'병원명': '365매일한방병원'}, "
            "숫자 조건: {'병상수': {'operator': '>=', 'value': 100}}"
        )
    )


query_key_inference_tool = pydantic_function_tool(
    QueryKeyInferenceResult,
    name="infer_required_keys",
    description="사용자 질문을 분석하여 필요한 그래프 Key들을 추론합니다."
)
