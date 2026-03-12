from typing import List, Literal, Optional

from openai import pydantic_function_tool
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import Column, Integer, Text, DateTime
from sqlalchemy.orm import declarative_base


class MetadataInfo(BaseModel):
    """메타데이터 정보 데이터 모델"""
    entities: List[str] = Field(
        default_factory=list,
        title="엔티티",
        description="문서에서 추출된 핵심 엔티티 목록 (인명, 지명, 기관명, 개념 등). 반드시 배열 형식으로 반환: [\"엔티티1\", \"엔티티2\"]"
    )
    topics: List[str] = Field(
        default_factory=list,
        title="주제",
        description="데이터의 주요 주제 목록. 반드시 배열 형식으로 반환: [\"주제1\", \"주제2\", \"주제3\"]"
    )
    description: str = Field(default="", title="설명", description="데이터에 대한 간략한 설명")
    embedding: Optional[List[float]] = Field(default=None, title="임베딩", description="Description의 embedding 벡터 (384차원)")

    @field_validator('entities', 'topics')
    @classmethod
    def validate_non_empty_list(cls, v: List[str]) -> List[str]:
        return [item.strip() for item in v if item.strip()]


class CypherQueryInfo(BaseModel):
    """Cypher 쿼리 생성 데이터 모델"""
    cypher_query: str = Field(..., title="Cypher 쿼리", description="Neo4j 데이터베이스에 실행할 완전한 Cypher 쿼리")


TopicRelation = Literal["SIMILAR_TO", "SUBTYPE_OF", "SAME_AS", "BELONGS_TO"]

DocumentRelationType = Literal[
    "SIMILAR_TO",  # 유사한 내용/주제
    "CONTRADICTS",  # 상반된 견해
    "EXTENDS",  # 확장/발전
    "PREREQUISITE_OF",  # 선행 지식 필요
    "IMPLEMENTS",  # 이론→실제 구현
    "CO_OCCURS"  # 그래프 구조상 자주 함께 나타남
]


class TopicCategoryMapping(BaseModel):
    """Topic과 Category 간 매핑 정보"""
    topic: str = Field(..., description="토픽명")
    relation: TopicRelation = Field(..., description="관계 타입")


class CategoryInfo(BaseModel):
    """카테고리 정보 데이터 모델"""
    name: str = Field(..., description="카테고리명 (일반명사)")
    topics: List[TopicCategoryMapping] = Field(default_factory=list, description="포함되는 토픽 목록")


class CategoryGenerationResult(BaseModel):
    """카테고리 생성 결과"""
    categories: List[CategoryInfo] = Field(default_factory=list, description="생성된 카테고리 목록")


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


class QueryCondition(BaseModel):
    """쿼리 조건"""
    type: Literal["entity", "property"] = Field(..., description="조건 타입: entity (Entity 노드와 연결) 또는 property (Document 속성)")
    value: str = Field(..., description="조건 값 (예: '가평', '재활의학과', '종합병원')")
    semantic: str = Field(..., description="의미적 역할 (예: '위치', '진료과목', '병원유형')")


class QueryGraphPattern(BaseModel):
    """쿼리 그래프 패턴"""
    target: str = Field(default="Document", description="검색 대상 노드 타입 (보통 'Document')")
    conditions: List[QueryCondition] = Field(
        default_factory=list,
        description="검색 조건 목록. 조건이 없으면 빈 배열 반환"
    )


Base = declarative_base()


class DataInfo(Base):
    __tablename__ = 'data_info'
    id = Column(Integer, primary_key=True, index=True)
    origin_text = Column(Text, nullable=False)
    tags = Column(Text, nullable=False, default="[]")
    topics = Column(Text, nullable=False, default="[]")
    description = Column(Text, nullable=False, default="")
    extract_time = Column(DateTime, nullable=True)


# Pydantic 모델 기반 OpenAI Tool 정의
metadata_tool = pydantic_function_tool(
    MetadataInfo,
    name="extract_metadata_info",
    description="주어진 데이터에서 메타데이터 정보(엔티티, 주제, 설명)를 추출합니다."
)

cypher_tool = pydantic_function_tool(
    CypherQueryInfo,
    name="generate_cypher_query",
    description="Neo4j Cypher 쿼리 생성 전용 도구입니다. 메타데이터와 텍스트를 Neo4j 그래프 데이터베이스 쿼리로 변환할 때만 사용합니다."
)

category_generation_tool = pydantic_function_tool(
    CategoryGenerationResult,
    name="generate_categories",
    description="Topic 목록을 분석하여 의미적으로 묶일 수 있는 상위 Category를 생성합니다."
)

document_relation_tool = pydantic_function_tool(
    DocumentRelationResult,
    name="infer_document_relations",
    description="새 문서와 기존 유사 문서들 간의 의미적 관계를 추론합니다."
)

query_graph_tool = pydantic_function_tool(
    QueryGraphPattern,
    name="parse_query_to_graph",
    description="검색 쿼리를 그래프 패턴으로 변환합니다. 쿼리에서 검색 조건들을 추출하여 그래프 구조로 표현합니다."
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
