from enum import Enum
from typing import List, Literal, Optional

from openai import pydantic_function_tool
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import Column, Integer, Text, DateTime
from sqlalchemy.orm import declarative_base


class MetadataInfo(BaseModel):
    """메타데이터 정보 데이터 모델"""
    tags: List[str] = Field(default_factory=list, title="태그", description="데이터와 관련된 태그 목록")
    topics: List[str] = Field(default_factory=list, title="주제", description="데이터의 주요 주제 목록")
    description: str = Field(default="", title="설명", description="데이터에 대한 간략한 설명")
    embedding: Optional[List[float]] = Field(default=None, title="임베딩", description="Description의 embedding 벡터 (384차원)")

    @field_validator('tags', 'topics')
    @classmethod
    def validate_non_empty_list(cls, v: List[str]) -> List[str]:
        return [item.strip() for item in v if item.strip()]


class CypherQueryInfo(BaseModel):
    """Cypher 쿼리 생성 데이터 모델"""
    cypher_query: str = Field(..., title="Cypher 쿼리", description="Neo4j 데이터베이스에 실행할 완전한 Cypher 쿼리")


TopicRelation = Literal["SIMILAR_TO", "SUBTYPE_OF", "SAME_AS", "BELONGS_TO"]


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
    description="주어진 데이터에서 메타데이터 정보를 추출합니다."
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
