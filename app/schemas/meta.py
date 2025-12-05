from typing import List
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, Text, DateTime
from enum import Enum


class MetadataInfo(BaseModel):
    """메타데이터 정보 데이터 모델"""
    tags: List[str] = Field(default_factory=list, title="태그", description="데이터와 관련된 태그 목록")
    topics: List[str] = Field(default_factory=list, title="주제", description="데이터의 주요 주제 목록")
    description: str = Field(default="", title="설명", description="데이터에 대한 간략한 설명")

    @field_validator('tags', 'topics')
    @classmethod
    def validate_non_empty_list(cls, v: List[str]) -> List[str]:
        return [item.strip() for item in v if item.strip()]


Base = declarative_base()


class DataInfo(Base):
    __tablename__ = 'data_info'
    id = Column(Integer, primary_key=True, index=True)
    origin_text = Column(Text, nullable=False)
    tags = Column(Text, nullable=False, default="[]")
    topics = Column(Text, nullable=False, default="[]")
    description = Column(Text, nullable=False, default="")
    extract_time = Column(DateTime, nullable=True)


class MetadataRequest(BaseModel):
    text: str = "데이터 값 입력 ex. 대법원 2025.09.18 선고 2023다221885 전원합의체 판결 ★ [손해배상(기)]〈중도상환수수료가 이자제한법 제4조 제1항에 따른 간주이자에 해당하여 이자제한법상 최고이자율 제한에 관한 규정이 적용되는지 문제된 사건〉 [공2025하,1874]"
    prompt_name: str = "generate_metadata"
    prompt_label: str = "production"


class SearchField(str, Enum):
    id = "id"
    origin_text = "origin_text"
    tags = "tags"
    topics = "topics"
    description = "description"


class SearchOption(str, Enum):
    all = "all"
    field = "field"
