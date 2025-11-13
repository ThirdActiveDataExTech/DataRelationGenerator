from typing import List
from pydantic import BaseModel, Field, field_validator


class MetadataInfo(BaseModel):
    """메타데이터 정보 데이터 모델"""
    tags: List[str] = Field(default_factory=list, title="태그", description="데이터와 관련된 태그 목록")
    topics: List[str] = Field(default_factory=list, title="주제", description="데이터의 주요 주제 목록")
    description: str = Field(default="", title="설명", description="데이터에 대한 간략한 설명")

    @field_validator('tags', 'topics')
    @classmethod
    def validate_non_empty_list(cls, v: List[str]) -> List[str]:
        return [item.strip() for item in v if item.strip()]
