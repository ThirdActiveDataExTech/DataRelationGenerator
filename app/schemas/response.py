from typing import Any, List
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
import json

from app.config import settings
from app.log import Log
from app.version import VERSION


class APIResponseModel(BaseModel):
    """기본 API 응답 포맷 by AIP Restful API 디자인 가이드

    AI플랫폼팀 API 정식 포맷으로 그대로 사용 권장
    """
    code: int = Field(default=int(f"{settings.SERVICE_CODE}200"))  # 6자리 숫자 권장
    message: str = Field(
        default=f"API Response Success ({VERSION})" if Log.is_debug_enable() else "API Response Success")
    result: Any = Field(default={})  # API response result
    description: str = Field(default="응답과 관련된 자세한 설명 작성")

    def to_dict(self):
        return {
            "code": self.code,
            "message": self.message,
            "result": self.result,
            "description": self.description
        }


class DataInfoResponse(BaseModel):
    """DB 조회용 DataInfo 응답 스키마"""
    model_config = ConfigDict(from_attributes=True)

    id: int = Field(gt=0, description="데이터 ID")
    origin_text: str = Field(min_length=1, description="원본 텍스트")
    topics: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    description: str
    extract_time: datetime

    @classmethod
    def from_orm(cls, data_info) -> "DataInfoResponse":
        """DataInfo ORM 객체를 DataInfoResponse로 변환"""
        try:
            topics = json.loads(data_info.topics) if data_info.topics else []
            tags = json.loads(data_info.tags) if data_info.tags else []
        except json.JSONDecodeError:
            topics = []
            tags = []

        return cls(
            id=data_info.id,
            origin_text=data_info.origin_text,
            topics=topics,
            tags=tags,
            description=data_info.description,
            extract_time=data_info.extract_time
        )
