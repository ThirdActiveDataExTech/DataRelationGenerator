from typing import Dict, List, Literal

from openai import pydantic_function_tool
from pydantic import BaseModel, Field


class FieldAnalysis(BaseModel):
    """필드 분석 결과"""
    korean_name: str = Field(..., description="필드의 한글 이름")
    description: str = Field(..., description="필드의 의미 설명")
    data_type: Literal["text", "number", "date", "phone", "address", "email", "url"] = Field(
        ..., description="필드의 데이터 타입"
    )


class SchemaAnalysisResult(BaseModel):
    """스키마 분석 결과"""
    domain: str = Field(..., description="데이터 도메인 (예: 의료, 법률, 부동산, 음식점)")
    centrality_name: str = Field(..., description="중심 노드 이름 (예: 병원현황, 판례정보)")
    fields: Dict[str, FieldAnalysis] = Field(
        ..., description="필드명을 키로 하는 필드 분석 결과 딕셔너리"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="스키마 추론의 신뢰도 (0.0~1.0)"
    )
    notes: str = Field(
        default="",
        description="추가 메모나 주의사항"
    )


schema_analysis_tool = pydantic_function_tool(
    SchemaAnalysisResult,
    name="analyze_schema",
    description="데이터 샘플을 분석하여 스키마를 추론합니다."
)


class FieldStatistics(BaseModel):
    """필드 통계 정보"""
    total_count: int = Field(..., description="전체 레코드 수")
    unique_count: int = Field(..., description="고유값 개수")
    null_count: int = Field(..., description="null/빈값 개수")
    null_ratio: float = Field(..., description="null 비율")
    samples: List[str] = Field(..., description="샘플 값들 (최대 5개)")
    value_lengths: Dict[str, int] = Field(
        default_factory=dict,
        description="값 길이 분포 (min, max, avg)"
    )
