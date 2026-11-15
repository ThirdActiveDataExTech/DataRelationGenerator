from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.keyvalue_graph import ValueType


class DataRecord(BaseModel):
    """데이터 레코드 (하나의 JSON 객체)."""

    source_file: str = Field(description="원본 파일명", examples=["data.json"])
    record_index: int = Field(description="파일 내 인덱스", examples=[0, 1, 2])
    raw_data: dict[str, Any] = Field(description="원본 JSON 데이터")
    key_values: dict[str, list[str]] = Field(
        default_factory=dict,
        description="추출된 Key-Value 쌍 (한글 필드명: 값 목록)"
    )
    original_key_mapping: dict[str, str] = Field(
        default_factory=dict,
        description="한글 필드명 → 원본 필드명 매핑 (예: {'병원명': 'BIZPLC_NM'})"
    )

    def get_document_id(self) -> str:
        """문서 ID 생성."""
        # 파일명에서 확장자 제거
        file_stem = Path(self.source_file).stem
        return f"{file_stem}_{self.record_index:04d}"

    def infer_value_type(self, key: str, value: str) -> ValueType:
        """값 타입 추론.

        Args:
            key: Key명
            value: 값

        Returns:
            ValueType
        """
        # 숫자형 필드
        numeric_keywords = ["수", "면적", "좌표", "위도", "경도"]
        if any(kw in key for kw in numeric_keywords):
            try:
                float(value)
                return ValueType.NUMBER
            except ValueError:
                pass

        # 날짜형 필드
        date_keywords = ["일자", "날짜"]
        if any(kw in key for kw in date_keywords):
            return ValueType.DATE

        return ValueType.TEXT

