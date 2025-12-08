from typing import Dict, Any

from pydantic import BaseModel, Field


class GraphWorkflowState(BaseModel):
    """LangGraph 워크플로우 상태"""

    # 입력 부분은 다른 스키마로 분리할 것
    input_text: str = Field(description="입력 텍스트")
    prompt_name: str = Field(default="default", description="프롬프트 이름")
    prompt_label: str = Field(default="default", description="프롬프트 라벨")

    # 추출된 메타데이터
    metadata: Dict[str, Any] = Field(default_factory=dict,
                                     description="추출된 메타데이터 {topics: [...], tags: [...], description: '...'}")

    # SQLite 저장 결과
    sqlite_record_id: int = Field(default=0, description="SQLite 레코드 ID")

    # 상태
    status: str = Field(default="started", description="워크플로우 상태")
