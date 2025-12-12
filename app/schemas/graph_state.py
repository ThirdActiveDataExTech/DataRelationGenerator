from typing import Dict, Any, List

from pydantic import BaseModel, Field


class GraphWorkflowState(BaseModel):
    """LangGraph 워크플로우 상태"""

    # 입력 정보
    text: str = Field(description="원본 입력 텍스트")
    prompt_name: str = Field(default="default", description="프롬프트 이름")
    prompt_label: str = Field(default="default", description="프롬프트 라벨")

    # 처리 결과
    metadata: Dict[str, Any] = Field(default_factory=dict,
                                     description="LLM이 추출한 태그, 토픽, 설명 정보")
    cypher_queries: List[str] = Field(default_factory=list, description="Neo4j에 실행할 쿼리 리스트")

    # 실행 상태
    sqlite_status: str = Field(default="pending", description="SQLite 실행 결과 상태")
    execution_status: str = Field(default="pending", description="쿼리 실행 결과 상태 (success / fail)")

    # 에러 정보
    sqlite_error: str = Field(default="", description="SQLite 에러 내용")
    cypher_error: str = Field(default="", description="Cypher 실행 에러 내용")
    llm_error: str = Field(default="", description="LLM 처리 에러 내용")

    # 재시도 관리
    retry_count: int = Field(default=0, description="LLM 노드 재시도 횟수 카운터")
    fallback_retry: bool = Field(default=False, description="하드코딩 쿼리 시도 여부 플래그")
