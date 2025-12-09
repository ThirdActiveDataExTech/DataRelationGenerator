from typing import Dict, Any, List

from pydantic import BaseModel, Field


class GraphWorkflowState(BaseModel):
    """LangGraph 워크플로우 상태"""

    # 입력 텍스트
    text: str = Field(description="원본 입력 텍스트")
    prompt_name: str = Field(default="default", description="프롬프트 이름")
    prompt_label: str = Field(default="default", description="프롬프트 라벨")

    # 추출된 메타데이터
    metadata: Dict[str, Any] = Field(default_factory=dict,
                                     description="LLM이 추출한 태그, 토픽, 설명 정보")

    # 상태 관리
    sqlite_status: str = Field(default="pending", description="SQLite 실행 결과 상태")
    sqlite_error: str = Field(default="", description="SQLite 에러 내용")
    cypher_queries: List[str] = Field(default_factory=list, description="Neo4j에 실행할 쿼리 리스트")
    execution_status: str = Field(default="pending", description="쿼리 실행 결과 상태 (success / fail)")
    cypher_error: str = Field(default="", description="Cypher 실행 에러 내용")
    llm_error: str = Field(default="", description="LLM 처리 에러 내용")
    retry_count: int = Field(default=0, description="LLM 노드 재시도 횟수 카운터")
    fallback_retry: bool = Field(default=False, description="하드코딩 쿼리 시도 여부 플래그")
