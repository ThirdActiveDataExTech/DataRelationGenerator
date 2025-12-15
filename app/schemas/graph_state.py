from typing import Any, Optional, List, Dict

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """Agent State - MessagesState를 확장하여 워크플로우 컨텍스트 관리"""
    origin_text: str
    prompt_name: str = "generate_metadata"
    prompt_label: str = "production"

    # 중간 결과 저장
    extracted_metadata: Optional[Dict[str, Any]] = None
    generated_queries: Optional[List[str]] = None
    sqlite_result: Optional[Dict[str, Any]] = None

    # Retry 관리
    metadata_extract_retry: int = 0
    cypher_retry: int = 0
    last_failed_step: Optional[str] = None
