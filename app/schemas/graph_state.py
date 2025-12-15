from typing import Any, Optional, List, Dict

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """Agent State - MessagesState를 확장하여 워크플로우 컨텍스트 관리"""
    origin_text: str
    prompt_name: str
    prompt_label: str

    extracted_metadata: Optional[Dict[str, Any]]
    generated_queries: Optional[List[str]]
    sqlite_result: Optional[Dict[str, Any]]

    metadata_extract_retry: int
    cypher_retry: int
    last_failed_step: Optional[str]
