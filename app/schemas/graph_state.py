from typing import Any, Optional, List, Dict

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """Agent State - MessagesState를 확장하여 워크플로우 컨텍스트 관리"""
    origin_text: str

    # 중간 결과 저장
    extracted_metadata: Optional[Dict[str, Any]]
    generated_queries: Optional[List[str]]
    neo4j_result: Optional[Dict[str, Any]]

    # Embedding 관련 state
    generated_embedding: Optional[List[float]]
    similarity_relationships: Optional[List[Dict[str, Any]]]
    document_id: Optional[str]

    # Retry 관리
    metadata_extract_retry: int
    cypher_retry: int
    fallback_used: bool
    last_failed_step: Optional[str]
