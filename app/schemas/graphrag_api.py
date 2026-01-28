from typing import List, Dict, Any

from pydantic import BaseModel, Field


class GraphRAGSearchRequest(BaseModel):
    """Graph RAG 검색 요청"""
    query: str = Field(..., description="사용자 질의", min_length=1)
    top_k: int = Field(5, description="반환할 Document 개수", ge=1, le=20)
    depth: int = Field(1, description="그래프 탐색 깊이 (1=1-hop, 2=2-hop)", ge=1, le=2)


class DocumentResult(BaseModel):
    """검색된 Document 정보"""
    doc_id: str = Field(..., description="Document ID")
    text: str = Field("", description="Document 본문")
    description: str = Field("", description="Document 설명")
    score: float = Field(0.0, description="유사도 점수")


class GraphNode(BaseModel):
    """그래프 노드"""
    id: str = Field(..., description="노드 ID")
    labels: List[str] = Field(default_factory=list, description="노드 레이블 (예: Document, Topic, Tag)")
    properties: Dict[str, Any] = Field(default_factory=dict, description="노드 속성")


class GraphRelationship(BaseModel):
    """그래프 관계"""
    type: str = Field(..., description="관계 타입 (예: HAS_TOPIC, EXTENDS)")
    start: str = Field(..., description="시작 노드 ID")
    end: str = Field(..., description="종료 노드 ID")
    properties: Dict[str, Any] = Field(default_factory=dict, description="관계 속성")


class GraphStructure(BaseModel):
    """그래프 구조"""
    nodes: List[GraphNode] = Field(default_factory=list, description="노드 목록")
    relationships: List[GraphRelationship] = Field(default_factory=list, description="관계 목록")


class GraphRAGSearchResponse(BaseModel):
    """Graph RAG 검색 응답"""
    query: str = Field(..., description="사용자 질의")
    documents: List[DocumentResult] = Field(default_factory=list, description="검색된 Document 목록")
    graph: GraphStructure = Field(..., description="그래프 구조")
    error: str | None = Field(None, description="에러 메시지 (있을 경우)")
