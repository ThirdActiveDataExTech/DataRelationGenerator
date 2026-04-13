from typing import List, Dict, Any

from pydantic import BaseModel, Field


class GraphRAGSearchRequest(BaseModel):
    """Graph RAG 검색 요청"""
    query: str = Field(..., description="사용자 질의", min_length=1)
    top_k: int = Field(5, description="반환할 Document 개수", ge=1, le=20)


class RelatedDocument(BaseModel):
    """함께 보면 좋은 연관 문서"""
    target_doc_id: str = Field(..., description="연관 문서 ID")
    target_description: str = Field("", description="연관 문서 설명")
    relation_type: str = Field(..., description="관계 타입 (SIMILAR_TO, EXTENDS 등)")
    reason: str = Field("", description="왜 함께 보면 좋은지 설명")
    confidence: float = Field(0.0, description="관계 확신도", ge=0.0, le=1.0)


class DocumentResult(BaseModel):
    """검색된 Document 정보"""
    doc_id: str = Field(..., description="Document ID")
    text: str = Field("", description="Document 본문")
    description: str = Field("", description="Document 설명")
    score: float = Field(0.0, description="유사도 점수")
    related_documents: List[RelatedDocument] = Field(
        default_factory=list,
        description="함께 보면 좋은 연관 문서 목록"
    )


class GraphNode(BaseModel):
    """그래프 노드"""
    id: str = Field(..., description="노드 ID")
    labels: List[str] = Field(default_factory=list, description="노드 레이블 (예: Document, Key, Value)")
    properties: Dict[str, Any] = Field(default_factory=dict, description="노드 속성")


class GraphRelationship(BaseModel):
    """그래프 관계"""
    type: str = Field(..., description="관계 타입 (예: HAS_KEY, HAS_VALUE, EXTENDS)")
    start: str = Field(..., description="시작 노드 ID")
    end: str = Field(..., description="종료 노드 ID")
    properties: Dict[str, Any] = Field(default_factory=dict, description="관계 속성")


class GraphStructure(BaseModel):
    """그래프 구조"""
    nodes: List[GraphNode] = Field(default_factory=list, description="노드 목록")
    relationships: List[GraphRelationship] = Field(default_factory=list, description="관계 목록")


class CommonMetadata(BaseModel):
    """검색 결과의 공통 메타데이터"""
    common_keys: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="공통 Key (name, doc_count, coverage)"
    )
    common_values: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="공통 Value (name, doc_count, coverage)"
    )


class GraphRAGSearchResponse(BaseModel):
    """Graph RAG 검색 응답"""
    query: str = Field(..., description="사용자 질의")
    documents: List[DocumentResult] = Field(default_factory=list, description="검색된 Document 목록")
    common_metadata: CommonMetadata = Field(
        default_factory=CommonMetadata,
        description="검색된 Document들의 공통 Key/Value 집계"
    )
    graph: GraphStructure = Field(
        default_factory=GraphStructure,
        description="간단한 그래프 구조 (Document + 공통 Key/Value만)"
    )
    error: str | None = None

    @classmethod
    def empty(cls, query: str, error: str | None = None) -> "GraphRAGSearchResponse":
        """빈 응답 생성"""
        return cls(query=query, error=error)


class NodeExploreRequest(BaseModel):
    """노드 탐색 요청"""
    node_id: str = Field(..., description="탐색할 노드 ID")


class NodeExploreResponse(BaseModel):
    """노드 탐색 응답"""
    center_node: GraphNode | None = None
    connected_documents: List[DocumentResult] = Field(
        default_factory=list,
        description="연결된 Document 목록"
    )
    graph: GraphStructure = Field(
        default_factory=GraphStructure,
        description="중심 노드 기준 1-hop 그래프"
    )
    error: str | None = None

    @classmethod
    def empty(cls, center_node: GraphNode | None = None, error: str | None = None) -> "NodeExploreResponse":
        """빈 응답 생성"""
        return cls(center_node=center_node, error=error)
