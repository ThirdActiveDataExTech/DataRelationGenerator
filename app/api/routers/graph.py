from typing import Annotated

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse, RedirectResponse

from app.schemas.graphrag_api import (
    GraphRAGSearchRequest,
    GraphRAGSearchResponse,
    NodeExploreRequest,
    NodeExploreResponse
)
from app.schemas.response import APIResponseModel
from app.src.graphrag.graph_service import graph_service

router = APIRouter(prefix="/graph", tags=["graph"])


@router.post("/embeddings", tags=["graph"], response_model=APIResponseModel,
             response_class=JSONResponse)
async def update_graph_embeddings(
        embedding_dim: Annotated[int, Body(embed=True)] = 128,
        walk_length: Annotated[int, Body(embed=True)] = 80,
        walks_per_node: Annotated[int, Body(embed=True)] = 10
):
    """전체 Document 노드에 대해 그래프 임베딩 일괄 생성 (Step 2)

    Args:
        embedding_dim: 임베딩 차원 (기본: 128)
        walk_length: Random walk 길이 (기본: 80)
        walks_per_node: 노드당 walk 횟수 (기본: 10)

    Flow:
        Step 1: /metadata/extract - 문서 추가 (text_embedding)
        Step 2: /graph/embeddings - 그래프 임베딩 생성 ← 현재
        Step 3: /graph/relations/analyze - 관계 분석

    Note:
        - Node2Vec 알고리즘으로 그래프 구조 학습
        - 전체 그래프에 대해 1회 실행
        - 처리 시간이 오래 걸릴 수 있음 (문서 수에 비례)
    """
    result = await graph_service.update_all_graph_embeddings(
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        walks_per_node=walks_per_node
    )
    return APIResponseModel(
        result=result,
        description=f"그래프 임베딩 생성 완료: {result.get('success', 0)}개 성공"
    )


@router.post("/relations/analyze", tags=["graph"], response_model=APIResponseModel,
             response_class=JSONResponse)
async def analyze_document_relations(
        doc_id: Annotated[str | None, Body(embed=True)] = None,
        top_k: Annotated[int, Body(embed=True)] = 10,
        text_weight: Annotated[float, Body(embed=True)] = 0.6,
        graph_weight: Annotated[float, Body(embed=True)] = 0.4
):
    """Document 간 관계 분석 및 생성 (Step 3)

    Args:
        doc_id: 분석할 Document ID (None이면 전체 문서 분석)
        top_k: 관련 문서 최대 개수 (기본: 10)
        text_weight: 텍스트 유사도 가중치 (기본: 0.6)
        graph_weight: 그래프 임베딩 유사도 가중치 (기본: 0.4)

    Flow:
        Step 1: /metadata/extract - 문서 추가 (text_embedding)
        Step 2: /graph/embeddings - 그래프 임베딩 생성
        Step 3: /graph/relations/analyze - 관계 분석 ← 현재

    Note:
        - 하이브리드 검색: 텍스트 + 그래프 임베딩 + 그래프 구조
        - LLM으로 의미적 관계 추론 (EXTENDS, USES, PREREQUISITE_OF 등)
        - doc_id=null: 모든 문서에 대해 관계 분석
        - 처리 시간이 오래 걸릴 수 있음 (문서 수 × top_k × LLM 호출)
    """
    # 빈 문자열을 None으로 처리
    if doc_id == "":
        doc_id = None

    result = await graph_service.analyze_document_relations(
        doc_id=doc_id,
        top_k=top_k,
        text_weight=text_weight,
        graph_weight=graph_weight
    )
    return APIResponseModel(
        result=result,
        description=f"관계 분석 완료: {result.get('total_relations_created', 0)}개 관계 생성"
    )


@router.post("/search", tags=["graph"], response_model=GraphRAGSearchResponse,
             response_class=JSONResponse)
async def search_graph_rag(request: GraphRAGSearchRequest):
    """사용자 질의 기반 Graph RAG 검색

    Args:
        request: GraphRAGSearchRequest
            - query: 사용자 질의
            - top_k: 반환할 Document 개수 (기본: 5, 최대: 20)

    Returns:
        GraphRAGSearchResponse
            - query: 사용자 질의
            - documents: 검색된 Document 목록 (벡터 유사도 순)
            - common_metadata: 검색된 Document들의 공통 Category/Tag 집계
                * categories: Category별 출현 빈도 및 비율
                * common_tags: Tag별 출현 빈도 및 비율

    Note:
        - 벡터 검색으로 관련 Document 찾기
        - 공통 메타데이터(Category, Tag)를 집계하여 검색 결과 요약 제공
        - 답변 생성(Generation)은 포함하지 않음

    Example:
        Request:
        {
            "query": "암환자 케어 가능한 병원",
            "top_k": 5
        }

        Response:
        {
            "query": "암환자 케어 가능한 병원",
            "documents": [
                {
                    "doc_id": "hospital_001",
                    "text": "...",
                    "description": "세브란스 협력센터",
                    "score": 0.95
                }
            ],
            "common_metadata": {
                "categories": [
                    {"name": "중소형병원", "doc_count": 5, "coverage": 1.0}
                ],
                "common_tags": [
                    {"name": "암환자케어", "doc_count": 5, "coverage": 1.0},
                    {"name": "병상가용", "doc_count": 4, "coverage": 0.8}
                ]
            }
        }
    """
    result = graph_service.graphrag_service.search_graph(
        query=request.query,
        top_k=request.top_k
    )

    return GraphRAGSearchResponse(**result)


@router.post("/explore", tags=["graph"], response_model=NodeExploreResponse,
             response_class=JSONResponse)
async def explore_node(request: NodeExploreRequest):
    """노드 중심 1-hop 그래프 탐색

    Args:
        request: NodeExploreRequest
            - node_id: 탐색할 노드 ID

    Returns:
        NodeExploreResponse
            - center_node: 중심 노드 정보
            - connected_documents: 연결된 Document 목록
            - graph: 중심 노드 기준 1-hop 그래프 구조

    Note:
        - 클릭한 노드를 중심으로 depth 1 연결된 모든 노드/관계 반환
        - 연결된 Document들도 함께 반환하여 탐색 가능

    Example:
        Request:
        {
            "node_id": "hospital_001"
        }

        Response:
        {
            "center_node": {
                "id": "hospital_001",
                "labels": ["Document"],
                "properties": {"description": "..."}
            },
            "connected_documents": [
                {"doc_id": "related_001", "text": "...", "score": 0}
            ],
            "graph": {
                "nodes": [...],
                "relationships": [...]
            }
        }
    """
    result = graph_service.graphrag_service.explore_node(
        node_id=request.node_id
    )

    return NodeExploreResponse(**result)


@router.get("/viewer", tags=["graph"], include_in_schema=False)
async def graph_viewer():
    """Graph RAG 시각화 페이지로 리다이렉트

    Note:
        /static/graph-viewer.html로 리다이렉트
    """
    return RedirectResponse(url="/static/graph-viewer.html")
