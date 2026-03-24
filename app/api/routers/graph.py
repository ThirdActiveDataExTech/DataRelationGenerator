from typing import Annotated

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse, RedirectResponse

from app.schemas.graphrag_api import (
    GraphRAGSearchRequest,
    GraphRAGSearchResponse
)
from app.schemas.response import APIResponseModel
from app.src.graphrag.graph_service import graph_service

router = APIRouter(prefix="/graph", tags=["graph"])


@router.post("/embeddings", tags=["graph"], response_model=APIResponseModel,
             response_class=JSONResponse)
async def update_graph_embeddings(
        embedding_dim: Annotated[int, Body(embed=True)] = 128
):
    """전체 Document 노드에 대해 그래프 임베딩 일괄 생성 (Step 2)

    Args:
        embedding_dim: 임베딩 차원 (기본: 128)

    Flow:
        Step 1: /metadata/extract - 문서 추가 (text_embedding + binary_text_embedding)
        Step 2: /graph/embeddings - 그래프 임베딩 생성 ← 현재
        Step 3: /graph/relations/analyze - 관계 분석

    Note:
        - HashGNN 알고리즘으로 그래프 구조 학습
        - 관계 타입별로 다른 해시 함수 사용 (heterogeneous=True)
        - 전체 그래프에 대해 1회 실행
        - 처리 시간이 오래 걸릴 수 있음 (문서 수에 비례)
    """
    result = await graph_service.update_all_graph_embeddings(
        embedding_dim=embedding_dim
    )
    return APIResponseModel(
        result=result,
        description=f"[HashGNN] 그래프 임베딩 생성 완료: {result.get('success', 0)}개 성공"
    )


@router.post("/relations/analyze", tags=["graph"], response_model=APIResponseModel,
             response_class=JSONResponse)
async def analyze_document_relations(
        doc_id: Annotated[str | None, Body(embed=True)] = None,
        top_k: Annotated[int, Body(embed=True)] = 4,
        text_weight: Annotated[float, Body(embed=True)] = 0.4,
        graph_weight: Annotated[float, Body(embed=True)] = 0.3,
        normalized_value_weight: Annotated[float, Body(embed=True)] = 0.3
):
    """Document 간 관계 분석 및 생성 (Step 3)

    Args:
        doc_id: 분석할 Document ID (None이면 전체 문서 분석)
        top_k: 관련 문서 최대 개수 (기본: 4)
        text_weight: 텍스트 유사도 가중치 (기본: 0.4)
        graph_weight: 그래프 임베딩 유사도 가중치 (기본: 0.3)
        normalized_value_weight: NormalizedValue 공유 가중치 (기본: 0.3)

    Flow:
        Step 1: /metadata/extract - 문서 추가 (text_embedding)
        Step 2: /graph/embeddings - 그래프 임베딩 생성
        Step 3: /graph/relations/analyze - 관계 분석 ← 현재

    Note:
        - 하이브리드 검색: 텍스트 임베딩 + 그래프 임베딩 + NormalizedValue 공유
        - NormalizedValue 공유로 Key-Value 쌍의 실제 값 유사도 반영
        - LLM으로 의미적 관계 추론 (SIMILAR_TO, EXTENDS, CONTRADICTS 등)
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
        graph_weight=graph_weight,
        normalized_value_weight=normalized_value_weight
    )
    return APIResponseModel(
        result=result,
        description=f"관계 분석 완료: {result.get('total_relations_created', 0)}개 관계 생성"
    )


@router.post("/search", tags=["graph"], response_model=GraphRAGSearchResponse,
             response_class=JSONResponse)
async def search_graph_rag(request: GraphRAGSearchRequest):
    """사용자 질의 기반 Graph RAG 검색 (Key-Value 구조)

    Args:
        request: GraphRAGSearchRequest
            - query: 사용자 질의
            - top_k: 반환할 Document 개수 (기본: 5, 최대: 20)

    Returns:
        GraphRAGSearchResponse
            - query: 사용자 질의
            - documents: 검색된 Document 목록 (벡터 유사도 순)
            - graph: Key-Value 그래프 구조

    Note:
        - 텍스트 임베딩 벡터 검색 + Reranker + LLM 검증
        - Key-Value 그래프 구조 반환
        - 답변 생성(Generation)은 포함하지 않음

    Example:
        Request:
        {
            "query": "병상수가 100개 이상인 병원",
            "top_k": 5
        }

        Response:
        {
            "query": "병상수가 100개 이상인 병원",
            "documents": [
                {
                    "doc_id": "hospital_001",
                    "text": "...",
                    "description": "병원데이터_test.jsonl",
                    "score": 0.95
                }
            ],
            "graph": {
                "nodes": [...],
                "relationships": [...]
            }
        }
    """
    result = graph_service.graphrag_service.search_graph(
        query=request.query,
        top_k=request.top_k
    )

    return GraphRAGSearchResponse(**result)




@router.get("/viewer", tags=["graph"], include_in_schema=False)
async def graph_viewer():
    """Graph RAG 시각화 페이지로 리다이렉트

    Note:
        /static/graph-viewer.html로 리다이렉트
    """
    return RedirectResponse(url="/static/graph-viewer.html")
