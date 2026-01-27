from typing import Annotated

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

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
        doc_id: Annotated[str, Body(embed=True)] = None,
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
