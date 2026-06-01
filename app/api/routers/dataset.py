import json
from typing import Annotated

import logging
from fastapi import APIRouter, Body, UploadFile, File
from fastapi.responses import JSONResponse

from app.schemas.dataset import (
    DatasetIngestRequest,
    DatasetRelation,
    DatasetSearchRequest,
    RunRelationPipelineRequest,
)
from app.schemas.response import APIResponseModel
from app.src.dataset.dataset_service import dataset_service
from app.src.dataset.relation_pipeline import relation_pipeline

router = APIRouter(prefix="/dataset", tags=["dataset"])
logger = logging.getLogger(__name__)


@router.post(
    "/ingest",
    response_model=APIResponseModel,
    response_class=JSONResponse,
    summary="테이블 메타데이터 추출 및 Neptune/OpenSearch 적재",
)
async def ingest_dataset(
        file: Annotated[UploadFile, File(description="JSON 배열 파일 [{...}, {...}, ...]")],
        table_name: Annotated[str, Body(embed=True, description="테이블명")],
        domain: Annotated[str, Body(embed=True, description="도메인 (medical, broadcast, general 등)")],
        description: Annotated[str, Body(embed=True, description="짧은 설명")] = "",
        table_summary: Annotated[str, Body(embed=True, description="상세 요약 (비어있으면 description 사용)")] = "",
):
    """JSON 파일을 업로드하면 메타데이터를 추출하여 Neptune Dataset 노드와 OpenSearch에 저장.

    Flow:
        1. 파일에서 레코드 로드
        2. 기본 프로파일 추출 (record_count, column_names, sample_records)
        3. table_summary 임베딩 생성
        4. Neptune에 Dataset 노드 저장 (tableName, domain, description)
        5. OpenSearch datasets 인덱스에 저장 (KNN 검색용)

    Returns:
        저장된 DatasetProfile 정보
    """
    content = await file.read()
    records = json.loads(content.decode("utf-8"))

    if not isinstance(records, list):
        records = [records]

    profile = dataset_service.ingest_table(
        records=records,
        table_name=table_name,
        domain=domain,
        description=description,
        table_summary=table_summary,
    )

    return APIResponseModel(
        result=profile.model_dump(exclude={"table_embedding"}),
        description=(
            f"Dataset '{table_name}' 등록 완료: "
            f"{profile.record_count}건, {profile.column_count}개 컬럼"
        ),
    )


@router.post(
    "/ingest_json",
    response_model=APIResponseModel,
    response_class=JSONResponse,
    summary="JSON 바디로 테이블 메타데이터 적재",
)
async def ingest_dataset_json(
        request: DatasetIngestRequest,
        records: Annotated[list[dict], Body(embed=True, description="레코드 목록")],
):
    """JSON 바디로 레코드와 메타데이터를 함께 전달하여 적재."""
    profile = dataset_service.ingest_table(
        records=records,
        table_name=request.table_name,
        domain=request.domain,
        description=request.description,
        table_summary=request.table_summary,
    )

    return APIResponseModel(
        result=profile.model_dump(exclude={"table_embedding"}),
        description=f"Dataset '{request.table_name}' 등록 완료",
    )


@router.post(
    "/relation",
    response_model=APIResponseModel,
    response_class=JSONResponse,
    summary="테이블 간 관계 Neptune 저장",
)
async def save_relation(relation: DatasetRelation):
    """두 테이블 간 관계를 Neptune RDF Reification으로 저장.

    Args:
        relation: DatasetRelation (table_a, table_b, score, method, relation_type)

    method 값:
        - rule: FK 또는 동일 컬럼명 자동 확정 (Tier 1)
        - embedding: KNN 유사도 기반 (Tier 2)
        - llm: LLM 검증 통과 (Tier 3)
    """
    dataset_service.save_relation(relation)

    return APIResponseModel(
        result=relation.model_dump(),
        description=(
            f"관계 저장 완료: {relation.table_a} --[{relation.relation_type}]--> {relation.table_b} "
            f"(score={relation.score}, method={relation.method})"
        ),
    )


@router.post(
    "/search",
    response_model=APIResponseModel,
    response_class=JSONResponse,
    summary="쿼리로 유사 테이블 검색 (KNN)",
)
async def search_datasets(request: DatasetSearchRequest):
    """쿼리 텍스트 임베딩으로 유사 테이블 KNN 검색.

    Args:
        request: DatasetSearchRequest (query, domain, top_k)

    Returns:
        유사 테이블 목록 (score 내림차순)
    """
    results = dataset_service.search_datasets(
        query=request.query,
        domain=request.domain,
        top_k=request.top_k,
    )

    return APIResponseModel(
        result=[r.model_dump() for r in results],
        description=f"'{request.query}' 검색 결과: {len(results)}개",
    )


@router.post(
    "/relation/pipeline",
    response_model=APIResponseModel,
    response_class=JSONResponse,
    summary="3-Tier 관계 생성 파이프라인 실행",
)
async def run_relation_pipeline(request: RunRelationPipelineRequest):
    """테이블 목록에 대해 3-Tier 관계 생성 파이프라인을 실행.

    **Tier 1 (rule)**: 동일 컬럼명 자동 확정 (score=1.0, method=rule)
    - 두 테이블에 동일한 컬럼명이 있으면 즉시 관계 확정
    - Tier 1에서 확정된 쌍은 Tier 2/3 생략

    **Tier 2 (embedding)**: table_embedding 코사인 유사도 계산
    - tier2_threshold 이상인 쌍만 Tier 3으로 진행

    **Tier 3 (llm)**: LLM이 샘플 레코드 + 메타데이터 보고 관계 검증
    - tier3_confidence_threshold 이상인 쌍만 Neptune에 저장 (method=llm)

    Args:
        request.table_names: 대상 테이블명 목록 (OpenSearch에 등록된 테이블만 처리)
        request.tier2_threshold: Tier 2 통과 임계값 (기본 0.75)
        request.tier3_confidence_threshold: Tier 3 통과 임계값 (기본 0.7)

    Returns:
        테이블 쌍별 처리 결과 + 저장된 관계 목록
    """
    results = relation_pipeline.run(
        table_names=request.table_names,
        tier2_threshold=request.tier2_threshold,
        tier3_confidence_threshold=request.tier3_confidence_threshold,
    )

    total_saved = sum(len(r.saved_relations) for r in results)
    tier1_count = sum(len(r.tier1) for r in results)
    tier3_count = sum(len(r.tier3_verified) for r in results)

    return APIResponseModel(
        result=[r.model_dump() for r in results],
        description=(
            f"파이프라인 완료: {len(results)}쌍 처리, "
            f"Tier1={tier1_count}개, Tier3={tier3_count}개, "
            f"총 {total_saved}개 관계 저장"
        ),
    )
