"""데이터 찾기 API — 분야 탐색(디지털 집현전 분류), 검색, 상세, 함께 쓸 수 있는 데이터."""

from typing import Annotated, Literal
from urllib.parse import quote

from fastapi import APIRouter, Query
from fastapi.responses import Response

from app.api.routers.dataset import run_relation_pipeline
from app.schemas.dataset import RunRelationPipelineRequest
from app.schemas.response import APIResponseModel
from app.src.dataset.explore_service import explore_service

# 조회는 원장(PostgreSQL)·임베딩 동기 호출이라 def 로 두어 스레드풀에서 실행한다
router = APIRouter(prefix="/explore", tags=["Explore"])


@router.get("/overview", response_model=APIResponseModel, summary="전체 수·분류표(분야별 수)·제공기관 목록")
def overview():
    """홈·검색 화면 공통 데이터 — 분류표는 디지털 집현전 7대분류·중분류 전체(0건 포함)."""
    result = explore_service.overview()
    return APIResponseModel(
        result=result, description=f"데이터셋 {result['dataset_count']}개, 관계 {result['relation_count']}개"
    )


@router.get("/datasets", response_model=APIResponseModel, summary="데이터셋 검색·분야 탐색")
def search_datasets(
        q: Annotated[str, Query(description="검색어 — 제목·요약·키워드 일치 또는 의미 유사")] = "",
        category: Annotated[str, Query(description="대분류 (예: 의료)")] = "",
        subcategory: Annotated[str, Query(description="'대분류 > 중분류' (예: 의료 > 공중보건)")] = "",
        party: Annotated[Literal["", "1st", "3rd"], Query(description="1st 기관 보유 / 3rd 외부 개방")] = "",
        provider: Annotated[str, Query(description="제공기관")] = "",
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
        offset: Annotated[int, Query(ge=0)] = 0,
):
    """검색어가 있으면 텍스트 일치 → 의미 유사도 순, 없으면 최근 등록 순."""
    page = explore_service.search(q, category, subcategory, party, provider, limit, offset)
    return APIResponseModel(result=page.model_dump(), description=f"{page.total}건 중 {len(page.items)}건")


@router.get("/datasets/{table_name}", response_model=APIResponseModel, summary="데이터셋 상세")
def dataset_detail(table_name: str):
    """카드 정보 + 컬럼·샘플."""
    detail = explore_service.detail(table_name)
    return APIResponseModel(result=detail.model_dump(), description=f"'{table_name}' 상세")


@router.get(
    "/datasets/{table_name}/linkable",
    response_model=APIResponseModel,
    summary="함께 쓸 수 있는 데이터 (관계 유형별)",
    responses={200: {"content": {"text/csv": {}}, "description": "format=csv 이면 CSV 파일"}},
)
def linkable(table_name: str, format: Literal["json", "csv"] = "json"):
    """기준 데이터와 연계 가능한 데이터를 관계 유형별로 반환. 기관 시스템 연동·보고용 CSV 지원.

    - 묶음 순서: 속성 보완(DETAIL_FOR) → 집계 맥락(STATISTIC_FOR) → 인접 맥락(ADJACENT_TO) → 동종 판본(SAME_KIND)
    - `candidates`: 아직 관계 판정 전인 의미 유사 후보 — `POST .../generate` 로 판정
    """
    result = explore_service.linkable(table_name)
    if format == "csv":
        filename = quote(f"{result.anchor.display_name or table_name}_연계데이터.csv")
        return Response(
            content=explore_service.to_csv(result),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"},
        )
    judged = sum(len(g.items) for g in result.groups)
    return APIResponseModel(
        result=result.model_dump(),
        description=f"'{table_name}' 연계 데이터 {judged}건, 판정 전 후보 {len(result.candidates)}건",
    )


@router.post(
    "/datasets/{table_name}/generate",
    response_model=APIResponseModel,
    summary="이 데이터 기준 관계 생성 실행 (비동기)",
)
async def generate(table_name: str):
    """의미가 가까운 외부 개방 데이터 상위 20건 + 컬럼이 겹치는 외부 개방 데이터와의 관계를 판정한다.

    진행 상태는 `GET /dataset/relation/pipeline/{job_id}` 로 조회.
    """
    return await run_relation_pipeline(RunRelationPipelineRequest(
        table_names=explore_service.generation_targets(table_name),
        seed_table_names=[table_name],
    ))
