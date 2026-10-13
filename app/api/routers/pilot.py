"""실증 기관(SBSi·연세의료원) API — 기관별 보유 데이터 목록.

보유 데이터 1건 기준의 연계 조회·관계 생성은 /explore/datasets/{table_name}/... 를 쓴다.
"""

from fastapi import APIRouter

from app.schemas.response import APIResponseModel
from app.src.dataset.pilot_service import pilot_service

router = APIRouter(prefix="/pilot", tags=["Pilot"])


@router.get(
    "/orgs",
    response_model=APIResponseModel,
    summary="실증 기관 목록과 기관별 보유 데이터",
)
def list_orgs():
    """실증 기관과 기관 보유 데이터(1st 파티, provider 기준) 목록."""
    orgs = pilot_service.list_orgs()
    return APIResponseModel(
        result=[o.model_dump() for o in orgs],
        description=", ".join(f"{o.name} {len(o.datasets)}건" for o in orgs),
    )
