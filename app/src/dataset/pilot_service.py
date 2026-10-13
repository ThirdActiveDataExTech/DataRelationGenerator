"""실증 기관 — 기관 보유 데이터(1st 파티) 목록.

기관 사용자는 자기 데이터를 골라 상세 화면에서 함께 쓸 외부 개방 데이터를 본다
(연계 조회·관계 생성은 ExploreService 가 모든 데이터셋에 공통으로 제공).
"""

from app.repositories.dataset_repository import DatasetRepository, get_dataset_repository
from app.schemas.pilot import PilotOrg, PilotOrgSummary
from app.src.dataset.explore_service import ExploreService, explore_service, to_card

PILOT_ORGS: dict[str, PilotOrg] = {
    org.org_id: org
    for org in (
        PilotOrg(
            org_id="sbsi",
            name="SBSi",
            providers=["에스비에스아이"],
            summary="SBS 방송 프로그램·회차·출연진 데이터를 기준으로 함께 활용할 수 있는 외부 개방 데이터를 찾습니다.",
        ),
        PilotOrg(
            org_id="yuhs",
            name="연세대학교의료원",
            providers=["연세대학교의료원", "연세대학교 세브란스병원", "원주세브란스기독병원"],
            summary=(
                "세브란스 컨소시엄 의료데이터의 메타데이터를 기준으로 건강보험심사평가원 요양기관 데이터 등 "
                "연계 가능한 공공데이터를 찾습니다. 원본 의료데이터는 사용하지 않습니다."
            ),
        ),
    )
}


class PilotService:
    """실증 기관별 보유 데이터."""

    def __init__(self, dataset_repo: DatasetRepository | None = None, explore: ExploreService | None = None):
        """저장소와 탐색 서비스(관계 수 집계)를 주입받아 초기화."""
        self.dataset_repo = dataset_repo or get_dataset_repository()
        self.explore = explore or explore_service

    def list_orgs(self) -> list[PilotOrgSummary]:
        """실증 기관 목록과 기관 보유 데이터 — 1st 파티이면서 기관 provider 로 등록된 데이터셋."""
        profiles = [p for p in self.dataset_repo.list_all() if p.party == "1st"]
        counts = self.explore.relation_counts([p.table_name for p in profiles])
        return [
            PilotOrgSummary(
                **org.model_dump(),
                datasets=[to_card(p, counts[p.table_name]) for p in profiles if p.provider in org.providers],
            )
            for org in PILOT_ORGS.values()
        ]


pilot_service = PilotService()
