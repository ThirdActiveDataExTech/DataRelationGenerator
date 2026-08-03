"""① 통합 카탈로그(active-metadata-management catalog-service) HTTP 클라이언트.

카탈로그 엔트리를 pull 하여 ②의 데이터셋 등록(ingest_from_catalog)에 사용한다.
응답 포맷은 catalog-service 의 APIResponseModel({"result": ...}) 를 따른다.
"""

import logging
from typing import TYPE_CHECKING, Any

import requests

from app.config import settings
from app.exceptions.service import CatalogEntryNotFoundError

if TYPE_CHECKING:
    from app.schemas.dataset import DatasetRelation

logger = logging.getLogger(__name__)


def dataset_relation_payload(relation: "DatasetRelation", status: str = "APPROVED") -> dict[str, Any]:
    """DatasetRelation → ① dataset_relation 생성 파라미터 매핑 (②→①).

    catalog_identifier_a/b 가 모두 있어야 환류 가능하다.
    status: APPROVED(자동 확정) 또는 PENDING(① 승인 대기 제안).
    """
    if not (relation.catalog_identifier_a and relation.catalog_identifier_b):
        raise ValueError("환류하려면 양측 catalog_identifier가 필요합니다")
    return {
        "source_identifier": relation.catalog_identifier_a,
        "target_identifier": relation.catalog_identifier_b,
        "relation_type": relation.relation_type,
        "score": relation.score,
        "method": relation.method,
        "status": status,
        "evidence": relation.evidence,
    }


class CatalogClient:
    """catalog-service REST 클라이언트 (x-token 헤더 인증)."""

    def __init__(self, base_url: str | None = None, x_token: str | None = None):
        self.base_url = (base_url or settings.CATALOG_SERVICE_URL).rstrip("/")
        self.headers = {"x-token": x_token or settings.CATALOG_SERVICE_X_TOKEN}

    def _get(self, path: str, params: dict | None = None) -> Any:
        resp = requests.get(
            f"{self.base_url}{path}", headers=self.headers, params=params, timeout=30
        )
        resp.raise_for_status()
        return resp.json().get("result")

    def push_dataset_relation(self, relation: "DatasetRelation", status: str = "APPROVED") -> Any:
        """②의 관계 생성 결과를 ①의 dataset_relation 으로 환류 (POST /relation/datasets)."""
        resp = requests.post(
            f"{self.base_url}/relation/datasets",
            headers=self.headers,
            json=dataset_relation_payload(relation, status=status),
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("result")

    def get_entry(self, catalog_entry_id: int) -> dict[str, Any]:
        """카탈로그 엔트리 단건 조회 (숫자 PK)."""
        result = self._get(f"/catalog/entries/{catalog_entry_id}")
        if not result:
            raise CatalogEntryNotFoundError(catalog_entry_id)
        return result

    def list_entries_updated_between(self, date_from: str, date_to: str) -> list[dict[str, Any]]:
        """updated_at 기준 갱신 엔트리 목록 조회 (YYYY-MM-DD, 전체 페이지 수집).

        반환 항목은 목록용 요약(CatalogEntrySummary — description 없음)이므로,
        등록에 쓰려면 get_entry(id) 로 상세를 재조회해야 한다.
        """
        limit, offset, entries = 100, 0, []
        for _ in range(settings.CATALOG_SERVICE_MAX_PAGES):
            page = self._get(
                "/catalog/entries",
                params={
                    "date_field": "updated_at",
                    "date_from": date_from,
                    "date_to": date_to,
                    "limit": limit,
                    "offset": offset,
                },
            ) or []
            entries.extend(page)
            if len(page) < limit:
                break
            offset += limit
        return entries

    def find_entry_by_identifier(self, identifier: str) -> dict[str, Any]:
        """DCAT identifier 로 엔트리 조회.

        ponytail: catalog-service 검색 API에 identifier 필터가 없어 목록을 페이지 순회하며
        클라이언트에서 매칭한다. ①에 identifier 검색 파라미터가 추가되면 단건 호출로 교체.
        """
        limit = 100
        for page in range(settings.CATALOG_SERVICE_MAX_PAGES):
            entries = self._get(
                "/catalog/entries", params={"limit": limit, "offset": page * limit}
            ) or []
            for entry in entries:
                if entry.get("identifier") == identifier:
                    # 목록 응답(CatalogEntrySummary)에는 없는 필드가 있을 수 있어 상세 재조회
                    return self.get_entry(entry["id"])
            if len(entries) < limit:
                break
        raise CatalogEntryNotFoundError(identifier)


def ingest_args_from_entry(
    entry: dict[str, Any],
    table_name: str | None = None,
    domain: str | None = None,
    party: str = "3rd",
) -> dict[str, Any]:
    """카탈로그 엔트리(CatalogEntryResponse dict) → ingest_table 인자 매핑.

    매핑 규칙 (①→②):
        identifier      → catalog_identifier (연동 키)
        title           → table_name(기본값), description(짧은 설명)
        description     → table_summary (임베딩 소스)
        keyword + theme → keywords (시드 키워드, LLM 추출 대체)
        publisher       → provider (제공 기관)
        landing_page/access_url → landing_page (수집 단계 원본 제공처 링크)
        domain          → 미지정 시 빈 값 → ingest_table 의 특징 기반 자동 분류에 위임
        party           → 기본 3rd (카탈로그 개방 데이터)
    """
    identifier = entry.get("identifier", "")
    if not identifier:
        raise ValueError("카탈로그 엔트리에 identifier가 없습니다")

    title = entry.get("title") or ""
    description = entry.get("description") or ""
    keywords = list(dict.fromkeys((entry.get("keyword") or []) + (entry.get("theme") or [])))

    return {
        "catalog_identifier": identifier,
        "table_name": table_name or title or identifier,
        "domain": domain or "",
        "description": title,
        "table_summary": description or title or identifier,
        "keywords": keywords,
        "party": party,
        "provider": entry.get("publisher") or "",
        "landing_page": entry.get("landing_page") or entry.get("access_url") or "",
    }


def get_catalog_client() -> CatalogClient:
    return CatalogClient()
