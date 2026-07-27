"""카탈로그 변경 감지 → 데이터셋 동기화 → 관계 파이프라인 증분 실행 (액티브 루프).

외부 스케줄러 없이 ② 단독으로 "카탈로그에 데이터가 실리면 관계가 자동으로 생긴다"
루프를 완성한다:

  ① catalog-service 갱신 엔트리 조회 (updated_at 기준, watermark 날짜부터)
  → 미등록 엔트리는 신규 등록, 기등록 엔트리는 변동 반영(refresh 대상 표시)
  → 관계 파이프라인 증분 실행 (신규 쌍 + refresh 테이블이 낀 쌍 재평가)
  → watermark 갱신

트리거 방식:
  - POST /dataset/sync_from_catalog (수동 또는 외부 스케줄러의 단발 호출)
  - CATALOG_SYNC_ENABLED=True 시 앱 내장 주기 폴링 루프 (CATALOG_SYNC_INTERVAL_MINUTES)

watermark 는 날짜(YYYY-MM-DD) 단위 — 카탈로그 검색 필터가 date 정밀도라 같은 날짜의
엔트리는 다음 주기에 한 번 더 조회될 수 있으나, 재등록/재평가는 대체(upsert) 의미론이라
결과는 동일하다 (중복 실행 비용만 소폭 발생).
"""

import asyncio
import logging
from datetime import datetime, timezone

from app.config import settings

logger = logging.getLogger(__name__)

_WATERMARK_JOB_ID = "catalog_sync_watermark"


class CatalogSyncService:
    """카탈로그 → 관계 생성 서비스 동기화 사이클."""

    def __init__(self, catalog_client=None, dataset_repo=None, dataset_service=None, pipeline=None, job_store=None):
        """의존성 주입 (기본값은 각 모듈 싱글턴 — 테스트에서 페이크 주입)."""
        if catalog_client is None:
            from app.clients.catalog_client import get_catalog_client
            catalog_client = get_catalog_client()
        if dataset_repo is None:
            from app.repositories.dataset_repository import get_dataset_repository
            dataset_repo = get_dataset_repository()
        if dataset_service is None:
            from app.src.dataset.dataset_service import dataset_service as ds
            dataset_service = ds
        if pipeline is None:
            from app.src.dataset.relation_pipeline import relation_pipeline
            pipeline = relation_pipeline
        if job_store is None:
            from app.src.dataset.job_store import get_job_store
            job_store = get_job_store()
        self.catalog_client = catalog_client
        self.dataset_repo = dataset_repo
        self.dataset_service = dataset_service
        self.pipeline = pipeline
        self.job_store = job_store

    def run_cycle(self, date_from: str | None = None) -> dict:
        """동기화 사이클 1회 실행.

        Args:
            date_from: 조회 시작일(YYYY-MM-DD) 재정의 — 과거분 백필용. 생략 시 watermark
                (최초 실행이면 오늘 — 과거 전체 스윕은 명시적 백필로만)

        Returns:
            사이클 요약 (조회/등록/재평가/관계 수)
        """
        from app.clients.catalog_client import ingest_args_from_entry

        today = datetime.now(timezone.utc).date().isoformat()
        start = date_from or self._get_watermark() or today

        summaries = self.catalog_client.list_entries_updated_between(start, today)
        tables: list[str] = []
        refresh_tables: list[str] = []
        failed: list[str] = []

        for summary in summaries:
            identifier = summary.get("identifier")
            if not identifier:
                continue
            try:
                existing = self.dataset_repo.get_by_identifier(identifier)
                if existing is not None and existing.record_count > 0:
                    # 실데이터로 프로파일링된 데이터셋 — 메타데이터-only 재등록으로
                    # 컬럼 정보를 덮어쓰지 않고, 관계 재평가 대상으로만 표시
                    tables.append(existing.table_name)
                    refresh_tables.append(existing.table_name)
                    continue

                entry = self.catalog_client.get_entry(summary["id"])  # 목록엔 description 없음
                args = ingest_args_from_entry(entry)
                profile = self.dataset_service.ingest_table(records=[], **args)
                tables.append(profile.table_name)
                if existing is not None:
                    refresh_tables.append(profile.table_name)
            except Exception as e:
                logger.warning(f"[CatalogSync] 엔트리 동기화 실패 ({identifier}): {e}")
                failed.append(identifier)

        results = []
        if tables:
            results = self.pipeline.run(table_names=[], refresh_table_names=refresh_tables)

        self._set_watermark(today)
        cycle_summary = {
            "date_from": start,
            "date_to": today,
            "entries": len(summaries),
            "synced_tables": len(tables),
            "refreshed_tables": len(refresh_tables),
            "failed_identifiers": failed,
            "pairs_processed": len(results),
            "relations_saved": sum(len(r.saved_relations) for r in results),
            "relations_pending": sum(len(r.pending_relations) for r in results),
        }
        logger.info(f"[CatalogSync] 사이클 완료: {cycle_summary}")
        return cycle_summary

    def _get_watermark(self) -> str | None:
        state = self.job_store.get(_WATERMARK_JOB_ID)
        return state.get("date") if state else None

    def _set_watermark(self, date_str: str) -> None:
        self.job_store.put(_WATERMARK_JOB_ID, "state", {"date": date_str})


def get_catalog_sync_service() -> CatalogSyncService:
    return CatalogSyncService()


async def periodic_catalog_sync() -> None:
    """앱 내장 주기 동기화 루프 (CATALOG_SYNC_ENABLED=True 일 때 lifespan 에서 기동)."""
    interval_sec = settings.CATALOG_SYNC_INTERVAL_MINUTES * 60
    logger.info(f"[CatalogSync] 주기 동기화 시작 (interval={settings.CATALOG_SYNC_INTERVAL_MINUTES}분)")
    loop = asyncio.get_event_loop()
    while True:
        try:
            await loop.run_in_executor(None, get_catalog_sync_service().run_cycle)
        except Exception as e:
            logger.warning(f"[CatalogSync] 주기 동기화 실패 (다음 주기에 재시도): {e}")
        await asyncio.sleep(interval_sec)
