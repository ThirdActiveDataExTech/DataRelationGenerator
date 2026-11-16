"""자동 트리거 + job 영속화(DRG-6) 검증 — job_store, 증분 파이프라인, 동기화 사이클."""

from app.schemas.dataset import DatasetProfile
from app.src.dataset.job_store import JobStore


def _profile(name, ident="", records=0) -> DatasetProfile:
    return DatasetProfile(
        table_name=name, catalog_identifier=ident or f"id-{name}", domain="d",
        record_count=records, column_count=1, column_names=["c"], sample_records=[{}],
    )


# ── job_store ────────────────────────────────────────────────

def test_job_store_crud(tmp_path):
    store = JobStore(db_path=str(tmp_path / "jobs.db"))
    assert store.get("nope") is None

    store.put("j1", "relation_pipeline", {"status": "running", "table_names": []})
    assert store.get("j1")["status"] == "running"

    store.update("j1", status="done", total_saved=3)
    job = store.get("j1")
    assert job["status"] == "done" and job["total_saved"] == 3 and job["table_names"] == []

    store.update("missing", status="done")  # 없는 job 은 무시

    # 재시작(새 인스턴스)에도 상태 유지
    store2 = JobStore(db_path=str(tmp_path / "jobs.db"))
    assert store2.get("j1")["status"] == "done"


# ── 파이프라인 증분 모드 ─────────────────────────────────────

def _pipeline_with_fakes(monkeypatch, profiles, existing_pairs):
    from app.src.dataset.relation_pipeline import RelationPipeline

    class FakeRepo:
        def list_all(self):
            return profiles

        def get(self, name):
            return next((p for p in profiles if p.table_name == name), None)

    class FakeService:
        def get_existing_relation_pairs(self):
            return existing_pairs

    pipeline = RelationPipeline.__new__(RelationPipeline)
    pipeline.dataset_repo = FakeRepo()
    pipeline.dataset_service = FakeService()
    ran_pairs = []

    def fake_run_pair(a, b, t2, t3):
        from app.schemas.dataset import RelationPipelineResult
        ran_pairs.append(tuple(sorted([a.table_name, b.table_name])))
        return RelationPipelineResult(table_a=a.table_name, table_b=b.table_name)

    monkeypatch.setattr(pipeline, "_run_pair", fake_run_pair)
    return pipeline, ran_pairs


def test_run_empty_table_names_means_all(monkeypatch):
    profiles = [_profile("a"), _profile("b"), _profile("c")]
    pipeline, ran = _pipeline_with_fakes(monkeypatch, profiles, existing_pairs=set())
    results = pipeline.run(table_names=[])
    assert len(results) == 3  # 3C2 전 쌍
    assert ("a", "b") in ran and ("a", "c") in ran and ("b", "c") in ran


def test_run_skips_existing_pairs_unless_refreshed(monkeypatch):
    profiles = [_profile("a"), _profile("b"), _profile("c")]
    existing = {("a", "b"), ("a", "c")}
    pipeline, ran = _pipeline_with_fakes(monkeypatch, profiles, existing)

    pipeline.run(table_names=[])
    assert ran == [("b", "c")]  # 기존 쌍 스킵

    ran.clear()
    pipeline.run(table_names=[], refresh_table_names=["a"])
    assert ("a", "b") in ran and ("a", "c") in ran  # a가 낀 기존 쌍 재평가
    assert ("b", "c") in ran  # 신규 쌍은 여전히 포함


def test_run_seed_table_names_limits_pairs(monkeypatch):
    """기준(seed) 테이블 지정 시 해당 테이블이 낀 쌍만 평가 — 1st 파티 기준 연계 탐색."""
    profiles = [_profile("a"), _profile("b"), _profile("c")]
    pipeline, ran = _pipeline_with_fakes(monkeypatch, profiles, existing_pairs=set())
    results = pipeline.run(table_names=[], seed_table_names=["a"])
    assert sorted(ran) == [("a", "b"), ("a", "c")]  # b-c 쌍은 제외
    assert len(results) == 2


# ── 동기화 사이클 ────────────────────────────────────────────

def _sync_service(tmp_path, entries, detail, existing_profiles):
    from app.src.dataset.catalog_sync import CatalogSyncService

    class FakeCatalog:
        def list_entries_updated_between(self, date_from, date_to):
            return entries

        def get_entry(self, entry_id):
            return detail[entry_id]

    class FakeRepo:
        def get_by_identifier(self, ident):
            return existing_profiles.get(ident)

    ingested, pipeline_calls = [], []

    class FakeDatasetService:
        def ingest_table(self, records, **kwargs):
            ingested.append(kwargs["catalog_identifier"])
            return _profile(kwargs["table_name"], kwargs["catalog_identifier"])

    class FakePipeline:
        def run(self, table_names, refresh_table_names=None, seed_table_names=None, **kw):
            pipeline_calls.append((table_names, refresh_table_names, seed_table_names))
            return []

    svc = CatalogSyncService(
        catalog_client=FakeCatalog(), dataset_repo=FakeRepo(),
        dataset_service=FakeDatasetService(), pipeline=FakePipeline(),
        job_store=JobStore(db_path=str(tmp_path / "jobs.db")),
    )
    return svc, ingested, pipeline_calls


def test_sync_cycle_new_and_changed(tmp_path):
    entries = [
        {"id": 1, "identifier": "id-new"},
        {"id": 2, "identifier": "id-meta"},   # 기등록 (메타데이터-only) → 재등록 + refresh
        {"id": 3, "identifier": "id-full"},   # 기등록 (실데이터 프로파일) → 재등록 생략 + refresh
    ]
    detail = {
        1: {"identifier": "id-new", "title": "신규", "description": "d", "keyword": [], "theme": []},
        2: {"identifier": "id-meta", "title": "메타", "description": "d", "keyword": [], "theme": []},
    }
    existing = {
        "id-meta": _profile("메타", "id-meta", records=0),
        "id-full": _profile("실데이터", "id-full", records=100),
    }
    svc, ingested, pipeline_calls = _sync_service(tmp_path, entries, detail, existing)

    summary = svc.run_cycle()

    assert ingested == ["id-new", "id-meta"]  # 실데이터 프로파일은 덮어쓰지 않음
    # 상대 후보는 전체, 평가 쌍은 동기화된 3개 테이블이 낀 쌍만 + 변동분 재평가
    assert pipeline_calls == [([], ["메타", "실데이터"], ["신규", "메타", "실데이터"])]
    assert summary["synced_tables"] == 3 and summary["refreshed_tables"] == 2
    assert svc._get_watermark() == summary["date_to"]  # watermark 갱신


def test_sync_cycle_failure_isolated(tmp_path):
    """엔트리 하나 실패해도 나머지는 계속 처리."""
    entries = [{"id": 1, "identifier": "id-bad"}, {"id": 2, "identifier": "id-ok"}]
    detail = {2: {"identifier": "id-ok", "title": "ok", "description": "d", "keyword": [], "theme": []}}
    # id-bad 는 detail 에 없음 → get_entry 에서 KeyError
    svc, ingested, pipeline_calls = _sync_service(tmp_path, entries, detail, existing_profiles={})

    summary = svc.run_cycle()
    assert ingested == ["id-ok"]
    assert summary["failed_identifiers"] == ["id-bad"]
    assert len(pipeline_calls) == 1


def test_sync_cycle_noop_without_entries(tmp_path):
    svc, ingested, pipeline_calls = _sync_service(tmp_path, [], {}, {})
    summary = svc.run_cycle(date_from="2026-07-01")  # 백필 date_from 재정의
    assert summary["date_from"] == "2026-07-01"
    assert ingested == [] and pipeline_calls == []  # 대상 없으면 파이프라인 미실행
