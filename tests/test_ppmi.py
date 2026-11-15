"""컬럼·값 특징 PPMI·SVD 상호 연관성 분석 검증 — 특징 집계·PPMI 수치·SVD 임베딩·Tier2 결합·evidence."""

import numpy as np

from app.schemas.dataset import DatasetProfile, Tier2Candidate, Tier3Result
from app.src.dataset.ppmi_analyzer import PPMIAnalyzer, feature_counts, ppmi_matrix, svd_embeddings
from app.utils.vector import cosine_similarity


def _profile(name, column_names=("c",), sample_records=({},)) -> DatasetProfile:
    return DatasetProfile(
        table_name=name, catalog_identifier=f"id-{name}", domain="d",
        record_count=1, column_count=len(column_names), column_names=list(column_names), nature="record",
        sample_records=list(sample_records), table_embedding=[1.0, 0.0],
    )


# ── 특징 집계 ────────────────────────────────────────────────

def test_feature_counts_columns_and_identity_values():
    profile = _profile(
        "a",
        column_names=["시도명", "어린이집 명", "정원", "법정동코드", "비고"],
        sample_records=[
            {"시도명": "경기도", "어린이집 명": "햇살어린이집", "정원": 99, "법정동코드": "4113510900", "비고": ""},
            {"시도명": "경기도", "어린이집 명": "숲속어린이집", "정원": 120, "법정동코드": "4113510900", "비고": None},
        ],
    )
    counts = feature_counts(profile)
    assert counts["key:시도명"] == 2 and counts["key:어린이집명"] == 2  # 공백 제거·소문자
    assert counts["value:경기도"] == 2 and counts["value:4113510900"] == 2  # 범주·코드 값
    assert "value:99" not in counts and "value:120" not in counts  # 수량 측정값 제외
    assert counts["key:비고"] == 1  # 값 없는 컬럼은 구성만 1회


def test_feature_counts_schema_only_profile():
    counts = feature_counts(_profile("a", column_names=["기관명", "주소"], sample_records=[]))
    assert counts == {"key:기관명": 1, "key:주소": 1}


# ── PPMI 수치 ────────────────────────────────────────────────

# 코퍼스에는 비교 쌍 외 배경 데이터셋이 필요 (2개뿐이면 균등 특징의 PMI 가 0으로 붕괴)
BACKGROUND = {"c": {"key:장르": 20, "value:드라마": 15}, "d": {"key:노선": 12, "value:버스": 9}}


def _analyzer(counts_a, counts_b):
    analyzer = PPMIAnalyzer()
    analyzer.load({"a": counts_a, "b": counts_b, **BACKGROUND})
    return analyzer


def test_ppmi_identical_features_high_similarity():
    counts = {"key:지역": 10, "value:서울": 5, "key:병원명": 8}
    result = _analyzer(counts, dict(counts)).score_pair("a", "b")
    assert result["ppmi_cosine"] > 0.99
    assert set(result["shared_features"]) == set(counts)


def test_ppmi_disjoint_features_zero():
    result = _analyzer(
        {"key:지역": 5, "value:서울": 3}, {"key:편성": 4, "value:영화": 2}
    ).score_pair("a", "b")
    assert result["ppmi_cosine"] == 0.0 and result["shared_features"] == []


def test_ppmi_partial_overlap_and_shared_ordering():
    result = _analyzer(
        {"key:지역": 10, "value:서울": 9, "key:전용필드A": 5},
        {"key:지역": 2, "value:서울": 8, "key:전용필드B": 5},
    ).score_pair("a", "b")
    assert 0.0 < result["ppmi_cosine"] < 1.0
    assert result["shared_features"] == ["value:서울", "key:지역"]  # min count 내림차순


def test_ppmi_none_without_features():
    # b는 특징 없음
    analyzer = PPMIAnalyzer()
    analyzer.load({"a": {"key:x": 1}, "b": {}, **BACKGROUND})
    assert analyzer.score_pair("a", "b") is None


def test_ppmi_none_with_small_corpus():
    # 특징 보유 데이터셋이 3개 미만이면 PPMI 신호 미사용 (붕괴 방지)
    analyzer = PPMIAnalyzer()
    analyzer.load({"a": {"key:x": 1}, "b": {"key:x": 2}})
    assert analyzer.score_pair("a", "b") is None


# ── SVD 임베딩 ─────────────────────────────────────────────

# 특징 x·y·w·v·u — a={x}, b={y}, c={x,y}, d={w}, e={v,u}. a·b 는 c 를 통해서만 이어진다
LATENT_COUNTS = np.array(
    [[5, 0, 0, 0, 0], [0, 5, 0, 0, 0], [5, 5, 0, 0, 0], [0, 0, 9, 0, 0], [0, 0, 0, 7, 7]], dtype=float
)


def _cos(vectors, i, j):
    return cosine_similarity(vectors[i].tolist(), vectors[j].tolist())


def test_svd_reduces_to_effective_rank():
    assert svd_embeddings(ppmi_matrix(LATENT_COUNTS)).shape == (5, 4)  # 특징 5개, 유효 랭크 4


def test_svd_large_corpus_uses_truncated_components():
    # 특징 공간이 dim 의 2배를 넘으면 상위 dim 성분만 구한다 (데이터셋 수천 규모 대응)
    rng = np.random.default_rng(0)
    counts = (rng.random((300, 400)) < 0.05) * 1.0
    counts[:, 0] = 1.0  # 빈 행 방지
    assert svd_embeddings(ppmi_matrix(counts), dim=50).shape == (300, 50)


def test_svd_truncation_links_datasets_through_shared_cooccurrence():
    full = svd_embeddings(ppmi_matrix(LATENT_COUNTS))
    truncated = svd_embeddings(ppmi_matrix(LATENT_COUNTS), dim=3)
    assert _cos(full, 0, 1) == 0.0  # 전체 랭크에서는 공유 특징이 없으면 0
    assert _cos(truncated, 0, 1) > 0.9  # 절단하면 c 를 매개로 가까워진다
    assert _cos(truncated, 0, 3) == 0.0  # 무관한 d 는 여전히 0


# ── Tier2 결합 ───────────────────────────────────────────────

def _tier2(monkeypatch, ppmi_result, enabled=True):
    from app.config import settings
    from app.src.dataset.relation_pipeline import RelationPipeline

    monkeypatch.setattr(settings, "PPMI_ENABLED", enabled)
    pipeline = RelationPipeline.__new__(RelationPipeline)
    monkeypatch.setattr(pipeline, "_run_ppmi", lambda a, b: ppmi_result)
    return pipeline._run_tier2(_profile("a"), _profile("b"))  # 임베딩 코사인 = 1.0


def test_tier2_blends_ppmi(monkeypatch):
    cand = _tier2(monkeypatch, {"ppmi_cosine": 0.5, "shared_features": ["key:지역"]})
    assert abs(cand.score - (0.8 * 1.0 + 0.2 * 0.5)) < 1e-9  # (1-w)·emb + w·ppmi
    assert cand.embedding_score == 1.0 and cand.ppmi_score == 0.5
    assert cand.shared_features == ["key:지역"]


def test_tier2_without_features_uses_embedding_only(monkeypatch):
    cand = _tier2(monkeypatch, None)
    assert cand.score == 1.0 and cand.ppmi_score is None


def test_tier2_disabled_skips_ppmi(monkeypatch):
    cand = _tier2(monkeypatch, {"ppmi_cosine": 0.0, "shared_features": []}, enabled=False)
    assert cand.score == 1.0 and cand.ppmi_score is None


def test_run_ppmi_scores_from_run_corpus():
    from app.src.dataset.ppmi_analyzer import PPMIAnalyzer as Analyzer
    from app.src.dataset.relation_pipeline import RelationPipeline

    shared = [{"시도": "경기도", "시군구": "성남시"}]
    profiles = [
        _profile("a", ["시도", "시군구", "정원"], shared),
        _profile("b", ["시도", "시군구", "출생아수"], shared),
        _profile("c", ["노선", "정류장"], [{"노선": "9407", "정류장": "야탑역"}]),
        _profile("d", ["장르"], [{"장르": "드라마"}]),
    ]
    pipeline = RelationPipeline.__new__(RelationPipeline)
    pipeline._ppmi_analyzer = Analyzer()
    pipeline._ppmi_analyzer.load_profiles(profiles)  # run() 이 실행 단위로 구성하는 코퍼스
    result = pipeline._run_ppmi(profiles[0], profiles[1])
    assert result["ppmi_cosine"] > 0.5
    assert {"key:시도", "value:경기도"} <= set(result["shared_features"])


def test_run_ppmi_alone_without_corpus_is_none():
    from app.src.dataset.relation_pipeline import RelationPipeline

    pipeline = RelationPipeline.__new__(RelationPipeline)  # 코퍼스 없이 두 프로필만 → 붕괴 방지로 None
    assert pipeline._run_ppmi(_profile("a"), _profile("b")) is None


# ── evidence 전파 ────────────────────────────────────────────

def test_ppmi_recorded_in_relation_evidence(monkeypatch):
    from app.src.dataset import dataset_service as ds_module
    from app.src.dataset.dataset_service import DatasetService
    from app.src.dataset.relation_pipeline import RelationPipeline

    monkeypatch.setattr(ds_module.settings, "CATALOG_PUSH_ENABLED", False)

    class FakeRelationRepo:
        def replace(self, relation, status):
            pass

    svc = DatasetService.__new__(DatasetService)
    svc.relation_repo = FakeRelationRepo()

    pipeline = RelationPipeline.__new__(RelationPipeline)
    pipeline.dataset_service = svc
    monkeypatch.setattr(pipeline, "_run_tier1", lambda a, b: [])
    monkeypatch.setattr(pipeline, "_run_tier15", lambda a, b: [])
    monkeypatch.setattr(
        pipeline, "_run_tier2",
        lambda a, b: Tier2Candidate(table_a="a", table_b="b", score=0.9,
                                    embedding_score=0.95, ppmi_score=0.7,
                                    shared_features=["key:지역", "value:서울"]),
    )
    monkeypatch.setattr(
        pipeline, "_run_tier3",
        lambda a, b, es, th, cc: Tier3Result(table_a="a", table_b="b", relation_type="RELATED",
                                             evidence_type="semantic", confidence=0.9,
                                             reason="r", embedding_score=es),
    )
    result = pipeline._run_pair(_profile("a"), _profile("b"),
                                tier2_threshold=0.4, tier3_confidence_threshold=0.6)
    evidence = result.saved_relations[0].evidence
    assert evidence["ppmi"] == {"score": 0.7, "shared_features": ["key:지역", "value:서울"]}
