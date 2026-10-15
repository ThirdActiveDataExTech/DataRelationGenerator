"""KV 지식그래프 기반 PPMI·SVD 상호 연관성 분석.

  데이터셋-특징 출현 횟수(지식그래프 가중치) → 코퍼스 PPMI 행렬 → 절단 SVD 임베딩 → 코사인

특징 = KV 그래프의 Key 이름("key:...")과 Value 내용("value:...").
코퍼스는 파이프라인 실행 대상 전체 데이터셋 — 소수 데이터셋만 공유하는(희소한)
Key/Value 일수록 PPMI 가중치가 높아진다. SVD 는 수천 차원의 특징 공간을 최대 _SVD_DIM
차원으로 줄인다. 코퍼스의 유효 랭크가 _SVD_DIM 을 넘으면 하위 성분이 잘려, 특징을 직접
공유하지 않아도 같은 특징군과 함께 나타나는 데이터셋끼리 가까워진다.
임베딩(문장 의미) 신호와 상보적 — 설명 문구가 달라도 실제 값이 겹치면 잡아낸다.

주의: 코퍼스가 2개뿐이면 균등 분포 특징의 PMI 가 0으로 붕괴하므로, KV 특징을 가진
데이터셋이 3개 미만이면 None 을 반환한다 (임베딩 신호만 사용).
"""

import logging

import numpy as np

from app.utils.vector import cosine_similarity

logger = logging.getLogger(__name__)

_MIN_CORPUS_TABLES = 3
_SVD_DIM = 100
# 특이값 가중 지수 p (임베딩 = U·Σ^p). PPMI-SVD 는 p=0.5 가 p=1 보다 유사도 품질이 낫다
# (Levy, Goldberg & Dagan, 2015).
_SVD_EIGEN_WEIGHT = 0.5


def ppmi_matrix(counts: np.ndarray) -> np.ndarray:
    """출현 횟수 행렬(데이터셋 × 특징) → PPMI 행렬. 행·열 합은 0 이 아니어야 한다."""
    total = counts.sum()
    p_dataset = counts.sum(axis=1, keepdims=True) / total
    p_feature = counts.sum(axis=0, keepdims=True) / total
    with np.errstate(divide="ignore"):
        pmi = np.log(counts / total / (p_dataset * p_feature))
    return np.maximum(pmi, 0.0)  # 미출현 log(0) = -inf → 0


def svd_embeddings(ppmi: np.ndarray, dim: int = _SVD_DIM) -> np.ndarray:
    """PPMI 행렬의 절단 SVD 임베딩 U_k·Σ_k^p (행 = 데이터셋).

    ponytail: dense SVD — 데이터셋 수천·특징 수십만이면 scipy.sparse.linalg.svds 로.
    """
    u, s, _ = np.linalg.svd(ppmi, full_matrices=False)
    k = min(dim, int((s > 1e-10).sum()))
    return u[:, :k] * s[:k] ** _SVD_EIGEN_WEIGHT


class PPMIAnalyzer:
    """전체 데이터셋 코퍼스 기준 PPMI·SVD 연관성 수치 산출기 (인스턴스 = 실행 1회의 코퍼스 캐시)."""

    def __init__(self, opensearch_repo=None):
        """OpenSearch 저장소를 주입받아 초기화 (기본: 모듈 싱글턴)."""
        if opensearch_repo is None:
            from app.repositories.opensearch_repository import get_opensearch_repository
            opensearch_repo = get_opensearch_repository()
        self.opensearch_repo = opensearch_repo
        self._counts: dict[str, dict[str, int]] = {}
        self._embeddings: dict[str, np.ndarray] | None = None

    def load_tables(self, table_names: list[str]) -> None:
        """코퍼스 구성 — 테이블별 KV 특징 출현 횟수 1회 수집. 코퍼스가 바뀌면 SVD 를 다시 한다."""
        new = [name for name in table_names if name not in self._counts]
        for name in new:
            self._counts[name] = self.opensearch_repo.get_table_feature_counts(name)
        if new:
            self._embeddings = None
        loaded = sum(1 for c in self._counts.values() if c)
        logger.info(f"[PPMI] 코퍼스 구성: {loaded}/{len(self._counts)}개 데이터셋에 KV 특징 존재")

    def _corpus_embeddings(self) -> dict[str, np.ndarray]:
        """KV 특징을 가진 데이터셋 전체의 PPMI-SVD 임베딩."""
        if self._embeddings is None:
            corpus = {t: c for t, c in self._counts.items() if c}
            index = {f: i for i, f in enumerate(sorted({f for c in corpus.values() for f in c}))}
            counts = np.zeros((len(corpus), len(index)))
            for row, features in enumerate(corpus.values()):
                for f, n in features.items():
                    counts[row, index[f]] = n
            vectors = svd_embeddings(ppmi_matrix(counts))
            self._embeddings = dict(zip(corpus, vectors))
            logger.info(f"[PPMI] SVD 임베딩: {counts.shape[1]}개 특징 → {vectors.shape[1]}차원")
        return self._embeddings

    def score_pair(self, table_a: str, table_b: str) -> dict | None:
        """두 데이터셋의 PPMI-SVD 임베딩 코사인.

        Returns:
            {"ppmi_cosine": 0~1, "shared_features": 상위 공유 특징} —
            어느 한쪽이 KV 미구축이거나 코퍼스가 3개 미만이면 None (임베딩 신호만 사용)
        """
        self.load_tables([table_a, table_b])
        counts_a, counts_b = self._counts[table_a], self._counts[table_b]
        corpus_size = sum(1 for c in self._counts.values() if c)
        if not counts_a or not counts_b or corpus_size < _MIN_CORPUS_TABLES:
            return None

        embeddings = self._corpus_embeddings()
        cosine = cosine_similarity(embeddings[table_a].tolist(), embeddings[table_b].tolist())
        shared = sorted(
            (f for f in counts_a if f in counts_b),
            key=lambda f: min(counts_a[f], counts_b[f]),
            reverse=True,
        )
        result = {"ppmi_cosine": round(cosine, 4), "shared_features": shared[:10]}
        logger.debug(f"[PPMI] {table_a} <-> {table_b}: {result}")
        return result
