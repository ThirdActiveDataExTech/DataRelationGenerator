"""데이터셋 특징 동시출현 기반 PPMI·SVD 상호 연관성 분석.

  데이터셋-특징 출현 횟수 → 코퍼스 PPMI 행렬 → 절단 SVD 임베딩 → 코사인

특징 = 원장 프로필의 컬럼명("key:...")과 샘플 레코드 값("value:...") — 데이터셋을 노드,
특징을 이웃 노드로 보는 이분 그래프의 간선 가중치다. 코퍼스는 파이프라인 실행 대상 전체
데이터셋이며, 소수 데이터셋만 공유하는(희소한) 특징일수록 PPMI 가중치가 높아진다.
SVD 는 특징 공간을 최대 _SVD_DIM 차원으로 줄인다. 코퍼스의 유효 랭크가 _SVD_DIM 을 넘으면
하위 성분이 잘려, 특징을 직접 공유하지 않아도 같은 특징군과 함께 나타나는 데이터셋끼리
가까워진다. 임베딩(문장 의미) 신호와 상보적 — 설명 문구가 달라도 컬럼·값이 겹치면 잡아낸다.

주의: 코퍼스가 2개뿐이면 균등 분포 특징의 PMI 가 0으로 붕괴하므로, 특징을 가진
데이터셋이 3개 미만이면 None 을 반환한다 (임베딩 신호만 사용).
"""

import logging
import re
from collections import Counter

import numpy as np
from scipy import sparse
from sklearn.utils.extmath import randomized_svd

from app.schemas.dataset import DatasetProfile
from app.utils.vector import cosine_similarity

logger = logging.getLogger(__name__)

_MIN_CORPUS_TABLES = 3
_SVD_DIM = 100
# 특이값 가중 지수 p (임베딩 = U·Σ^p). PPMI-SVD 는 p=0.5 가 p=1 보다 유사도 품질이 낫다
# (Levy, Goldberg & Dagan, 2015).
_SVD_EIGEN_WEIGHT = 0.5
_MAX_VALUE_LEN = 50
_NUMBER = re.compile(r"[-+]?\d+(\.\d+)?")


def _value_feature(value) -> str | None:
    """샘플 값 중 데이터셋을 잇는 값(범주·명칭·코드)만 특징으로 쓴다.

    짧은 정수·소수는 수량 측정값이라 서로 무관한 데이터셋에서도 우연히 일치하므로 제외하고,
    5자리 이상 숫자열은 행정구역·기관 코드 같은 식별자로 보아 남긴다. 긴 서술문도 제외.
    """
    text = str(value).strip() if value is not None else ""
    if not text or len(text) > _MAX_VALUE_LEN:
        return None
    if _NUMBER.fullmatch(text) and not (text.isdigit() and len(text) >= 5):
        return None
    return f"value:{text}"


def feature_counts(profile: DatasetProfile) -> dict[str, int]:
    """프로필의 특징 출현 횟수 — 샘플 레코드마다 값이 있는 컬럼과 그 값을 1회씩 센다.

    샘플이 없는 컬럼은 1회로 센다 (컬럼 구성만 공개된 데이터셋도 코퍼스에 참여).
    """
    counts: Counter[str] = Counter()
    seen_columns = set()
    for record in profile.sample_records:
        for column, value in record.items():
            if value is None or str(value).strip() == "":
                continue
            key = f"key:{column.strip().lower().replace(' ', '')}"
            counts[key] += 1
            seen_columns.add(key)
            if feature := _value_feature(value):
                counts[feature] += 1
    for column in profile.column_names:
        key = f"key:{column.strip().lower().replace(' ', '')}"
        if key not in seen_columns:
            counts[key] += 1
    return dict(counts)


def ppmi_matrix(counts) -> sparse.csr_matrix:
    """출현 횟수 행렬(데이터셋 × 특징, dense 또는 sparse) → PPMI 희소 행렬.

    미출현 칸은 PMI = log(0) = -inf → 0 이므로 출현한 칸만 계산한다.
    """
    m = sparse.coo_matrix(counts, dtype=float)
    total = m.sum()
    row_sum = np.asarray(m.sum(axis=1)).ravel()
    col_sum = np.asarray(m.sum(axis=0)).ravel()
    pmi = np.log(m.data * total / (row_sum[m.row] * col_sum[m.col]))
    keep = pmi > 0
    return sparse.csr_matrix((pmi[keep], (m.row[keep], m.col[keep])), shape=m.shape)


def svd_embeddings(ppmi, dim: int = _SVD_DIM) -> np.ndarray:
    """PPMI 행렬의 절단 SVD 임베딩 U_k·Σ_k^p (행 = 데이터셋).

    작은 코퍼스는 정확한 SVD, 특징 공간이 dim 의 2배를 넘으면 무작위 절단 SVD(시드 고정)로
    상위 dim 성분만 구한다 — 데이터셋 수천·특징 수천이어도 희소 행렬 그대로 처리한다.
    """
    if min(ppmi.shape) <= 2 * dim:
        dense = ppmi.toarray() if sparse.issparse(ppmi) else ppmi
        u, s, _ = np.linalg.svd(dense, full_matrices=False)
    else:
        u, s, _ = randomized_svd(ppmi, n_components=dim, random_state=0)
    k = min(dim, int((s > 1e-10).sum()))
    return u[:, :k] * s[:k] ** _SVD_EIGEN_WEIGHT


class PPMIAnalyzer:
    """전체 데이터셋 코퍼스 기준 PPMI·SVD 연관성 수치 산출기 (인스턴스 = 실행 1회의 코퍼스 캐시)."""

    def __init__(self):
        self._counts: dict[str, dict[str, int]] = {}
        self._embeddings: dict[str, np.ndarray] | None = None

    def load_profiles(self, profiles: list[DatasetProfile]) -> None:
        """코퍼스 구성 — 프로필별 특징 출현 횟수 1회 집계."""
        self.load({p.table_name: feature_counts(p) for p in profiles if p.table_name not in self._counts})

    def load(self, counts_by_table: dict[str, dict[str, int]]) -> None:
        """테이블별 특징 출현 횟수를 코퍼스에 추가. 코퍼스가 바뀌면 SVD 를 다시 한다."""
        new = {t: c for t, c in counts_by_table.items() if t not in self._counts}
        self._counts.update(new)
        if new:
            self._embeddings = None
        loaded = sum(1 for c in self._counts.values() if c)
        logger.info(f"[PPMI] 코퍼스 구성: {loaded}/{len(self._counts)}개 데이터셋에 특징 존재")

    def _corpus_embeddings(self) -> dict[str, np.ndarray]:
        """특징을 가진 데이터셋 전체의 PPMI-SVD 임베딩.

        PPMI 는 전체 특징으로 계산해 주변 확률을 보존하고, SVD 에는 2개 이상 데이터셋이
        공유하는 특징 열만 넣는다 — 한 데이터셋에만 있는 특징은 데이터셋 간 연관성에 기여하지 않는다.
        """
        if self._embeddings is None:
            corpus = {t: c for t, c in self._counts.items() if c}
            index = {f: i for i, f in enumerate(sorted({f for c in corpus.values() for f in c}))}
            rows, cols, vals = [], [], []
            for row, features in enumerate(corpus.values()):
                for f, n in features.items():
                    rows.append(row)
                    cols.append(index[f])
                    vals.append(n)
            counts = sparse.csr_matrix((vals, (rows, cols)), shape=(len(corpus), len(index)), dtype=float)
            shared = np.flatnonzero(np.asarray((counts > 0).sum(axis=0)).ravel() >= 2)
            if shared.size:
                vectors = svd_embeddings(ppmi_matrix(counts)[:, shared])
            else:
                vectors = np.zeros((len(corpus), 1))
            self._embeddings = dict(zip(corpus, vectors))
            logger.info(
                f"[PPMI] SVD 임베딩: 특징 {len(index)}개 중 공유 {shared.size}개 → {vectors.shape[1]}차원"
            )
        return self._embeddings

    def score_pair(self, table_a: str, table_b: str) -> dict | None:
        """두 데이터셋의 PPMI-SVD 임베딩 코사인.

        Returns:
            {"ppmi_cosine": 0~1, "shared_features": 상위 공유 특징} —
            어느 한쪽이 특징이 없거나 코퍼스가 3개 미만이면 None (임베딩 신호만 사용)
        """
        counts_a, counts_b = self._counts.get(table_a), self._counts.get(table_b)
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
