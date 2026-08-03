"""KV 지식그래프 기반 PPMI 상호 연관성 분석 (계획서 ②-2 "상호 연관성 분석 모델").

계획서 파이프라인 대응:
  지식그래프 가중치(데이터셋-특징 출현 횟수) → 전체 코퍼스 PPMI 행렬 → 특징 벡터 코사인

특징 = KV 그래프의 Key 이름("key:...")과 Value 내용("value:...").
코퍼스는 파이프라인 실행 대상 전체 데이터셋 — 소수 데이터셋만 공유하는(희소한)
Key/Value 일수록 PPMI 가중치가 높아져, 그 특징을 공유하는 쌍의 연관성 수치가 커진다.
임베딩(문장 의미) 신호와 상보적 — 설명 문구가 달라도 실제 값이 겹치면 잡아낸다.

주의: 코퍼스가 2개뿐이면 균등 분포 특징의 PMI 가 0으로 붕괴하므로, KV 특징을 가진
데이터셋이 3개 미만이면 None 을 반환한다 (임베딩 신호만 사용).

ponytail: SVD 임베딩은 생략 — 현 규모에서 희소 PPMI 벡터 코사인으로 충분. 필요 시 SVD.
"""

import logging
import math

from app.utils.vector import sparse_cosine_similarity

logger = logging.getLogger(__name__)

_MIN_CORPUS_TABLES = 3


class PPMIAnalyzer:
    """전체 데이터셋 코퍼스 기준 PPMI 연관성 수치 산출기 (인스턴스 = 실행 1회의 코퍼스 캐시)."""

    def __init__(self, opensearch_repo=None):
        """OpenSearch 저장소를 주입받아 초기화 (기본: 모듈 싱글턴)."""
        if opensearch_repo is None:
            from app.repositories.opensearch_repository import get_opensearch_repository
            opensearch_repo = get_opensearch_repository()
        self.opensearch_repo = opensearch_repo
        self._counts: dict[str, dict[str, int]] = {}

    def load_tables(self, table_names: list[str]) -> None:
        """코퍼스 구성 — 테이블별 KV 특징 출현 횟수 1회 수집."""
        for name in table_names:
            if name not in self._counts:
                self._counts[name] = self.opensearch_repo.get_table_feature_counts(name)
        loaded = sum(1 for c in self._counts.values() if c)
        logger.info(f"[PPMI] 코퍼스 구성: {loaded}/{len(self._counts)}개 데이터셋에 KV 특징 존재")

    def score_pair(self, table_a: str, table_b: str) -> dict | None:
        """두 데이터셋의 PPMI 연관성 수치.

        Returns:
            {"ppmi_cosine": 0~1, "shared_features": 상위 공유 특징} —
            어느 한쪽이 KV 미구축이거나 코퍼스가 3개 미만이면 None (임베딩 신호만 사용)
        """
        self.load_tables([table_a, table_b])
        counts_a, counts_b = self._counts[table_a], self._counts[table_b]
        corpus = {t: c for t, c in self._counts.items() if c}
        if not counts_a or not counts_b or len(corpus) < _MIN_CORPUS_TABLES:
            return None

        total = sum(sum(c.values()) for c in corpus.values())
        feature_totals: dict[str, int] = {}
        for c in corpus.values():
            for f, n in c.items():
                feature_totals[f] = feature_totals.get(f, 0) + n

        def ppmi_vector(counts: dict[str, int]) -> dict[str, float]:
            p_d = sum(counts.values()) / total
            vec = {}
            for f, n in counts.items():
                joint = n / total
                p_f = feature_totals[f] / total
                vec[f] = max(0.0, math.log(joint / (p_d * p_f)))
            return vec

        cosine = sparse_cosine_similarity(ppmi_vector(counts_a), ppmi_vector(counts_b))
        shared = sorted(
            (f for f in counts_a if f in counts_b),
            key=lambda f: min(counts_a[f], counts_b[f]),
            reverse=True,
        )
        result = {"ppmi_cosine": round(cosine, 4), "shared_features": shared[:10]}
        logger.debug(f"[PPMI] {table_a} <-> {table_b}: {result}")
        return result

