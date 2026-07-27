"""데이터 특징 기반 도메인 자동 분류 (계획서 ②-1 "데이터 특징 기반 분류 기술").

특징 추출: 요약 텍스트 임베딩 — 데이터 의미 특징의 저차원 표현 (차원축소 대응)
분류: 등록 데이터셋들의 도메인별 임베딩 중심점(centroid)과의 코사인 유사도 (상관분석 대응)

- 최고 유사도가 DOMAIN_CLASSIFY_THRESHOLD 미만이면 "general" 폴백
- "general"(미분류) 데이터셋은 centroid 후보에서 제외 — 폴백이 스스로를 강화하지 않게
- 부트스트랩: 초기 시드 데이터셋 몇 건은 도메인을 명시해 등록해야 분류가 동작 (few-shot)
"""

import logging

from app.config import settings
from app.utils.vector import cosine_similarity, mean_vector

logger = logging.getLogger(__name__)

FALLBACK_DOMAIN = "general"


class DomainClassifier:
    """도메인별 centroid 기반 분류기 (모델 파일 없음 — 등록 데이터가 곧 학습 데이터)."""

    def __init__(self, dataset_repo):
        self.dataset_repo = dataset_repo

    def classify(self, embedding: list[float]) -> tuple[str, float, dict[str, float]]:
        """특징 임베딩을 도메인으로 분류.

        Returns:
            (도메인, 신뢰도(최고 코사인 유사도), 도메인별 유사도) —
            분류 불가(임베딩 없음/시드 없음)면 ("general", 0.0, {})
        """
        if not embedding:
            return FALLBACK_DOMAIN, 0.0, {}

        groups: dict[str, list[list[float]]] = {}
        for domain, emb in (self.dataset_repo.list_domain_embeddings() or []):
            if domain and domain != FALLBACK_DOMAIN and emb and len(emb) == len(embedding):
                groups.setdefault(domain, []).append(emb)
        if not groups:
            return FALLBACK_DOMAIN, 0.0, {}

        scores: dict[str, float] = {}
        for domain, embs in groups.items():
            scores[domain] = round(cosine_similarity(embedding, mean_vector(embs)), 4)

        best_domain, best_score = max(scores.items(), key=lambda kv: kv[1])
        if best_score < settings.DOMAIN_CLASSIFY_THRESHOLD:
            return FALLBACK_DOMAIN, best_score, scores
        return best_domain, best_score, scores
