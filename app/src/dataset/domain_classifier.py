"""데이터 특징 기반 분야 분류 .

분류의 목적은 데이터셋을 같은 분야로 **묶는 것**(분야별 탐색·필터)이므로 소스마다 다른
어휘(카탈로그 theme, 기관 분류코드)를 라벨로 쓰지 않고 항상 이 분류표로 수렴시킨다.

- 데이터셋 하나에 '대분류 > 중분류'를 1~3개 붙인다.
  첫 번째가 대표 분류이며 그 대분류가 DatasetProfile.domain 이 된다.
- 1차: LLM 이 분류표 전체를 보고 그 안에서만 고른다. 분류표 밖 값은 버린다.
- 폴백: LLM 이 실패하면 요약 임베딩과 중분류 라벨 임베딩의 코사인이 가장 큰 중분류 하나.
"""

import logging

from openai import APIError

from app.exceptions.base import ApplicationError
from app.utils.vector import cosine_similarity

logger = logging.getLogger(__name__)

TAXONOMY: dict[str, tuple[str, ...]] = {
    "과학": ("수학", "통계학", "물리학", "천문학", "화학", "생물학", "지구과학", "지질학", "대기과학", "생활과학"),
    "기술": (
        "기계공학", "재료공학", "화학공학", "전기·전자", "정보·통신", "인공지능·컴퓨팅", "에너지·자원",
        "원자력", "환경공학", "토목·건설·교통", "도시공학", "항공우주공학", "조선·해양",
    ),
    "인문": ("문학", "언어", "철학", "종교", "심리", "역사", "지리"),
    "사회": (
        "공공질서·안전", "복지", "사회문제", "인권", "국토·지역개발", "일반공공행정", "외교", "안보국방",
        "경제·경영", "산업", "정치",
    ),
    "교육": ("유아", "초등학교", "중학교", "고등학교", "고등(대학)", "해외유학", "특수교육", "직업·평생", "자격", "교양"),
    "의료": (
        "의학", "치의학", "한의학", "약학", "간호학", "공중보건", "질병관리", "건강검진", "재활", "요양",
        "의료공학", "의약품",
    ),
    "문화": (
        "미술", "방송·미디어", "공예", "게임·오락", "축제·전시", "연극", "무용", "영상·영화", "음악",
        "디자인", "사진", "만화", "스포츠",
    ),
} # 분류 체계는 국가지식정보 통합플랫폼 「디지털 집현전」의 주제 분류를 따른다.

SEPARATOR = " > "
CATEGORIES: tuple[str, ...] = tuple(f"{major}{SEPARATOR}{minor}" for major, minors in TAXONOMY.items() for minor in minors)
MAX_CATEGORIES = 3


def major_of(category: str) -> str:
    """'의료 > 공중보건' → '의료'."""
    return category.split(SEPARATOR, 1)[0]


def normalize_categories(values: list[str]) -> list[str]:
    """분류표에 있는 값만 순서대로, 중복 없이 최대 3개."""
    valid = [v.strip() for v in values if v and v.strip() in CATEGORIES]
    return list(dict.fromkeys(valid))[:MAX_CATEGORIES]


def taxonomy_prompt_text() -> str:
    """프롬프트에 넣을 분류표 ('대분류: 중분류, 중분류 ...' 한 줄씩)."""
    return "\n".join(f"- {major}: {', '.join(minors)}" for major, minors in TAXONOMY.items())


class CategoryClassifier:
    """디지털 집현전 분류표 기반 분류기 (LLM 판정 + 임베딩 폴백)."""

    # 중분류 라벨 임베딩 프로세스 캐시 (분류표는 불변)
    _label_embeddings: dict[str, list[float]] | None = None

    @classmethod
    def _category_embeddings(cls) -> dict[str, list[float]]:
        if cls._label_embeddings is None:
            from app.src.embedding.embedding_service import embedding_service

            cls._label_embeddings = {c: embedding_service.generate_embedding(c) for c in CATEGORIES}
        return cls._label_embeddings

    def nearest(self, embedding: list[float]) -> str:
        """특징 임베딩과 가장 가까운 중분류 (폴백)."""
        scores = {
            category: cosine_similarity(embedding, label)
            for category, label in self._category_embeddings().items()
            if label and len(label) == len(embedding)
        }
        return max(scores, key=scores.__getitem__)

    def classify(
        self, title: str, summary: str, keywords: list[str], columns: list[str], embedding: list[float]
    ) -> list[str]:
        """데이터셋을 '대분류 > 중분류' 1~3개로 분류한다. 대표 분류가 맨 앞.

        Returns:
            분류 목록. LLM·임베딩 모두 쓸 수 없으면 빈 목록 (미분류)
        """
        from app.src.meta_extract.llm_response import LLMResponse

        try:
            result = LLMResponse("dataset_category_classification").generate_llm_response({
                "title": title,
                "summary": summary,
                "keywords": ", ".join(keywords[:15]),
                "columns": ", ".join(columns[:40]),
                "taxonomy": taxonomy_prompt_text(),
            })
            picked = normalize_categories(result.get("categories") or [])
            if picked:
                return picked
            logger.info(f"[Category] '{title}' LLM 응답에 분류표 값 없음 — 임베딩 폴백: {result}")
        except (APIError, ApplicationError) as e:
            logger.warning(f"[Category] '{title}' LLM 분류 실패 — 임베딩 폴백: {e}")

        return [self.nearest(embedding)] if embedding else []
