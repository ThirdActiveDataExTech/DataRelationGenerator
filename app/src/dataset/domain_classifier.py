"""데이터 특징 기반 도메인 자동 분류 (계획서 ②-1 "데이터 특징 기반 분류 기술").

도메인의 목적은 데이터셋을 같은 분야로 **묶는 것**이다. 따라서 소스마다 제각각인
어휘(카탈로그 theme 문자열, 분류코드 등)를 라벨로 쓰지 않고, 항상 이 분류기가
표준 어휘로 수렴시킨다.

특징 추출: 요약 텍스트 임베딩 — 데이터 의미 특징의 저차원 표현 (차원축소 대응)
분류 후보 두 층위:
  1. 표준 도메인 분류체계(STANDARD_DOMAINS, 공공데이터포털 국가 분류 기반) 라벨
     임베딩 — zero-shot. 등록 데이터가 없어도(콜드스타트) 표준 도메인으로 분류된다.
  2. 등록 데이터셋들의 도메인별 임베딩 중심점(centroid) — 사용자 지정 도메인 등
     분류체계 밖 도메인도 데이터가 쌓이면 후보가 된다.

- 최고 유사도가 DOMAIN_CLASSIFY_THRESHOLD 미만이면 "general" 폴백
- "general"(미분류) 데이터셋은 centroid 후보에서 제외 — 폴백이 스스로를 강화하지 않게
- 재등록(갱신) 시 자기 자신의 기존 문서는 centroid 에서 제외 — 이전 라벨이
  자기 자신과 매칭되는 순환 방지 (exclude_table_name)
"""

import logging

from app.config import settings
from app.utils.vector import cosine_similarity, mean_vector

logger = logging.getLogger(__name__)

FALLBACK_DOMAIN = "general"

# 공공데이터포털 국가데이터 분류체계 기반 표준 도메인.
# 값은 라벨 의미를 임베딩하기 위한 키워드 설명문 — 분류 결과 라벨은 key 를 사용한다.
STANDARD_DOMAINS: dict[str, str] = {
    "보건의료": "보건, 의료, 병원, 약국, 질병, 진료, 건강보험, 요양기관, 응급의료, 의약품",
    "사회복지": "복지, 노인, 아동, 장애인, 요양시설, 보육, 기초생활보장, 돌봄, 연금",
    "교통물류": "교통, 도로, 철도, 버스, 물류, 화물, 운송, 항공, 주차, 자동차",
    "농축수산": "농업, 축산, 수산, 임업, 작물, 곡물, 농산물, 가축, 어획, 양식",
    "환경기상": "환경, 대기, 수질, 폐기물, 기상, 기후, 미세먼지, 재활용, 오염",
    "재난안전": "재난, 안전, 소방, 경보, 지진, 홍수, 사고, 예방, 대피, 구조",
    "교육": "학교, 학생, 교육과정, 입시, 도서관, 평생교육, 학원, 강좌",
    "문화관광": "문화, 관광, 공연, 축제, 문화재, 박물관, 방송, 미디어, 콘텐츠, 체육",
    "산업고용": "산업, 기업, 고용, 일자리, 창업, 공장, 제조, 노동, 에너지, 발전",
    "재정금융": "재정, 세금, 예산, 금융, 물가, 가격, 경제지표, 보조금, 조달",
    "국토관리": "국토, 도시계획, 건축, 부동산, 토지, 주택, 지적, 상하수도",
    "공공행정": "행정, 민원, 인허가, 조직, 법령, 정책, 선거, 통계",
    "과학기술": "과학, 기술, 연구개발, 특허, 정보통신, 데이터, 인공지능, 소프트웨어",
    "식품건강": "식품, 위생, 영양, 급식, 외식, 식중독, 원산지, 축산물 검사",
    "법률": "법률, 판례, 소송, 범죄, 교정, 법원, 검찰, 계약",
    "통일외교안보": "통일, 외교, 안보, 국방, 북한, 출입국, 재외국민, 병무",
}


class DomainClassifier:
    """표준 분류체계 + 도메인별 centroid 기반 분류기 (모델 파일 없음)."""

    # 표준 라벨 임베딩 프로세스 캐시 (라벨은 불변)
    _label_embeddings: dict[str, list[float]] | None = None

    def __init__(self, dataset_repo):
        """도메인별 임베딩 조회에 사용할 DatasetRepository 를 주입받는다."""
        self.dataset_repo = dataset_repo

    @classmethod
    def _taxonomy_embeddings(cls) -> dict[str, list[float]]:
        if cls._label_embeddings is None:
            from app.src.embedding.embedding_service import embedding_service

            cls._label_embeddings = {
                label: embedding_service.generate_embedding(f"{label}: {desc}")
                for label, desc in STANDARD_DOMAINS.items()
            }
        return cls._label_embeddings

    def classify(
        self, embedding: list[float], exclude_table_name: str = ""
    ) -> tuple[str, float, dict[str, float]]:
        """특징 임베딩을 도메인으로 분류.

        Args:
            embedding: 분류할 특징 임베딩 (table_summary 임베딩)
            exclude_table_name: 재등록 중인 데이터셋명 — 자기 자신의 기존 문서를
                centroid 후보에서 제외

        Returns:
            (도메인, 신뢰도(최고 코사인 유사도), 도메인별 유사도) —
            분류 불가(임베딩 없음/후보 없음)면 ("general", 0.0, {})
        """
        if not embedding:
            return FALLBACK_DOMAIN, 0.0, {}

        # 1) 등록 데이터셋 도메인 중심점 (자기 자신·general 제외)
        groups: dict[str, list[list[float]]] = {}
        for table_name, domain, emb in (self.dataset_repo.list_domain_embeddings() or []):
            if exclude_table_name and table_name == exclude_table_name:
                continue
            if domain and domain != FALLBACK_DOMAIN and emb and len(emb) == len(embedding):
                groups.setdefault(domain, []).append(emb)

        scores: dict[str, float] = {
            domain: round(cosine_similarity(embedding, mean_vector(embs)), 4)
            for domain, embs in groups.items()
        }

        # 2) 표준 분류체계 라벨 (zero-shot) — 소스별 어휘와 무관하게 표준 도메인으로 수렴
        for label, label_emb in self._taxonomy_embeddings().items():
            if not label_emb or len(label_emb) != len(embedding):
                continue
            score = round(cosine_similarity(embedding, label_emb), 4)
            if score > scores.get(label, -1.0):
                scores[label] = score

        if not scores:
            return FALLBACK_DOMAIN, 0.0, {}

        best_domain, best_score = max(scores.items(), key=lambda kv: kv[1])
        if best_score < settings.DOMAIN_CLASSIFY_THRESHOLD:
            return FALLBACK_DOMAIN, best_score, scores
        return best_domain, best_score, scores
