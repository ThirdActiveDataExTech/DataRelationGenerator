"""3-Tier 테이블 간 관계 생성 파이프라인."""

import itertools
import json
import logging

from app.repositories.dataset_repository import DatasetRepository, get_dataset_repository
from app.schemas.dataset import (
    ColumnMatchInfo,
    DatasetProfile,
    DatasetRelation,
    Level1RelationType,
    RelationPipelineResult,
    Tier1Result,
    Tier2Candidate,
    Tier3Result,
    canonicalize_level1_type,
)
from app.src.dataset.dataset_service import DatasetService
from app.src.embedding.embedding_service import embedding_service
from app.src.meta_extract.llm_response import LLMResponse

logger = logging.getLogger(__name__)

_SAMPLE_DISPLAY = 3  # LLM에게 보여줄 샘플 레코드 수

# 단독으로 존재할 때 의미 없는 공통 컬럼명 (노이즈 필터)
_COLUMN_BLACKLIST = {"ID", "NAME", "CODE", "TYPE", "STATUS", "DATE", "FLAG", "VALUE", "NOTE", "MEMO"}


class RelationPipeline:
    """3-Tier 테이블 간 관계 생성 파이프라인.

    Tier 1 (rule): FK, 동일 컬럼명, 코드사전 자동 확정 → score=1.0
    Tier 2 (embedding): table_embedding 코사인 유사도 KNN → 후보 생성
    Tier 3 (llm): LLM 검증으로 FP 제거 → Neptune 저장
    """

    def __init__(
        self,
        dataset_repo: DatasetRepository | None = None,
        dataset_service: DatasetService | None = None,
    ):
        self.dataset_repo = dataset_repo or get_dataset_repository()
        self.dataset_service = dataset_service or DatasetService()

    # ──────────────────────────────────────────────
    # Entry point
    # ──────────────────────────────────────────────

    def run(
        self,
        table_names: list[str],
        tier2_threshold: float = 0.4,
        tier3_confidence_threshold: float = 0.7,
    ) -> list[RelationPipelineResult]:
        """모든 테이블 쌍에 대해 3-Tier 파이프라인 실행.

        Args:
            table_names: 대상 테이블명 목록
            tier2_threshold: Tier 2 유사도 임계값
            tier3_confidence_threshold: Tier 3 확신도 임계값

        Returns:
            테이블 쌍별 RelationPipelineResult 리스트
        """
        profiles = self._load_profiles(table_names)
        if len(profiles) < 2:
            logger.warning("관계 생성에는 2개 이상의 테이블이 필요합니다")
            return []

        existing_pairs = self.dataset_service.get_existing_relation_pairs()
        all_pairs = list(itertools.combinations(profiles, 2))

        new_pairs = [
            (a, b) for a, b in all_pairs
            if tuple(sorted([a.table_name, b.table_name])) not in existing_pairs
        ]

        logger.info(
            f"[Pipeline] 전체 {len(all_pairs)}쌍 중 신규 {len(new_pairs)}쌍 처리 "
            f"(기존 {len(all_pairs) - len(new_pairs)}쌍 스킵)"
        )

        results = []
        for profile_a, profile_b in new_pairs:
            result = self._run_pair(
                profile_a, profile_b, tier2_threshold, tier3_confidence_threshold
            )
            results.append(result)

        total_saved = sum(len(r.saved_relations) for r in results)
        logger.info(f"[Pipeline] 완료: {len(new_pairs)}쌍 처리, {total_saved}개 관계 저장")
        return results

    # ──────────────────────────────────────────────
    # Per-pair logic
    # ──────────────────────────────────────────────

    def _run_pair(
        self,
        a: DatasetProfile,
        b: DatasetProfile,
        tier2_threshold: float,
        tier3_confidence_threshold: float,
    ) -> RelationPipelineResult:
        result = RelationPipelineResult(table_a=a.table_name, table_b=b.table_name)

        # Tier 1
        tier1 = self._run_tier1(a, b)
        result.tier1 = tier1

        # Tier 2
        candidate = self._run_tier2(a, b)
        if candidate:
            result.tier2_candidates = [candidate]

        # Tier 1 후보 있거나 Tier 2 임계값 이상이면 LLM 판단으로 넘김
        has_tier1 = bool(tier1)
        has_tier2 = bool(candidate and candidate.score >= tier2_threshold)

        if not has_tier1 and not has_tier2:
            return result

        embedding_score = candidate.score if candidate else 1.0
        tier3 = self._run_tier3(a, b, embedding_score, tier3_confidence_threshold)

        if tier3:
            result.tier3_verified = [tier3]
            # 관계 유형을 Level 1 표준 3종(§7.4)으로 정규화 (원본 라벨은 하위 보존)
            canonical_type = canonicalize_level1_type(
                tier3.relation_type, has_common_columns=has_tier1
            )
            rel = DatasetRelation(
                table_a=tier3.table_a, table_b=tier3.table_b,
                score=tier3.confidence, method=tier3.method,
                relation_type=canonical_type,
            )
            self.dataset_service.save_relation(rel)
            result.saved_relations.append(rel)
            logger.info(
                f"[Pipeline] {a.table_name} <-> {b.table_name}: "
                f"관계 확정 ({tier3.relation_type}, confidence={tier3.confidence:.2f})"
            )
        else:
            logger.debug(f"[Pipeline] {a.table_name} <-> {b.table_name}: LLM 판단 — 관계 없음")

        return result

    # ──────────────────────────────────────────────
    # Tier 1 — 규칙 기반
    # ──────────────────────────────────────────────

    def _run_tier1(self, a: DatasetProfile, b: DatasetProfile) -> list[Tier1Result]:
        """FK 및 동일 컬럼명 자동 매칭 (블랙리스트 컬럼 단독 매칭 제외)."""
        cols_a = {c.upper() for c in a.column_names}
        cols_b = {c.upper() for c in b.column_names}
        common = (cols_a & cols_b) - _COLUMN_BLACKLIST

        if not common:
            return []

        matched = [
            ColumnMatchInfo(column_a=c, column_b=c, match_type="exact_name")
            for c in common
        ]
        relation_type = f"공통컬럼({', '.join(sorted(common)[:3])})"

        return [
            Tier1Result(
                table_a=a.table_name,
                table_b=b.table_name,
                relation_type=relation_type,
                matched_columns=matched,
            )
        ]

    # ──────────────────────────────────────────────
    # Tier 2 — 임베딩 유사도
    # ──────────────────────────────────────────────

    _DESC_WEIGHT = 0.6
    _KW_WEIGHT = 0.4

    def _run_tier2(self, a: DatasetProfile, b: DatasetProfile) -> Tier2Candidate | None:
        """description 임베딩 + keywords 평균 임베딩 결합 유사도 계산."""
        if not a.table_embedding or not b.table_embedding:
            logger.debug(f"[Tier2] {a.table_name} 또는 {b.table_name} 임베딩 없음, 스킵")
            return None

        desc_score = self._cosine(a.table_embedding, b.table_embedding)

        if a.keywords_embedding and b.keywords_embedding:
            kw_score = self._cosine(a.keywords_embedding, b.keywords_embedding)
            score = self._DESC_WEIGHT * desc_score + self._KW_WEIGHT * kw_score
        else:
            score = desc_score

        logger.debug(f"[Tier2] {a.table_name} <-> {b.table_name}: score={score:.3f}")
        return Tier2Candidate(table_a=a.table_name, table_b=b.table_name, score=score)

    @staticmethod
    def _cosine(v1: list[float], v2: list[float]) -> float:
        dot = sum(x * y for x, y in zip(v1, v2))
        norm1 = sum(x ** 2 for x in v1) ** 0.5
        norm2 = sum(x ** 2 for x in v2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return max(0.0, min(1.0, dot / (norm1 * norm2)))

    # ──────────────────────────────────────────────
    # Tier 3 — LLM 검증
    # ──────────────────────────────────────────────

    def _run_tier3(
        self,
        a: DatasetProfile,
        b: DatasetProfile,
        embedding_score: float,
        confidence_threshold: float,
    ) -> Tier3Result | None:
        """LLM으로 두 테이블 관계 검증."""
        try:
            variables = {
                "table_a": a.table_name,
                "domain_a": a.domain,
                "description_a": a.description,
                "summary_a": a.table_summary,
                "columns_a": ", ".join(a.column_names[:20]),
                "samples_a": json.dumps(a.sample_records[:_SAMPLE_DISPLAY], ensure_ascii=False),
                "table_b": b.table_name,
                "domain_b": b.domain,
                "description_b": b.description,
                "summary_b": b.table_summary,
                "columns_b": ", ".join(b.column_names[:20]),
                "samples_b": json.dumps(b.sample_records[:_SAMPLE_DISPLAY], ensure_ascii=False),
                "embedding_score": round(embedding_score, 3),
            }

            result = LLMResponse("table_relation_verification").generate_llm_response(
                variables=variables
            )

            if not isinstance(result, dict):
                logger.warning(f"[Tier3] LLM 응답이 dict가 아님: {type(result)}")
                return None

            if not result.get("related", False):
                logger.debug(f"[Tier3] {a.table_name} <-> {b.table_name}: 관계 없음")
                return None

            confidence = result.get("confidence", 0.0)
            if confidence < confidence_threshold:
                logger.debug(
                    f"[Tier3] {a.table_name} <-> {b.table_name}: "
                    f"confidence={confidence:.2f} < threshold={confidence_threshold}"
                )
                return None

            logger.info(
                f"[Tier3] {a.table_name} <-> {b.table_name}: "
                f"관계 확정 (type={result['relation_type']}, confidence={confidence:.2f})"
            )
            return Tier3Result(
                table_a=a.table_name,
                table_b=b.table_name,
                relation_type=result["relation_type"],
                confidence=confidence,
                reason=result.get("reason", ""),
                embedding_score=embedding_score,
            )

        except Exception as e:
            logger.error(f"[Tier3] {a.table_name} <-> {b.table_name} LLM 검증 실패: {e}")
            return None

    # ──────────────────────────────────────────────
    # Helper
    # ──────────────────────────────────────────────

    def _load_profiles(self, table_names: list[str]) -> list[DatasetProfile]:
        """OpenSearch에서 테이블 프로파일 로드."""
        profiles = []
        for name in table_names:
            profile = self.dataset_repo.get(name)
            if profile is None:
                logger.warning(f"[Pipeline] 테이블 '{name}' 프로파일 없음, 스킵")
                continue
            profiles.append(profile)
        return profiles


relation_pipeline = RelationPipeline()
