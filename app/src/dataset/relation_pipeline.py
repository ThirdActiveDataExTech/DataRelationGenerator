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
    column_jaccard,
    looks_aggregate,
)
from app.src.dataset.dataset_service import DatasetService
from app.src.embedding.embedding_service import embedding_service
from app.src.meta_extract.llm_response import LLMResponse
from app.utils.vector import cosine_similarity

logger = logging.getLogger(__name__)

_SAMPLE_DISPLAY = 3  # LLM에게 보여줄 샘플 레코드 수
_RULE_AUTO_APPROVE = 0.8  # 구조 근거(공통 컬럼)가 있는 쌍의 자동 확정 기준

# 단독으로 존재할 때 의미 없는 공통 컬럼명 (노이즈 필터)
_COLUMN_BLACKLIST = {"ID", "NAME", "CODE", "TYPE", "STATUS", "DATE", "FLAG", "VALUE", "NOTE", "MEMO"}



def _norm(text: str) -> str:
    return str(text).strip().lower().replace(" ", "")


def _column_in_both(a: DatasetProfile, b: DatasetProfile, column: str) -> bool:
    """인용된 컬럼이 양쪽에 실제로 있는지. 컬럼을 모르면 검증하지 않는다."""
    if not (a.column_names and b.column_names):
        return True
    target = _norm(column)
    return any(_norm(c) == target for c in a.column_names) and any(
        _norm(c) == target for c in b.column_names
    )


def _value_in_both(a: DatasetProfile, b: DatasetProfile, value: str) -> bool:
    """인용된 값이 양쪽 샘플에 실제로 등장하는지. 샘플을 모르면 검증하지 않는다."""
    if not (a.sample_records and b.sample_records):
        return True
    target = _norm(value)

    def appears(profile: DatasetProfile) -> bool:
        return any(
            target in _norm(cell)
            for record in profile.sample_records
            for cell in record.values()
            if cell is not None
        )

    return appears(a) and appears(b)


def _source_side_matches(matches: list[ColumnMatchInfo], source_is_a: bool) -> list[dict]:
    """컬럼 매칭을 column_a = source 쪽 컬럼이 되도록 맞춘다."""
    return [
        m.model_dump() if source_is_a else {**m.model_dump(), "column_a": m.column_b, "column_b": m.column_a}
        for m in matches
    ]


class RelationPipeline:
    """3-Tier 테이블 간 관계 생성 파이프라인.

    Tier 1 (rule): FK, 동일 컬럼명, 코드사전 자동 확정 → score=1.0
    Tier 2 (embedding): table_embedding 코사인 유사도 KNN → 후보 생성
    Tier 3 (llm): LLM 검증으로 FP 제거 → 원장(dcat_dataset_relation) 저장
    """

    def __init__(
        self,
        dataset_repo: DatasetRepository | None = None,
        dataset_service: DatasetService | None = None,
    ):
        """DatasetRepository와 DatasetService를 주입받아 초기화."""
        self.dataset_repo = dataset_repo or get_dataset_repository()
        self.dataset_service = dataset_service or DatasetService()

    # ──────────────────────────────────────────────
    # Entry point
    # ──────────────────────────────────────────────

    def run(
        self,
        table_names: list[str],
        tier2_threshold: float = 0.4,
        tier3_confidence_threshold: float = 0.6,
        refresh_table_names: list[str] | None = None,
        seed_table_names: list[str] | None = None,
    ) -> list[RelationPipelineResult]:
        """모든 테이블 쌍에 대해 관계 생성 파이프라인 실행 (증분: 기존 쌍 스킵).

        Args:
            table_names: 대상 테이블명 목록 (비어있으면 등록된 전체 테이블)
            tier2_threshold: Tier 2 유사도 임계값
            tier3_confidence_threshold: Tier 3 확신도 하한 — 이상이면 판정 대기, 자동 확정 기준 이상이면 확정
            refresh_table_names: 메타데이터가 변동된 테이블 — 이 테이블이 낀 쌍은
                기존 관계가 있어도 재평가 (save_relation 이 기존 관계를 대체)
            seed_table_names: 기준 테이블 — 지정 시 이 테이블이 낀 쌍만 평가.
                "1st 파티 데이터셋 기준 연계 가능한 3rd 파티 탐색" 시나리오용

        Returns:
            테이블 쌍별 RelationPipelineResult 리스트
        """
        if not table_names:
            table_names = [p.table_name for p in self.dataset_repo.list_all()]
        profiles = self._load_profiles(table_names)
        if len(profiles) < 2:
            logger.warning("관계 생성에는 2개 이상의 테이블이 필요합니다")
            return []

        # PPMI 코퍼스는 실행 1회 단위로 새로 구성 (전체 대상 테이블 기준)
        from app.config import settings as _settings

        self._ppmi_analyzer = None
        if _settings.PPMI_ENABLED:
            from app.src.dataset.ppmi_analyzer import PPMIAnalyzer

            self._ppmi_analyzer = PPMIAnalyzer()
            try:
                self._ppmi_analyzer.load_tables([p.table_name for p in profiles])
            except Exception as e:
                logger.warning(f"[Pipeline] PPMI 코퍼스 구성 실패 — 임베딩 신호만 사용: {e}")
                self._ppmi_analyzer = None

        existing_pairs = self.dataset_service.get_existing_relation_pairs()
        all_pairs = list(itertools.combinations(profiles, 2))

        # 기준(seed) 테이블 지정 시 해당 테이블이 낀 쌍만 평가 — 방향성 있는 연계 탐색
        seeds = set(seed_table_names or [])
        if seeds:
            all_pairs = [
                (a, b) for a, b in all_pairs
                if a.table_name in seeds or b.table_name in seeds
            ]

        refresh = set(refresh_table_names or [])

        new_pairs = [
            (a, b) for a, b in all_pairs
            if tuple(sorted([a.table_name, b.table_name])) not in existing_pairs
            or a.table_name in refresh
            or b.table_name in refresh
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

        # Tier 1 (규칙) / Tier 1.5 (① 컬럼 AI 매칭) / Tier 2 (임베딩) — 후보 탐지
        tier1 = self._run_tier1(a, b)
        result.tier1 = tier1

        # Tier1이 이미 컬럼을 찾았으면 AI 매칭은 생략 (비용 절약, 규칙 매칭이 더 확실)
        tier15 = self._run_tier15(a, b) if not tier1 else []
        result.tier15_matches = tier15

        candidate = self._run_tier2(a, b)
        if candidate:
            result.tier2_candidates = [candidate]

        # Tier 1/1.5 후보 있거나 Tier 2 임계값 이상이면 LLM 판단으로 넘김
        has_tier1 = bool(tier1)
        has_tier15 = bool(tier15)
        has_tier2 = bool(candidate and candidate.score >= tier2_threshold)

        if not has_tier1 and not has_tier15 and not has_tier2:
            return result

        # Tier 3 — 모든 후보는 LLM 검증을 반드시 거친다.
        # 저장 method 태그는 "검증 통과 방식"이 아니라 "후보 출처"를 나타낸다:
        # 공통 컬럼이 있으면 rule(Tier1), 없으면 embedding(Tier2).
        embedding_score = candidate.score if candidate else 1.0
        tier1_common_columns = (len(tier1[0].matched_columns) if tier1 else 0) + len(tier15)
        tier3 = self._run_tier3(
            a, b, embedding_score, tier3_confidence_threshold, tier1_common_columns
        )

        if tier3:
            result.tier3_verified = [tier3]
            origin_method = "rule" if has_tier1 else "embedding"
            # 관계 유형을 4종으로 정규화 (원본 라벨은 하위 보존)
            jaccard = column_jaccard(a.column_names, b.column_names)
            aggregate_a, aggregate_b = looks_aggregate(a.column_names), looks_aggregate(b.column_names)
            is_aggregate = aggregate_a or aggregate_b
            canonical_type = canonicalize_level1_type(
                tier3.relation_type,
                has_common_columns=has_tier1 or has_tier15,
                jaccard=jaccard,
                is_aggregate=is_aggregate,
                llm_type=tier3.level1_type,
            )
            # 역할을 주는 쪽이 source (정의서 방향). LLM 은 "A를 보는 사람에게 B가 무엇인지"로
            # 판정하므로 B 가 주는 쪽이고, 컬럼으로 A 만 집계표로 확인되면 A 가 집계(source)다.
            source_is_a = (
                canonical_type.startswith(Level1RelationType.STATISTIC_FOR.value)
                and aggregate_a and not aggregate_b
            )
            source, target = (a, b) if source_is_a else (b, a)
            rel = DatasetRelation(
                table_a=source.table_name, table_b=target.table_name,
                catalog_identifier_a=source.catalog_identifier,
                catalog_identifier_b=target.catalog_identifier,
                score=tier3.confidence, method=origin_method,
                relation_type=canonical_type,
                evidence={
                    "reason": tier3.reason,
                    "usage": tier3.usage,
                    "evidence_columns": tier3.evidence_columns,
                    "evidence_values": tier3.evidence_values,
                    "embedding_score": tier3.embedding_score,
                    "column_jaccard": round(jaccard, 3),
                    "is_aggregate": is_aggregate,
                    "llm_level1_type": tier3.level1_type,
                    "matched_columns": _source_side_matches(tier1[0].matched_columns if tier1 else [], source_is_a),
                    "ai_matched_columns": _source_side_matches(tier15, source_is_a),
                    "ppmi": (
                        {"score": candidate.ppmi_score, "shared_features": candidate.shared_features}
                        if candidate and candidate.ppmi_score is not None
                        else None
                    ),
                },
            )
            # 승인 게이팅 (정본 = 원장 dcat_dataset_relation): 자동 확정 기준 이상이면 확정,
            # 그 아래는 판정 대기 — 관리 화면 '관계 판정'에서 승인·반려한다.
            # 구조 근거(공통 컬럼)가 있는 쌍은 종전 판정 기준(0.8)부터 확정한다.
            from app.config import settings

            auto_approved = tier3.confidence >= (
                _RULE_AUTO_APPROVE if origin_method == "rule" else settings.RELATION_AUTO_APPROVE_THRESHOLD
            )
            if auto_approved:
                self.dataset_service.save_relation(rel)
                result.saved_relations.append(rel)
            else:
                self.dataset_service.propose_relation_to_catalog(rel)
                result.pending_relations.append(rel)
            logger.info(
                f"[Pipeline] {a.table_name} <-> {b.table_name}: "
                f"관계 {'확정' if auto_approved else '승인 대기'} "
                f"(method={origin_method}, {tier3.relation_type}, "
                f"confidence={tier3.confidence:.2f})"
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
    # Tier 1.5 — ① 컬럼 연관성 분석 서비스 (AI 매칭)
    # ──────────────────────────────────────────────

    def _run_tier15(self, a: DatasetProfile, b: DatasetProfile) -> list[ColumnMatchInfo]:
        """이름이 달라 Tier1 규칙이 놓치는 컬럼 쌍을 ①의 AI 스키마 매칭 서비스로 탐지.

        (예: hr ↔ heart_rate, user_ref ↔ patient_id)
        비활성(CORRELATION_SERVICE_ENABLED=False)/MinIO 미사용/호출 실패 시 빈 목록 (graceful).
        """
        from app.config import settings

        if not settings.CORRELATION_SERVICE_ENABLED:
            return []
        try:
            from app.clients.correlation_client import get_correlation_client, upload_sample_csv

            l_uri, r_uri = upload_sample_csv(a), upload_sample_csv(b)
            if not (l_uri and r_uri):
                logger.debug("[Tier1.5] 샘플 CSV 업로드 불가 (MinIO 비활성) — 생략")
                return []
            matches = get_correlation_client().match_columns(l_uri, r_uri)
            if matches:
                logger.info(
                    f"[Tier1.5] {a.table_name} <-> {b.table_name}: AI 컬럼 매칭 {len(matches)}건 "
                    f"({', '.join(f'{m.column_a}~{m.column_b}' for m in matches[:3])}...)"
                )
            return matches
        except Exception as e:
            logger.warning(f"[Tier1.5] 컬럼 매칭 서비스 호출 실패 — 생략: {e}")
            return []

    # ──────────────────────────────────────────────
    # Tier 2 — 임베딩 유사도
    # ──────────────────────────────────────────────

    _DESC_WEIGHT = 0.6
    _KW_WEIGHT = 0.4

    def _run_tier2(self, a: DatasetProfile, b: DatasetProfile) -> Tier2Candidate | None:
        """Description 임베딩 + keywords 평균 임베딩 결합 유사도 계산."""
        if not a.table_embedding or not b.table_embedding:
            logger.debug(f"[Tier2] {a.table_name} 또는 {b.table_name} 임베딩 없음, 스킵")
            return None

        desc_score = cosine_similarity(a.table_embedding, b.table_embedding)

        if a.keywords_embedding and b.keywords_embedding:
            kw_score = cosine_similarity(a.keywords_embedding, b.keywords_embedding)
            score = self._DESC_WEIGHT * desc_score + self._KW_WEIGHT * kw_score
        else:
            score = desc_score

        # PPMI-SVD 보조 신호 — KV 그래프 특징 co-occurrence 기반 연관성 수치.
        # 설명 문구가 달라도 실제 Key/Value 가 겹치면 잡아내는, 임베딩과 상보적인 신호.
        from app.config import settings

        embedding_score, ppmi_score, shared_features = score, None, []
        if settings.PPMI_ENABLED:
            ppmi = self._run_ppmi(a.table_name, b.table_name)
            if ppmi is not None:
                ppmi_score = ppmi["ppmi_cosine"]
                shared_features = ppmi["shared_features"]
                score = (1 - settings.PPMI_WEIGHT) * score + settings.PPMI_WEIGHT * ppmi_score

        logger.debug(
            f"[Tier2] {a.table_name} <-> {b.table_name}: score={score:.3f} "
            f"(embedding={embedding_score:.3f}, ppmi={ppmi_score})"
        )
        return Tier2Candidate(
            table_a=a.table_name, table_b=b.table_name, score=score,
            embedding_score=embedding_score, ppmi_score=ppmi_score,
            shared_features=shared_features,
        )

    def _run_ppmi(self, table_a: str, table_b: str) -> dict | None:
        """KV 그래프 PPMI 연관성 수치 (graceful — 특징 없음/코퍼스 부족/실패 시 None).

        run() 이 구성한 실행 단위 코퍼스를 사용하고, 단독 호출 시엔 지연 생성한다.
        """
        try:
            analyzer = getattr(self, "_ppmi_analyzer", None)
            if analyzer is None:
                from app.src.dataset.ppmi_analyzer import PPMIAnalyzer

                analyzer = self._ppmi_analyzer = PPMIAnalyzer()
            return analyzer.score_pair(table_a, table_b)
        except Exception as e:
            logger.warning(f"[Tier2] PPMI 분석 실패 — 임베딩 점수만 사용: {e}")
            return None

    # ──────────────────────────────────────────────
    # Tier 3 — LLM 검증
    # ──────────────────────────────────────────────

    def _run_tier3(
        self,
        a: DatasetProfile,
        b: DatasetProfile,
        embedding_score: float,
        confidence_threshold: float,
        tier1_common_columns: int = 0,
    ) -> Tier3Result | None:
        """LLM으로 두 테이블 관계 검증 (2회 판정, 불일치 시 3회차 다수결).

        LLM 판정은 경계 사례에서 실행마다 뒤집힐 수 있어, 저장 여부가 일치할 때까지
        최대 3회 판정해 다수결로 확정한다.
        """
        try:
            verdicts: list[dict | None] = []
            for _ in range(2):
                verdicts.append(self._judge_once(a, b, embedding_score))
            passed = [
                v for v in verdicts
                if v and self._passes_filters(a, b, v, confidence_threshold, tier1_common_columns)
            ]
            if len(passed) == 1:  # 1:1 불일치 → tiebreak
                v3 = self._judge_once(a, b, embedding_score)
                if v3 and self._passes_filters(a, b, v3, confidence_threshold, tier1_common_columns):
                    passed.append(v3)
                else:
                    logger.info(f"[Tier3] {a.table_name} <-> {b.table_name}: 다수결 기각 (1/3)")
                    return None

            if not passed:
                # 기각도 남긴다 — 관계가 안 생긴 이유를 사후에 알 수 있어야 한다
                judged = [
                    None if v is None else (v.get("relation_type"), v.get("confidence"))
                    for v in verdicts
                ]
                logger.info(
                    f"[Tier3] {a.table_name} <-> {b.table_name}: 전원 기각 "
                    f"(임계 {confidence_threshold}, 판정={judged})"
                )
                return None

            result = passed[0]
            confidence = result.get("confidence", 0.0)
            cited = (result.get("evidence_columns") or []) + (result.get("evidence_values") or [])
            logger.info(
                f"[Tier3] {a.table_name} <-> {b.table_name}: "
                f"관계 확정 (type={result.get('level1_type')}, 근거인용={cited}, "
                f"confidence={confidence:.2f}, votes={len(passed)}/{len(verdicts) if len(passed) != 2 else 2})"
            )
            return Tier3Result(
                table_a=a.table_name,
                table_b=b.table_name,
                relation_type=result["relation_type"],
                level1_type=result.get("level1_type", ""),
                evidence_columns=result.get("evidence_columns") or [],
                evidence_values=result.get("evidence_values") or [],
                confidence=confidence,
                reason=result.get("reason", ""),
                usage=result.get("usage", ""),
                embedding_score=embedding_score,
            )

        except Exception as e:
            logger.error(f"[Tier3] {a.table_name} <-> {b.table_name} LLM 검증 실패: {e}")
            return None

    def _judge_once(self, a: DatasetProfile, b: DatasetProfile, embedding_score: float) -> dict | None:
        """LLM 관계 판정 1회 호출."""
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
        result = LLMResponse("table_relation_verification").generate_llm_response(variables=variables)
        if not isinstance(result, dict):
            logger.warning(f"[Tier3] LLM 응답이 dict가 아님: {type(result)}")
            return None
        return result

    def _passes_filters(
        self,
        a: DatasetProfile,
        b: DatasetProfile,
        result: dict,
        confidence_threshold: float,
        tier1_common_columns: int,
    ) -> bool:
        """판정 1건이 저장 조건을 만족하는지 검사.

        근거는 LLM이 인용한 컬럼·값을 실제 프로파일에 대조해 검증한다.
        인용이 틀렸으면(환각) 기각하고, 인용이 아예 없으면 구조 근거가 필요한
        유형(DETAIL_FOR/STATISTIC_FOR/SAME_KIND)은 기각한다. ADJACENT_TO 는
        구조 근거가 없는 것이 정의이므로 인용을 요구하지 않는다.
        """
        pair = f"{a.table_name} <-> {b.table_name}"

        if not result.get("related", False):
            logger.debug(f"[Tier3] {pair}: 관계 없음")
            return False

        level1_type = result.get("level1_type", "")
        cited_columns = [c for c in (result.get("evidence_columns") or []) if str(c).strip()]
        cited_values = [v for v in (result.get("evidence_values") or []) if str(v).strip()]

        bad_columns = [c for c in cited_columns if not _column_in_both(a, b, c)]
        if bad_columns:
            logger.info(f"[Tier3] {pair}: 인용 컬럼이 양쪽에 없음 {bad_columns} (환각), 저장 제외")
            return False

        bad_values = [v for v in cited_values if not _value_in_both(a, b, v)]
        if bad_values:
            logger.info(f"[Tier3] {pair}: 인용 값이 양쪽 샘플에 없음 {bad_values} (환각), 저장 제외")
            return False

        # 인용 없이도 구조 근거가 확인되면 통과 (Tier1/Tier1.5 공통 컬럼)
        has_evidence = bool(cited_columns or cited_values or tier1_common_columns)
        verifiable = bool(a.column_names and b.column_names)
        if not has_evidence and level1_type != Level1RelationType.ADJACENT_TO.value:
            if verifiable:
                logger.info(f"[Tier3] {pair}: {level1_type} 인데 데이터 근거 인용 없음, 저장 제외")
                return False
            # 메타데이터-only 는 인용할 컬럼·샘플 자체가 없다 — 요구하지 않는다
            logger.info(f"[Tier3] {pair}: 메타데이터-only — 근거 인용 검증 불가, 통과")

        # 지식·콘텐츠성 데이터끼리의 값·주제 중첩은 관계로 보지 않음.
        # 예외: 동종 병렬(SAME_KIND)이거나 Tier1 공통 컬럼이 2개 이상
        natures = (result.get("table_a_nature"), result.get("table_b_nature"))
        is_parallel = (
            level1_type == Level1RelationType.SAME_KIND.value or tier1_common_columns >= 2
        )
        if natures == ("knowledge_content", "knowledge_content") and not is_parallel:
            logger.info(f"[Tier3] {pair}: 콘텐츠성 데이터 간 주제 중첩, 저장 제외")
            return False

        confidence = result.get("confidence", 0.0)
        if confidence < confidence_threshold:
            logger.debug(f"[Tier3] {pair}: confidence={confidence:.2f} < threshold={confidence_threshold}")
            return False

        return True

    # ──────────────────────────────────────────────
    # Helper
    # ──────────────────────────────────────────────

    def _load_profiles(self, table_names: list[str]) -> list[DatasetProfile]:
        """원장에서 테이블 프로파일 로드."""
        profiles = []
        for name in table_names:
            profile = self.dataset_repo.get(name)
            if profile is None:
                logger.warning(f"[Pipeline] 테이블 '{name}' 프로파일 없음, 스킵")
                continue
            profiles.append(profile)
        return profiles


relation_pipeline = RelationPipeline()
