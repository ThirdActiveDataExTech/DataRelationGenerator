"""Dataset 메타데이터 추출 및 원장(PostgreSQL) 적재 서비스."""

import json
import logging
from datetime import datetime, timezone
from typing import Any

from app.config import settings
from app.repositories.dataset_repository import DatasetRepository, get_dataset_repository
from app.repositories.relation_repository import RelationRepository, get_relation_repository
from app.schemas.dataset import DatasetProfile, DatasetRelation, DatasetSearchResult, type_seen_from
from app.src.dataset.domain_classifier import CategoryClassifier, major_of, normalize_categories
from app.src.embedding.embedding_service import embedding_service
from app.src.meta_extract.llm_response import LLMResponse

logger = logging.getLogger(__name__)

_SAMPLE_SIZE = 10


class DatasetService:
    """테이블 메타데이터 추출 → 데이터셋 원장 적재, 데이터셋 간 관계 기록."""

    def __init__(
        self,
        dataset_repo: DatasetRepository | None = None,
        relation_repo: RelationRepository | None = None,
    ):
        """데이터셋·관계 저장소를 주입받아 초기화."""
        self.dataset_repo = dataset_repo or get_dataset_repository()
        self.relation_repo = relation_repo or get_relation_repository()

    # ──────────────────────────────────────────────
    # 테이블 등록
    # ──────────────────────────────────────────────

    def ingest_table(
        self,
        records: list[dict[str, Any]],
        table_name: str,
        domain: str = "",
        description: str = "",
        table_summary: str = "",
        raw_storage_key: str = "",
        catalog_identifier: str = "",
        keywords: list[str] | None = None,
        party: str = "",
        provider: str = "",
        landing_page: str = "",
        column_names: list[str] | None = None,
    ) -> DatasetProfile:
        """테이블 데이터 → 메타데이터 추출 → 원장 저장.

        Args:
            records: 테이블의 전체 레코드 목록. 빈 목록이면 메타데이터-only 등록
                (카탈로그 pull 등 요약 텍스트가 있는 경우)
            table_name: 테이블명 (표시용)
            domain: 도메인 (medical, broadcast 등). 빈 값이면 특징 기반 자동 분류
            description: 짧은 설명
            table_summary: 상세 요약 (비어있으면 description 사용)
            raw_storage_key: 원본 파일 보관 키
            catalog_identifier: 통합 카탈로그 식별자 (① catalog_entry.identifier)
            keywords: 시드 키워드 (카탈로그 keyword/theme 등). 주어지면 LLM 추출 생략
            party: 데이터 파티 구분 (1st: 기관 보유, 3rd: 외부 개방)
            provider: 제공 기관/발행처 (DCAT publisher)
            landing_page: 원본 제공처 URL (수집 단계 연결용)
            column_names: 레코드 없이 스키마만 공개된 데이터의 컬럼 목록. records 가 있으면 무시

        Returns:
            저장된 DatasetProfile
        """
        if not records and not (table_summary or description):
            raise ValueError(f"'{table_name}': 레코드가 없고 요약 메타데이터도 없습니다")

        profile = self._build_profile(
            records, table_name, domain, description, table_summary,
            seed_keywords=keywords, column_names=column_names,
        )
        profile.raw_storage_key = raw_storage_key
        profile.catalog_identifier = catalog_identifier
        profile.party = party  # type: ignore[assignment]
        profile.provider = provider
        profile.landing_page = landing_page

        # 동일 table_name 재등록은 신규 생성이 아닌 기존 문서 대체 (중복 데이터셋 방지)
        existing = self.dataset_repo.get(table_name)
        if existing:
            profile.dataset_id = existing.dataset_id
            profile.created_at = existing.created_at

        self.dataset_repo.upsert(profile)

        logger.info(
            f"[Dataset] '{table_name}' 등록 완료: "
            f"{profile.record_count}건, {profile.column_count}개 컬럼, "
            f"embedding_dim={len(profile.table_embedding)}"
        )
        return profile

    # ──────────────────────────────────────────────
    # 관계 저장
    # ──────────────────────────────────────────────

    def save_relation(self, relation: DatasetRelation, push_to_catalog: bool = True) -> None:
        """두 Dataset 간 관계를 확정(APPROVED)으로 기록 — 같은 쌍의 기존 관계는 대체.

        Args:
            relation: DatasetRelation 객체
            push_to_catalog: ① 환류 여부 (① 승인 콜백 경로에서는 False — 재환류 방지)
        """
        self.relation_repo.replace(relation, "APPROVED")
        logger.info(
            f"[Relation] {relation.table_a} --[{relation.relation_type}]--> {relation.table_b} "
            f"(score={relation.score}, method={relation.method})"
        )

        if push_to_catalog:
            self._push_relation_to_catalog(relation, status="APPROVED")

    def propose_relation_to_catalog(self, relation: DatasetRelation) -> None:
        """저신뢰 관계를 승인 대기(PENDING)로 기록하고 ① 검토 화면에 제안한다.

        ① 검토자가 결정하면 /dataset/relation/decision 콜백으로 상태가 바뀐다.
        확정 관계만 조회·탐색에 쓰이므로 대기 중에는 노출되지 않는다.
        """
        self.relation_repo.replace(relation, "PENDING")
        logger.info(
            f"[Relation] 승인 대기 제안: {relation.table_a} <-> {relation.table_b} "
            f"(score={relation.score}, method={relation.method})"
        )
        self._push_relation_to_catalog(relation, status="PENDING")

    def review_relation(self, relation_id: int, approved: bool) -> dict:
        """판정 대기 관계를 승인·반려한다 (원장이 정본). 카탈로그와 연동된 관계는 결정을 환류한다.

        반려도 REJECTED 로 남겨 다음 파이프라인 실행에서 같은 쌍을 다시 제안하지 않게 한다.

        Raises:
            RelationNotFoundError: 없는 관계 id
        """
        row = self.relation_repo.set_status(relation_id, "APPROVED" if approved else "REJECTED")
        self._push_relation_to_catalog(
            DatasetRelation(
                table_a=row["table_a"], table_b=row["table_b"],
                catalog_identifier_a=row["catalog_identifier_a"], catalog_identifier_b=row["catalog_identifier_b"],
                score=row["score"], method=row["method"], relation_type=row["relation_type"], evidence=row["evidence"],
            ),
            status=row["status"],
        )
        return row

    def apply_relation_decision(
        self,
        source_identifier: str,
        target_identifier: str,
        relation_type: str,
        score: float,
        method: str,
        evidence: dict | None,
        approved: bool,
    ) -> DatasetRelation | None:
        """①의 승인/반려 결정을 관계 상태에 반영 (승인 워크플로우 콜백 전용, 재환류 없음).

        반려도 REJECTED 로 남겨 다음 파이프라인 실행에서 같은 쌍을 다시 제안하지 않게 한다.

        Returns:
            승인 시 기록된 DatasetRelation, 반려 시 None

        Raises:
            ValueError: 식별자에 해당하는 데이터셋 프로파일이 없을 때
        """
        profile_a = self.dataset_repo.get_by_identifier(source_identifier)
        profile_b = self.dataset_repo.get_by_identifier(target_identifier)
        if not profile_a or not profile_b:
            missing = source_identifier if not profile_a else target_identifier
            raise ValueError(f"카탈로그 식별자에 해당하는 데이터셋이 없습니다: {missing}")

        relation = DatasetRelation(
            table_a=profile_a.table_name,
            table_b=profile_b.table_name,
            catalog_identifier_a=source_identifier,
            catalog_identifier_b=target_identifier,
            score=score,
            method=method,
            relation_type=relation_type,
            evidence=evidence or {},
        )
        self.relation_repo.replace(relation, "APPROVED" if approved else "REJECTED")
        logger.info(
            f"[Relation] ① 결정 반영({'승인' if approved else '반려'}): "
            f"{source_identifier} <-> {target_identifier}"
        )
        return relation if approved else None

    def _push_relation_to_catalog(self, relation: DatasetRelation, status: str = "APPROVED") -> None:
        """관계를 ①의 dataset_relation 으로 환류. 실패해도 관계 저장 자체는 유지 (graceful)."""
        if not settings.CATALOG_PUSH_ENABLED:
            return
        if not (relation.catalog_identifier_a and relation.catalog_identifier_b):
            logger.debug(
                f"[Relation] 카탈로그 식별자 없음 — 환류 생략: {relation.table_a} <-> {relation.table_b}"
            )
            return
        try:
            from app.clients.catalog_client import get_catalog_client

            get_catalog_client().push_dataset_relation(relation, status=status)
            logger.info(
                f"[Relation] 카탈로그 환류 완료({status}): {relation.catalog_identifier_a} "
                f"--[{relation.relation_type}]--> {relation.catalog_identifier_b}"
            )
        except Exception as e:
            logger.warning(f"[Relation] 카탈로그 환류 실패 (원장 기록은 완료): {e}")

    def get_existing_relation_pairs(self) -> set[tuple[str, str]]:
        """판정 이력이 있는 쌍(확정·대기·반려) — 증분 실행에서 다시 평가하지 않는다."""
        return self.relation_repo.pairs()

    # ──────────────────────────────────────────────
    # 검색
    # ──────────────────────────────────────────────

    def search_datasets(
        self,
        query: str,
        domain: str | None = None,
        top_k: int = 5,
        party: str = "",
        provider: str = "",
    ) -> list[DatasetSearchResult]:
        """쿼리 임베딩으로 유사 테이블 KNN 검색.

        Args:
            query: 검색 쿼리 텍스트
            domain: 도메인 필터 (None이면 전체)
            top_k: 반환할 최대 결과 수
            party: 파티 필터 (1st/3rd, 빈 값이면 전체)
            provider: 제공 기관 필터 (빈 값이면 전체)

        Returns:
            DatasetSearchResult 리스트 (유사도 내림차순)
        """
        query_embedding = embedding_service.generate_embedding(query)
        return self.dataset_repo.search_knn(
            query_embedding, top_k=top_k, domain=domain, party=party, provider=provider
        )

    def nl_search(
        self,
        query: str,
        top_k: int = 5,
        domain: str | None = None,
        party: str = "",
        provider: str = "",
    ) -> dict[str, Any]:
        """자연어 검색 + 그래프 시각화용 부분 그래프 반환 (§9).

        자연어 질의를 검색 의도로 변환(LLM, 가능 시)하여 데이터셋 KNN 검색을 수행하고,
        상위 노드의 1-hop 이웃(Level 1 관계)을 묶어 부분 그래프(subgraph)로 반환한다.

        Args:
            query: 자연어 질의
            top_k: 상위 데이터셋 수
            domain: 도메인 필터
            party: 파티 필터 (1st/3rd)
            provider: 제공 기관 필터

        Returns:
            {"intent": str, "datasets": [...], "graph": {"nodes": [...], "edges": [...]}}
        """
        intent = self._extract_search_intent(query)
        results = self.search_datasets(intent, top_k=top_k, domain=domain, party=party, provider=provider)
        top_tables = [r.table_name for r in results]
        # 1위를 앵커로 삼아 엣지를 정리한다 (seed 전체의 1-hop 합집합은 읽을 수 없다)
        anchor = top_tables[0] if top_tables else None
        graph = self.get_dataset_subgraph(top_tables, anchor=anchor)
        answer, suggested = self._compose_search_answer(query, results, graph, anchor)

        return {
            "intent": intent,
            "answer": answer,
            "suggested_next": suggested,
            "datasets": [r.model_dump() for r in results],
            "graph": graph,
        }

    def _compose_search_answer(
        self,
        query: str,
        results: list[DatasetSearchResult],
        graph: dict[str, Any],
        anchor: str | None,
    ) -> tuple[str, list[str]]:
        """검색 결과와 관계 구조를 근거로 자연어 답변을 만든다.

        LLM 이 목록 밖 데이터셋을 지어내지 못하게 후보명을 대조해 걸러낸다.
        실패하면 빈 답변을 반환한다 — 검색 결과 자체는 그대로 쓸 수 있어야 한다.
        """
        if not results:
            return "", []

        names = {r.table_name for r in results}
        dataset_lines = "\n".join(
            f"- {r.table_name} (유사도 {r.score:.2f}, 제공기관 {r.provider or '미상'}): "
            f"{r.table_summary or r.description or '설명 없음'}"
            for r in results
        )
        relation_lines = "\n".join(
            f"- {e['source']} ─[{e['relation_type']}]→ {e['target']}" for e in graph["edges"]
        ) or "- (확정된 관계 없음 — 모두 유사도로만 검색됨)"

        try:
            result = LLMResponse("dataset_search_answer").generate_llm_response(
                variables={
                    "query": query,
                    "datasets": dataset_lines,
                    "relations": relation_lines,
                    "anchor": anchor or "없음",
                }
            )
        except Exception as e:
            logger.warning(f"[NL Search] 답변 생성 실패: {e}")
            return "", []

        if not isinstance(result, dict):
            return "", []
        suggested = [n for n in (result.get("suggested_next") or []) if n in names]
        return str(result.get("answer") or ""), suggested

    def related_search(
        self,
        table_name: str,
        top_k: int = 5,
        party: str = "3rd",
        domain: str | None = None,
        provider: str = "",
    ) -> dict[str, Any] | None:
        """기준 데이터셋(1st 파티) 중심의 연계 가능 데이터셋 검색 (사용자 시나리오 1단계).

        기존 확정 관계(1-hop)를 우선 신호로, 임베딩 KNN 유사 후보로 보강한다.
        nl_search 와 동일한 응답 형태를 반환하여 UI 렌더링을 공유한다.

        Args:
            table_name: 기준 데이터셋명
            top_k: KNN 후보 수
            party: 결과 파티 필터 — 기본 3rd (연계 가능한 외부 개방 데이터 탐색)
            domain: 도메인 필터
            provider: 제공 기관 필터

        Returns:
            {"intent", "datasets", "graph"} — 기준 데이터셋 미등록 시 None
        """
        profile = self.dataset_repo.get(table_name)
        if not profile:
            return None

        # 1) 기존 확정 관계 (1-hop) — 필터 조건에 맞는 이웃만 결과에 포함
        graph = self.get_dataset_subgraph([table_name])
        relation_by_neighbor: dict[str, dict[str, Any]] = {}
        for edge in graph["edges"]:
            neighbor = edge["target"] if edge["source"] == table_name else edge["source"]
            relation_by_neighbor[neighbor] = edge

        results: list[DatasetSearchResult] = []
        for neighbor, edge in relation_by_neighbor.items():
            p = self.dataset_repo.get(neighbor)
            if not p:
                continue
            if (party and p.party != party) or (domain and p.domain != domain) or (provider and p.provider != provider):
                continue
            results.append(DatasetSearchResult(
                table_name=p.table_name,
                display_name=p.display_name,
                domain=p.domain,
                description=p.description,
                table_summary=p.table_summary,
                score=edge["score"],
                party=p.party,
                provider=p.provider,
                landing_page=p.landing_page,
                match_type="relation",
                relation_type=type_seen_from(edge["relation_type"], edge["source"] == table_name),
                evidence=edge.get("evidence", {}),
            ))

        # 2) 임베딩 KNN 유사 후보로 보강 (기준 자신 제외, 관계로 이미 잡힌 것 제외)
        if profile.table_embedding:
            knn = self.dataset_repo.search_knn(
                profile.table_embedding,
                top_k=top_k,
                domain=domain,
                party=party,
                provider=provider,
                exclude_table_name=table_name,
            )
            results.extend(r for r in knn if r.table_name not in relation_by_neighbor)

        return {
            "intent": f"'{profile.display_name or table_name}' 기준 연계 가능 데이터셋",
            "datasets": [r.model_dump() for r in results],
            "graph": graph,
        }

    def _extract_search_intent(self, query: str) -> str:
        """자연어 질의를 데이터셋 검색 의도(요약문 유사 서술형 구문)로 변환.

        LLM(dataset_search_intent)을 시도하고 실패하면 원 질의를 그대로 사용한다.
        """
        try:
            result = LLMResponse("dataset_search_intent").generate_llm_response(
                variables={"query": query}
            )
            if isinstance(result, dict):
                for field in ("search_query", "query", "keywords", "intent"):
                    value = result.get(field)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
                    if isinstance(value, list) and value:
                        return " ".join(str(v) for v in value)
        except Exception as e:
            logger.debug(f"[NL Search] LLM 의도 추출 실패, 원 질의 사용: {e}")
        return query

    def get_dataset_subgraph(
        self, table_names: list[str], anchor: str | None = None
    ) -> dict[str, Any]:
        """주어진 데이터셋들과 그 1-hop 이웃을 부분 그래프로 반환 (§9).

        seed 가 여러 개면 seed 끼리도 서로 연결돼 있어 엣지가 급격히 늘고 읽을 수
        없게 된다(검색 상위 5건에서 노드 11·엣지 34). anchor 를 주면 그 데이터셋에
        직접 걸린 엣지만 남겨 "기준 데이터셋을 중심에 두고 거리로 표현한다"는
        설계(7항 화면 5)에 맞춘다. 노드는 그대로 두므로 앵커와 연결되지 않은
        검색 결과는 고립 노드로 보인다 — 유사도로만 잡혔다는 사실이 그대로 드러난다.

        Args:
            table_names: 중심 데이터셋명 목록 (검색 상위 노드)
            anchor: 엣지를 남길 기준 데이터셋. None 이면 seed 전체의 1-hop 을 합친다

        Returns:
            {"nodes": [{"id", "type", "seed"}], "edges": [{"source","target","relation_type","score","evidence"}]}
        """
        if not table_names:
            return {"nodes": [], "edges": []}
        nodes: dict[str, dict[str, Any]] = {
            name: {"id": name, "type": "Dataset", "seed": True} for name in table_names
        }
        edges = [
            {k: e[k] for k in ("source", "target", "relation_type", "score", "evidence")}
            for e in self.relation_repo.edges(table_names)
            if not anchor or anchor in (e["source"], e["target"])
        ]
        for e in edges:
            for t in (e["source"], e["target"]):
                nodes.setdefault(t, {"id": t, "type": "Dataset", "seed": False})

        return {"nodes": list(nodes.values()), "edges": edges}

    # ──────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────

    def _build_profile(
        self,
        records: list[dict[str, Any]],
        table_name: str,
        domain: str,
        description: str,
        table_summary: str,
        seed_keywords: list[str] | None = None,
        column_names: list[str] | None = None,
    ) -> DatasetProfile:
        column_names = list(records[0].keys()) if records else list(column_names or [])
        sample_records = records[:_SAMPLE_SIZE]
        summary_text = table_summary or description or f"{table_name} 테이블"
        table_embedding = embedding_service.generate_embedding(summary_text)

        # 표시명: LLM 경로에서 자동 생성. 시드 경로(카탈로그 pull)는 table_name이 이미 한글 title
        display_name = ""
        if seed_keywords:
            keywords, keywords_embedding = seed_keywords, self._embed_keywords(seed_keywords)
        else:
            keywords, keywords_embedding, display_name = self._extract_keywords(
                table_name, column_names, sample_records
            )

        categories = normalize_categories([domain]) or CategoryClassifier().classify(
            table_name, summary_text, keywords, column_names, table_embedding
        )
        logger.info(f"[Dataset] '{table_name}' 분야 분류: {categories or '미분류'}")

        return DatasetProfile(
            table_name=table_name,
            display_name=display_name,
            domain=major_of(categories[0]) if categories else "",
            categories=categories,
            record_count=len(records),
            column_count=len(column_names),
            column_names=column_names,
            sample_records=sample_records,
            description=description,
            table_summary=summary_text,
            table_embedding=table_embedding,
            keywords=keywords,
            keywords_embedding=keywords_embedding,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def _extract_keywords(
        self,
        table_name: str,
        column_names: list[str],
        sample_records: list[dict[str, Any]],
    ) -> tuple[list[str], list[float], str]:
        """LLM으로 키워드·표시명 추출 후 (키워드, 평균 임베딩, 표시명) 반환."""
        try:
            variables = {
                "table_name": table_name,
                "columns": ", ".join(column_names),
                "column_descriptions": ", ".join(column_names),
                "sample_records": json.dumps(sample_records[:3], ensure_ascii=False),
            }
            result = LLMResponse("table_metadata_extraction").generate_llm_response(variables)
            if not isinstance(result, dict):
                return [], [], ""
            keywords = result.get("keywords", [])
            display_name = (result.get("display_name") or "").strip()
            if not keywords:
                return [], [], display_name
            return keywords, self._embed_keywords(keywords), display_name
        except Exception as e:
            logger.warning(f"[Dataset] '{table_name}' 키워드 추출 실패: {e}")
            return [], [], ""

    @staticmethod
    def _embed_keywords(keywords: list[str]) -> list[float]:
        """키워드 목록의 평균 임베딩 벡터를 반환."""
        from app.utils.vector import mean_vector

        return mean_vector([embedding_service.generate_embedding(kw) for kw in keywords])

    def delete_table(self, table_name: str) -> bool:
        """데이터셋 삭제 — 파일·임베딩·관계는 원장에서 함께 정리된다."""
        ok = self.dataset_repo.delete(table_name)
        logger.info(f"[Dataset] '{table_name}' 삭제 {'완료' if ok else '대상 없음'}")
        return ok


dataset_service = DatasetService()
