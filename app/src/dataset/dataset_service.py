"""Dataset 메타데이터 추출 및 Neptune/OpenSearch 적재 서비스."""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

from app.clients.neptune import NeptuneClient, get_neptune_client
from app.config import settings
from app.repositories.dataset_repository import DatasetRepository, get_dataset_repository
from app.schemas.dataset import DatasetProfile, DatasetRelation, DatasetSearchResult, migrate_stored_type
from app.src.embedding.embedding_service import embedding_service
from app.src.meta_extract.llm_response import LLMResponse
from app.src.ontology.ontology_service import OntologyService

logger = logging.getLogger(__name__)

_DM = settings.DATASET_RDF_NAMESPACE
_SAMPLE_SIZE = 10


class DatasetService:
    """테이블 메타데이터 추출 → Neptune Dataset 노드 + OpenSearch 적재."""

    def __init__(
        self,
        neptune_client: NeptuneClient | None = None,
        dataset_repo: DatasetRepository | None = None,
        ontology_service: OntologyService | None = None,
    ):
        """Neptune·OpenSearch 클라이언트 및 온톨로지 서비스를 주입받아 초기화."""
        self.neptune = neptune_client or get_neptune_client()
        self.dataset_repo = dataset_repo or get_dataset_repository()
        self.ontology = ontology_service or OntologyService(neptune_client=self.neptune)

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
    ) -> DatasetProfile:
        """테이블 데이터 → 메타데이터 추출 → Neptune + OpenSearch 저장.

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

        Returns:
            저장된 DatasetProfile
        """
        if not records and not (table_summary or description):
            raise ValueError(f"'{table_name}': 레코드가 없고 요약 메타데이터도 없습니다")

        profile = self._build_profile(
            records, table_name, domain, description, table_summary, seed_keywords=keywords
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

        try:
            self._save_to_neptune(profile)
            # 온톨로지 의미 계층: 데이터셋을 도메인 개념에 자동 연결
            self.ontology.link_dataset_to_domain(profile.table_name, profile.domain)
        except Exception as e:
            logger.warning(f"[Dataset] Neptune 저장 실패 (OpenSearch는 계속 진행): {e}")

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
        """두 Dataset 간 관계를 Neptune에 upsert (기존 관계 삭제 후 재저장).

        Args:
            relation: DatasetRelation 객체
            push_to_catalog: ① 환류 여부 (① 승인 콜백 경로에서는 False — 재환류 방지)
        """
        subject_uri = self._dataset_uri(relation.table_a)
        object_uri = self._dataset_uri(relation.table_b)
        created_at = datetime.now(timezone.utc).isoformat()
        relation_type_escaped = relation.relation_type.replace('"', '\\"')

        # 두 테이블 간 기존 관계(양방향) 삭제 후 새로 INSERT
        update = f"""
        PREFIX dm: <{_DM}>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        DELETE {{
            ?rel ?p ?o .
        }}
        WHERE {{
            ?rel rdf:type rdf:Statement ;
                 ?p ?o .
            FILTER (
                (EXISTS {{ ?rel rdf:subject {subject_uri} . ?rel rdf:object {object_uri} }}) ||
                (EXISTS {{ ?rel rdf:subject {object_uri} . ?rel rdf:object {subject_uri} }})
            )
        }} ;

        INSERT DATA {{
            <{_DM}relation/{quote(relation.table_a)}-{quote(relation.table_b)}>
                rdf:type rdf:Statement ;
                rdf:subject {subject_uri} ;
                rdf:predicate dm:relatedTo ;
                rdf:object {object_uri} ;
                dm:relationScore "{relation.score}"^^xsd:float ;
                dm:relationMethod "{relation.method}" ;
                dm:relationType "{relation_type_escaped}" ;
                dm:createdAt "{created_at}"^^xsd:dateTime .
        }}
        """
        self.neptune.execute_update(update)
        logger.info(
            f"[Relation] {relation.table_a} --[{relation.relation_type}]--> {relation.table_b} "
            f"(score={relation.score}, method={relation.method})"
        )

        if push_to_catalog and settings.CATALOG_PUSH_ENABLED:
            self._push_relation_to_catalog(relation, status="APPROVED")

    def propose_relation_to_catalog(self, relation: DatasetRelation) -> None:
        """저신뢰 관계를 그래프 저장 없이 ①에 PENDING 으로 제안 (승인 SSOT = ①).

        ① 검토자가 승인하면 /dataset/relation/decision 콜백으로 그래프에 반영된다.
        """
        logger.info(
            f"[Relation] 승인 대기 제안: {relation.table_a} <-> {relation.table_b} "
            f"(score={relation.score}, method={relation.method})"
        )
        self._push_relation_to_catalog(relation, status="PENDING")

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
        """①의 승인/반려 결정을 그래프에 반영 (승인 워크플로우 콜백 전용).

        승인: 관계를 Neptune에 저장 (재환류 없음 — 상태는 ①이 이미 보유).
        반려: 그래프에 저장된 적 없으므로 no-op.

        Returns:
            승인 시 저장된 DatasetRelation, 반려 시 None

        Raises:
            ValueError: 식별자에 해당하는 데이터셋 프로파일이 없을 때
        """
        if not approved:
            logger.info(f"[Relation] 반려 결정 수신 — 그래프 미반영: {source_identifier} <-> {target_identifier}")
            return None

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
        self.save_relation(relation, push_to_catalog=False)
        return relation

    def _push_relation_to_catalog(self, relation: DatasetRelation, status: str = "APPROVED") -> None:
        """관계를 ①의 dataset_relation 으로 환류. 실패해도 관계 저장 자체는 유지 (graceful)."""
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
            logger.warning(f"[Relation] 카탈로그 환류 실패 (Neptune 저장은 완료): {e}")

    def get_existing_relation_pairs(self) -> set[tuple[str, str]]:
        """Neptune에 저장된 관계 쌍을 (table_a, table_b) 집합으로 반환 (정렬된 튜플)."""
        query = f"""
        PREFIX dm: <{_DM}>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

        SELECT ?tableA ?tableB WHERE {{
            ?rel rdf:type rdf:Statement ;
                 rdf:subject ?s ;
                 rdf:object  ?o .
            ?s dm:tableName ?tableA .
            ?o dm:tableName ?tableB .
        }}
        """
        try:
            rows = self.neptune.execute_query(query)
            return {
                tuple(sorted([r["tableA"]["value"], r["tableB"]["value"]]))
                for r in rows.get("results", {}).get("bindings", [])
            }
        except Exception as e:
            logger.warning(f"[Dataset] 기존 관계 쌍 조회 실패: {e}")
            return set()

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

        기존 확정 관계(Neptune 1-hop)를 우선 신호로, 임베딩 KNN 유사 후보로 보강한다.
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
                relation_type=edge["relation_type"],
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
            {"nodes": [{"id", "type", "seed"}], "edges": [{"source","target","relation_type","score"}]}
        """
        nodes: dict[str, dict[str, Any]] = {
            name: {"id": name, "type": "Dataset", "seed": True} for name in table_names
        }
        edges: list[dict[str, Any]] = []
        if not table_names:
            return {"nodes": [], "edges": []}

        values_filter = " ".join(f'"{t.replace(chr(34), "")}"' for t in table_names)
        query = f"""
        PREFIX dm: <{_DM}>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT ?tableA ?tableB ?relationType ?score WHERE {{
            ?rel rdf:type rdf:Statement ;
                 rdf:subject ?s ;
                 rdf:predicate dm:relatedTo ;
                 rdf:object ?o ;
                 dm:relationType ?relationType ;
                 dm:relationScore ?score .
            ?s dm:tableName ?tableA .
            ?o dm:tableName ?tableB .
            VALUES ?seed {{ {values_filter} }}
            FILTER (?tableA = ?seed || ?tableB = ?seed)
        }}
        """
        try:
            resp = self.neptune.execute_query(query)
            # 양끝이 모두 seed인 관계는 VALUES 절에 두 번 매칭되므로 쌍 단위로 중복을 제거한다
            seen: set[frozenset[str]] = set()
            for b in resp.get("results", {}).get("bindings", []):
                ta, tb = b["tableA"]["value"], b["tableB"]["value"]
                pair = frozenset((ta, tb))
                if pair in seen:
                    continue
                seen.add(pair)
                if anchor and anchor not in pair:
                    continue
                for t in (ta, tb):
                    if t not in nodes:
                        nodes[t] = {"id": t, "type": "Dataset", "seed": False}
                edges.append({
                    "source": ta,
                    "target": tb,
                    "relation_type": migrate_stored_type(b["relationType"]["value"]),
                    "score": float(b["score"]["value"]),
                })
        except Exception as e:
            logger.warning(f"[NL Search] 부분 그래프 조회 실패: {e}")

        return {"nodes": list(nodes.values()), "edges": edges}

    # ──────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────

    def _select_domain_via_llm(self, table_name: str, summary_text: str, keywords: list[str]) -> str:
        """분류기 폴백 시 LLM으로 도메인 결정 - 기존 도메인 재사용 우선.

        Returns:
            도메인 라벨. general 응답·호출 실패면 빈 값 (호출부에서 general 유지)
        """
        from app.src.dataset.domain_classifier import FALLBACK_DOMAIN, STANDARD_DOMAINS

        existing = {
            d for _, d, _ in (self.dataset_repo.list_domain_embeddings() or [])
            if d and d != FALLBACK_DOMAIN
        }
        candidates = list(STANDARD_DOMAINS) + sorted(existing - set(STANDARD_DOMAINS))

        try:
            result = LLMResponse("domain_selection").generate_llm_response(variables={
                "title": table_name,
                "summary": summary_text,
                "keywords": ", ".join(keywords[:15]),
                "candidates": ", ".join(candidates),
            })
            picked = (result.get("domain") or "").strip() if isinstance(result, dict) else ""
            if not picked or picked.lower() == FALLBACK_DOMAIN:
                return ""
            if picked not in candidates:
                logger.info(f"[Dataset] '{table_name}' LLM 도메인 신설: {picked}")
            return picked
        except Exception as e:
            logger.warning(f"[Dataset] '{table_name}' LLM 도메인 선택 실패 — general 유지: {e}")
            return ""

    def _build_profile(
        self,
        records: list[dict[str, Any]],
        table_name: str,
        domain: str,
        description: str,
        table_summary: str,
        seed_keywords: list[str] | None = None,
    ) -> DatasetProfile:
        column_names = list(records[0].keys()) if records else []
        sample_records = records[:_SAMPLE_SIZE]
        summary_text = table_summary or description or f"{table_name} 테이블"
        table_embedding = embedding_service.generate_embedding(summary_text)

        # 표시명: LLM 경로에서 자동 생성. 시드 경로(카탈로그 pull)는 table_name이 이미 한글 title
        display_name = ""
        llm_domain = ""
        if seed_keywords:
            keywords, keywords_embedding = seed_keywords, self._embed_keywords(seed_keywords)
        else:
            keywords, keywords_embedding, display_name, llm_domain = self._extract_keywords(
                table_name, column_names, sample_records
            )

        domain_confidence = None
        if not domain:
            from app.src.dataset.domain_classifier import (
                FALLBACK_DOMAIN,
                STANDARD_DOMAINS,
                DomainClassifier,
            )

            domain, domain_confidence, scores = DomainClassifier(self.dataset_repo).classify(
                table_embedding, exclude_table_name=table_name
            )
            if domain == FALLBACK_DOMAIN:
                llm_pick = (
                    llm_domain
                    if llm_domain in STANDARD_DOMAINS
                    else self._select_domain_via_llm(table_name, summary_text, keywords)
                )
                if llm_pick:
                    domain, domain_confidence = llm_pick, None
            logger.info(
                f"[Dataset] '{table_name}' 도메인 자동 분류: {domain} "
                f"(conf={domain_confidence}, llm_domain={llm_domain or '-'}, scores={scores})"
            )

        return DatasetProfile(
            table_name=table_name,
            display_name=display_name,
            domain=domain,
            domain_confidence=domain_confidence,
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
    ) -> tuple[list[str], list[float], str, str]:
        """LLM으로 키워드·표시명·도메인 추출 후 (키워드, 평균 임베딩, 표시명, 도메인) 반환.

        도메인은 기존 등록 도메인 목록을 프롬프트에 제공해 표기 통일을 유도하고,
        centroid 분류가 폴백될 때(콜드스타트 등)만 사용된다.
        """
        import json
        try:
            existing_domains = sorted({
                d for _, d, _ in (self.dataset_repo.list_domain_embeddings() or []) if d and d != "general"
            })
            variables = {
                "table_name": table_name,
                "columns": ", ".join(column_names),
                "column_descriptions": ", ".join(column_names),
                "sample_records": json.dumps(sample_records[:3], ensure_ascii=False),
                "existing_domains": ", ".join(existing_domains),
            }
            result = LLMResponse("table_metadata_extraction").generate_llm_response(variables)
            if not isinstance(result, dict):
                return [], [], "", ""
            keywords = result.get("keywords", [])
            display_name = (result.get("display_name") or "").strip()
            llm_domain = (result.get("domain") or "").strip()
            if not keywords:
                return [], [], display_name, llm_domain
            return keywords, self._embed_keywords(keywords), display_name, llm_domain
        except Exception as e:
            logger.warning(f"[Dataset] '{table_name}' 키워드 추출 실패: {e}")
            return [], [], "", ""

    @staticmethod
    def _embed_keywords(keywords: list[str]) -> list[float]:
        """키워드 목록의 평균 임베딩 벡터를 반환."""
        from app.utils.vector import mean_vector

        return mean_vector([embedding_service.generate_embedding(kw) for kw in keywords])

    def _save_to_neptune(self, profile: DatasetProfile) -> None:
        """Dataset 노드를 Neptune SPARQL INSERT로 저장 (upsert 효과)."""
        dataset_uri = self._dataset_uri(profile.table_name)
        description_escaped = profile.description.replace("\\", "\\\\").replace('"', '\\"')

        catalog_id_triple = (
            f' ;\n                dm:catalogIdentifier "{profile.catalog_identifier}"'
            if profile.catalog_identifier
            else ""
        )
        update = f"""
        PREFIX dm: <{_DM}>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        DELETE {{
            {dataset_uri} dm:tableName ?tableName ;
                           dm:domain ?domain ;
                           dm:description ?desc .
            {dataset_uri} dm:catalogIdentifier ?catalogId .
        }}
        WHERE {{
            OPTIONAL {{ {dataset_uri} dm:tableName ?tableName }}
            OPTIONAL {{ {dataset_uri} dm:domain ?domain }}
            OPTIONAL {{ {dataset_uri} dm:description ?desc }}
            OPTIONAL {{ {dataset_uri} dm:catalogIdentifier ?catalogId }}
        }} ;

        INSERT DATA {{
            {dataset_uri} rdf:type dm:Dataset ;
                dm:tableName "{profile.table_name}" ;
                dm:domain "{profile.domain}" ;
                dm:description "{description_escaped}"{catalog_id_triple} .
        }}
        """
        self.neptune.execute_update(update)
        logger.debug(f"[Neptune] Dataset 노드 저장: {profile.table_name}")

    def delete_table(self, table_name: str) -> bool:
        """OpenSearch + Neptune에서 Dataset 노드 및 연관 관계를 모두 삭제."""
        try:
            self._delete_from_neptune(table_name)
        except Exception as e:
            logger.warning(f"[Dataset] Neptune 삭제 실패 (OpenSearch는 계속 진행): {e}")

        ok = self.dataset_repo.delete(table_name)
        logger.info(f"[Dataset] '{table_name}' 삭제 완료 (opensearch={ok})")
        return ok

    def _delete_from_neptune(self, table_name: str) -> None:
        """Dataset 노드와 해당 테이블이 주체/대상인 관계 reification 트리플을 모두 삭제."""
        dataset_uri = self._dataset_uri(table_name)

        # Dataset 노드 트리플 + 해당 테이블이 subject 또는 object인 관계 reification 전체 삭제
        update = f"""
        PREFIX dm: <{_DM}>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

        DELETE {{
            {dataset_uri} ?p ?o .
            ?rel ?rp ?ro .
        }}
        WHERE {{
            {{
                {dataset_uri} ?p ?o .
            }}
            UNION
            {{
                ?rel rdf:type rdf:Statement ;
                     ?rp ?ro .
                FILTER (
                    ?rel != rdf:type &&
                    (
                        EXISTS {{ ?rel rdf:subject {dataset_uri} }} ||
                        EXISTS {{ ?rel rdf:object  {dataset_uri} }}
                    )
                )
            }}
        }}
        """
        self.neptune.execute_update(update)
        logger.debug(f"[Neptune] Dataset 삭제: {table_name}")

    def _dataset_uri(self, table_name: str) -> str:
        return f"<{_DM}dataset/{quote(table_name)}>"


dataset_service = DatasetService()
