"""Dataset 메타데이터 추출 및 Neptune/OpenSearch 적재 서비스."""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

from app.clients.neptune import NeptuneClient, get_neptune_client
from app.config import settings
from app.repositories.dataset_repository import DatasetRepository, get_dataset_repository
from app.schemas.dataset import DatasetProfile, DatasetRelation, DatasetSearchResult
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
    ) -> list[DatasetSearchResult]:
        """쿼리 임베딩으로 유사 테이블 KNN 검색.

        Args:
            query: 검색 쿼리 텍스트
            domain: 도메인 필터 (None이면 전체)
            top_k: 반환할 최대 결과 수

        Returns:
            DatasetSearchResult 리스트 (유사도 내림차순)
        """
        query_embedding = embedding_service.generate_embedding(query)
        return self.dataset_repo.search_knn(query_embedding, top_k=top_k, domain=domain)

    def nl_search(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """자연어 검색 + 그래프 시각화용 부분 그래프 반환 (§9).

        자연어 질의를 검색 의도로 변환(LLM, 가능 시)하여 데이터셋 KNN 검색을 수행하고,
        상위 노드의 1-hop 이웃(Level 1 관계)을 묶어 부분 그래프(subgraph)로 반환한다.

        Args:
            query: 자연어 질의
            top_k: 상위 데이터셋 수

        Returns:
            {"intent": str, "datasets": [...], "graph": {"nodes": [...], "edges": [...]}}
        """
        intent = self._extract_search_intent(query)
        results = self.search_datasets(intent, top_k=top_k)
        top_tables = [r.table_name for r in results]
        graph = self.get_dataset_subgraph(top_tables)

        return {
            "intent": intent,
            "datasets": [r.model_dump() for r in results],
            "graph": graph,
        }

    def _extract_search_intent(self, query: str) -> str:
        """자연어 질의를 검색 의도(키워드/구문)로 변환.

        LLM(query_key_inference)을 시도하고 실패하면 원 질의를 그대로 사용한다.
        """
        try:
            result = LLMResponse("query_key_inference").generate_llm_response(
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

    def get_dataset_subgraph(self, table_names: list[str]) -> dict[str, Any]:
        """주어진 데이터셋들과 그 1-hop 이웃을 부분 그래프로 반환 (§9).

        Args:
            table_names: 중심 데이터셋명 목록 (검색 상위 노드)

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
            for b in resp.get("results", {}).get("bindings", []):
                ta, tb = b["tableA"]["value"], b["tableB"]["value"]
                for t in (ta, tb):
                    if t not in nodes:
                        nodes[t] = {"id": t, "type": "Dataset", "seed": False}
                edges.append({
                    "source": ta,
                    "target": tb,
                    "relation_type": b["relationType"]["value"],
                    "score": float(b["score"]["value"]),
                })
        except Exception as e:
            logger.warning(f"[NL Search] 부분 그래프 조회 실패: {e}")

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
    ) -> DatasetProfile:
        column_names = list(records[0].keys()) if records else []
        sample_records = records[:_SAMPLE_SIZE]
        summary_text = table_summary or description or f"{table_name} 테이블"
        table_embedding = embedding_service.generate_embedding(summary_text)

        # 도메인 미지정 시 특징(요약 임베딩) 기반 자동 분류 (계획서 ②-1)
        domain_confidence = None
        if not domain:
            from app.src.dataset.domain_classifier import DomainClassifier

            domain, domain_confidence, scores = DomainClassifier(self.dataset_repo).classify(table_embedding)
            logger.info(
                f"[Dataset] '{table_name}' 도메인 자동 분류: {domain} "
                f"(conf={domain_confidence}, scores={scores})"
            )

        # 표시명: LLM 경로에서 자동 생성. 시드 경로(카탈로그 pull)는 table_name이 이미 한글 title
        display_name = ""
        if seed_keywords:
            keywords, keywords_embedding = seed_keywords, self._embed_keywords(seed_keywords)
        else:
            keywords, keywords_embedding, display_name = self._extract_keywords(
                table_name, column_names, sample_records
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
    ) -> tuple[list[str], list[float], str]:
        """LLM으로 키워드·한국어 표시명 추출 후 (키워드, 평균 임베딩, 표시명) 반환."""
        import json
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
