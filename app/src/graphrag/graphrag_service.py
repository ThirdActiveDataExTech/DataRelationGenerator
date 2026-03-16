import ast
import re
from typing import Dict, Any, List

import logging
from neo4j.exceptions import (
    ClientError,
    ServiceUnavailable,
    TransientError
)
from neo4j.time import DateTime
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever

from app.config import settings
from app.repositories.graph_queries import (
    GET_DOCUMENT_INFO,
    GET_RELATED_DOCUMENTS,
    EXTRACT_GRAPH_1HOP,
    EXTRACT_GRAPH_2HOP,
    CALCULATE_COMMON_METADATA,
    GET_DOCUMENT_NODES,
    GET_CATEGORY_RELATIONSHIPS,
    GET_ENTITY_RELATIONSHIPS,
    GET_NODE_BY_ENTITY,
    GET_NODE_BY_CATEGORY,
    GET_NODE_BY_TOPIC,
    GET_NODE_BY_ID,
    EXPLORE_NODE_TEMPLATE,
    GET_DOCUMENTS_BY_IDS
)
from app.repositories.graph_repository import GraphRepository
from app.schemas.graphrag_api import (
    GraphRAGSearchResponse,
    NodeExploreResponse,
    GraphStructure,
    CommonMetadata,
    GraphNode
)
from app.src.graphrag.reranker import BGEReranker
from app.src.meta_extract.llm_response import LLMResponse


class GraphRAGService:
    """Graph RAG 검색 서비스 - 사용자 질의 기반 그래프 구조 반환"""

    def __init__(self, graph_repo: GraphRepository):
        """GraphRAGService 초기화"""
        self.graph_repo = graph_repo
        self.driver = graph_repo.driver
        self.embedder = SentenceTransformerEmbeddings(model=settings.EMBEDDING_MODEL_NAME)
        self._reranker = None  # Lazy initialization

    @property
    def reranker(self) -> BGEReranker:
        """Reranker lazy initialization (첫 사용 시에만 로딩)"""
        if self._reranker is None:
            self._reranker = BGEReranker()
        return self._reranker

    @staticmethod
    def _sanitize_properties(properties: Dict[str, Any]) -> Dict[str, Any]:
        """Neo4j 속성을 JSON 직렬화 가능한 형태로 변환"""
        sanitized = {}
        for key, value in properties.items():
            if isinstance(value, DateTime):
                sanitized[key] = value.isoformat()
            elif isinstance(value, (list, tuple)):
                sanitized[key] = [v.isoformat() if isinstance(v, DateTime) else v for v in value]
            else:
                sanitized[key] = value
        return sanitized

    def search_graph(
            self,
            query: str,
            top_k: int = 5
    ) -> Dict[str, Any]:
        """사용자 질의 기반 Graph RAG 검색"""
        try:
            documents = self._search_documents(query, top_k)

            if not documents:
                return GraphRAGSearchResponse.empty(query).model_dump()

            doc_ids = [doc["doc_id"] for doc in documents]
            common_metadata = self._calculate_common_metadata(doc_ids)
            graph = self._build_simple_graph(doc_ids, common_metadata)

            logging.info(
                f"Graph RAG 검색 완료: {len(documents)}개 Document, "
                f"{len(graph['nodes'])}개 노드, {len(graph['relationships'])}개 관계"
            )

            response = GraphRAGSearchResponse(
                query=query,
                documents=documents,
                common_metadata=CommonMetadata(**common_metadata),
                graph=GraphStructure(**graph)
            )
            return response.model_dump()

        except (ServiceUnavailable, TransientError, ClientError, ValueError, KeyError, AttributeError) as e:
            logging.exception(f"Graph RAG 검색 실패: {e}")
            return GraphRAGSearchResponse.empty(query, error=str(e)).model_dump()

    def _search_documents(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """하이브리드 검색: 그래프 패턴 매칭 + 텍스트 임베딩 + LLM 검증"""
        logging.info(f"검색 시작: query='{query}', top_k={top_k}")

        # 1. 그래프 패턴 기반 검색 (후보군 확보)
        pattern_results = self._search_by_graph_pattern(query, top_k * 10)
        logging.info(f"[1단계] 패턴 매칭 결과: {len(pattern_results)}개")

        # 2. 텍스트 임베딩 기반 검색
        vector_results = self._search_by_text_embedding(query, top_k * 10)
        logging.info(f"[2단계] 벡터 검색 결과: {len(vector_results)}개")

        # 3. 결과 병합 및 재순위화
        merged_results = self._merge_and_rerank(pattern_results, vector_results, top_k * 3)
        logging.info(f"[3단계] 병합 후 상위 {top_k * 3}개 선택: {len(merged_results)}개")

        # 4. 경량 Reranker 필터링 (BGE Reranker)
        reranked_results = self.reranker.rerank(query, merged_results, top_k)
        logging.info(f"[4단계] Reranker 필터링: {len(merged_results)}개 → {len(reranked_results)}개")

        # 5. LLM 검증 (상위 후보만, early stopping 적용)
        verified_results = self._verify_results_with_llm(query, reranked_results, top_k)
        logging.info(f"[5단계] LLM 검증 통과: {len(verified_results)}개")

        # 6. 최종 재순위화
        final_results = self._final_rerank(verified_results, top_k)
        logging.info(f"[6단계] 최종 결과: {len(final_results)}개")

        return final_results

    def _verify_results_with_llm(
            self,
            query: str,
            results: List[Dict[str, Any]],
            top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """LLM으로 검색 결과 적합성 검증 (Early Stopping 적용)

        Args:
            query: 검색 쿼리
            results: 검증할 결과 목록
            top_k: 목표 결과 개수 (이만큼 확보하면 조기 종료)
        """
        verified = []
        filtered_out = []

        for idx, result in enumerate(results):
            # Early stopping: top_k개 확보하면 중단
            if len(verified) >= top_k:
                logging.info(f"✓ {top_k}개 확보 완료, LLM 검증 조기 종료 ({idx}/{len(results)} 검증)")
                break

            try:
                # 원본 text 전체 사용 (최대 10000자)
                document = result.get("text", "")
                if len(document) > 10000:
                    document = document[:10000]

                logging.info(
                    f"[LLM 검증 {idx + 1}/{len(results)}] doc_id: {result.get('doc_id')}, text_len: {len(document)}")

                # LLM 검증
                verification = LLMResponse("document_relevance_verification").generate_llm_response(
                    variables={"query": query, "document": document}
                )

                if isinstance(verification, dict):
                    result["llm_relevant"] = verification.get("is_relevant", False)
                    result["llm_reason"] = verification.get("reason", "")

                    # 적합하다고 판단된 것만 통과
                    if result["llm_relevant"]:
                        logging.info(f"  ✓ 통과 - {result['llm_reason'][:50]}...")
                        verified.append(result)
                    else:
                        logging.info(f"  ✗ 제외 - {result['llm_reason'][:100]}...")
                        filtered_out.append({
                            "doc_id": result.get('doc_id'),
                            "reason": result['llm_reason']
                        })
                else:
                    logging.warning(f"LLM 검증 결과가 dict가 아님: {type(verification)}")
                    # 검증 실패시 일단 포함 (보수적 접근)
                    result["llm_relevant"] = True
                    verified.append(result)

            except Exception as e:
                logging.exception(f"LLM 검증 중 에러: {e}")
                # 검증 실패시 일단 포함
                result["llm_relevant"] = True
                verified.append(result)

        logging.info(f"LLM 검증 완료: {len(results)}개 → {len(verified)}개 통과, {len(filtered_out)}개 제외")
        if filtered_out:
            logging.info(f"제외된 문서 목록:")
            for item in filtered_out:
                logging.info(f"  - {item['doc_id']}: {item['reason'][:100]}")

        return verified

    def _final_rerank(self, results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """최종 재순위화 (Pattern + Vector + Reranker)"""
        for result in results:
            pattern_score = result.get("pattern_score", 0)
            vector_score = result.get("vector_score", 0)
            rerank_score = result.get("rerank_score", 0)

            # 최종 점수: 패턴 33.3% + 벡터 33.3% + Reranker 33.3%
            result["final_score"] = (
                    pattern_score * 0.333 +
                    vector_score * 0.333 +
                    rerank_score * 0.334
            )

        # 정렬
        sorted_results = sorted(results, key=lambda x: x["final_score"], reverse=True)
        return sorted_results[:top_k]

    def _search_by_graph_pattern(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """그래프 패턴 매칭 기반 검색"""
        # LLM으로 쿼리 → 그래프 패턴 변환
        graph_pattern = self._parse_query_to_graph_pattern(query)

        if not graph_pattern or not graph_pattern.get("conditions"):
            return []

        # 그래프 패턴 → 결과 직접 생성 (다중 조건 처리)
        results = self._build_pattern_matching_query(graph_pattern, top_k)

        return [
            {
                "doc_id": r["doc_id"],
                "pattern_score": r.get("match_score", 0),
                "matched_conditions": r.get("matched_conditions", [])
            }
            for r in results
        ]

    def _search_by_text_embedding(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """기존 텍스트 임베딩 기반 검색"""
        retriever = VectorRetriever(
            driver=self.driver,
            index_name="text_embedding_index",
            embedder=self.embedder
        )
        result = retriever.search(query_text=query, top_k=top_k)

        documents = []
        for item in result.items:
            doc_id = self._extract_doc_id(item.content)
            if not doc_id:
                continue

            score = item.metadata.get("score", 0) if hasattr(item, 'metadata') and item.metadata else 0
            documents.append({
                "doc_id": doc_id,
                "vector_score": score
            })

        return documents

    def _merge_and_rerank(
            self,
            pattern_results: List[Dict[str, Any]],
            vector_results: List[Dict[str, Any]],
            top_k: int
    ) -> List[Dict[str, Any]]:
        """패턴 매칭과 벡터 검색 결과 병합 및 재순위화"""
        # doc_id를 키로 결과 병합
        merged = {}

        for r in pattern_results:
            doc_id = r["doc_id"]
            merged[doc_id] = {
                "doc_id": doc_id,
                "pattern_score": r.get("pattern_score", 0),
                "vector_score": 0,
                "matched_conditions": r.get("matched_conditions", [])
            }

        for r in vector_results:
            doc_id = r["doc_id"]
            if doc_id in merged:
                merged[doc_id]["vector_score"] = r.get("vector_score", 0)
            else:
                merged[doc_id] = {
                    "doc_id": doc_id,
                    "pattern_score": 0,
                    "vector_score": r.get("vector_score", 0),
                    "matched_conditions": []
                }

        # 최종 점수 계산 (패턴 매칭 60%, 벡터 유사도 40%)
        for doc_id, data in merged.items():
            data["final_score"] = data["pattern_score"] * 0.6 + data["vector_score"] * 0.4

        # 정렬
        sorted_results = sorted(merged.values(), key=lambda x: x["final_score"], reverse=True)

        # Document 정보 추가
        final_results = []
        for r in sorted_results[:top_k]:
            doc_info = self._get_document_info(r["doc_id"])
            if not doc_info:
                continue

            related_docs = self._get_related_documents(r["doc_id"], limit=5)

            final_results.append({
                "doc_id": r["doc_id"],
                "text": doc_info.get("text", ""),
                "description": doc_info.get("description", ""),
                "score": r["final_score"],
                "pattern_score": r["pattern_score"],
                "vector_score": r["vector_score"],
                "matched_conditions": r["matched_conditions"],
                "related_documents": related_docs
            })

        return final_results

    def _parse_query_to_graph_pattern(self, query: str) -> Dict[str, Any]:
        """LLM으로 쿼리를 그래프 패턴으로 변환

        Returns:
            {
                "target": "Document",
                "conditions": [
                    {"type": "entity", "value": "가평", "semantic": "위치/지역"},
                    {"type": "entity", "value": "재활의학과", "semantic": "진료과목"},
                    {"type": "property", "value": "종합병원", "semantic": "병원유형"}
                ]
            }
        """
        try:
            result = LLMResponse("query_graph_parsing").generate_llm_response(
                variables={"query": query}
            )

            if isinstance(result, dict):
                return result

            logging.warning(f"쿼리 그래프 파싱 결과가 dict가 아님: {type(result)}")
            return {"target": "Document", "conditions": []}

        except Exception as e:
            logging.exception(f"쿼리 그래프 파싱 실패: {e}")
            return {"target": "Document", "conditions": []}

    def _build_pattern_matching_query(
            self,
            graph_pattern: Dict[str, Any],
            top_k: int
    ) -> List[Dict[str, Any]]:
        """그래프 패턴 기반 검색 실행 (Full-Text Index 사용)

        각 조건마다 Full-Text 검색을 수행하고, 관계 가중치와 홉 거리를 반영한 점수 계산
        다중 조건 시 Python에서 결과 병합
        """
        conditions = graph_pattern.get("conditions", [])

        if not conditions:
            return []

        # 단일 조건인 경우 단순 쿼리
        if len(conditions) == 1:
            query_text = conditions[0].get("value", "") + "~"

            entity_query = """
            CALL db.index.fulltext.queryNodes('entity_fulltext', $query)
            YIELD node, score
            WHERE score > 0.3
            MATCH (d:Document)-[r:MENTIONS|HAS_TOPIC*1..2]-(node)
            WITH d,
                 score *
                 CASE type(r[0]) WHEN 'HAS_TOPIC' THEN 1.0 WHEN 'MENTIONS' THEN 0.5 ELSE 0.3 END *
                 CASE size(r) WHEN 1 THEN 1.0 WHEN 2 THEN 0.7 ELSE 0.5 END AS match_score
            RETURN d.id AS doc_id,
                   max(match_score) AS match_score
            ORDER BY match_score DESC
            LIMIT $top_k
            """

            topic_query = """
            CALL db.index.fulltext.queryNodes('topic_fulltext', $query)
            YIELD node, score
            WHERE score > 0.3
            MATCH (d:Document)-[r:MENTIONS|HAS_TOPIC*1..2]-(node)
            WITH d,
                 score *
                 CASE type(r[0]) WHEN 'HAS_TOPIC' THEN 1.0 WHEN 'MENTIONS' THEN 0.5 ELSE 0.3 END *
                 CASE size(r) WHEN 1 THEN 1.0 WHEN 2 THEN 0.7 ELSE 0.5 END AS match_score
            RETURN d.id AS doc_id,
                   max(match_score) AS match_score
            ORDER BY match_score DESC
            LIMIT $top_k
            """

            entity_results = self.graph_repo.execute_query(entity_query, {"query": query_text, "top_k": top_k})
            topic_results = self.graph_repo.execute_query(topic_query, {"query": query_text, "top_k": top_k})

            # 결과 병합 (doc_id 중복 제거)
            doc_scores = {}
            for result in entity_results + topic_results:
                doc_id = result["doc_id"]
                score = result["match_score"]
                if doc_id not in doc_scores or doc_scores[doc_id] < score:
                    doc_scores[doc_id] = score

            sorted_results = [
                {"doc_id": doc_id, "match_score": score, "matched_conditions": [0]}
                for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            ]

            return sorted_results[:top_k]

        # 다중 조건인 경우: 각 조건마다 별도 쿼리 후 Python에서 병합
        # (Cypher 스코프 문제 회피)
        params = {"top_k": top_k}

        all_results = []
        for idx, cond in enumerate(conditions):
            query_text = cond.get("value", "") + "~"

            # Entity 검색
            entity_query = """
            CALL db.index.fulltext.queryNodes('entity_fulltext', $query_text)
            YIELD node, score
            WHERE score > 0.3
            MATCH (d:Document)-[r:MENTIONS|HAS_TOPIC*1..2]-(node)
            WITH d, $cond_idx AS cond_idx,
                 score *
                 CASE type(r[0]) WHEN 'HAS_TOPIC' THEN 1.0 WHEN 'MENTIONS' THEN 0.5 ELSE 0.3 END *
                 CASE size(r) WHEN 1 THEN 1.0 WHEN 2 THEN 0.7 ELSE 0.5 END AS weighted_score
            RETURN d.id AS doc_id, cond_idx, max(weighted_score) AS score
            """

            # Topic 검색
            topic_query = """
            CALL db.index.fulltext.queryNodes('topic_fulltext', $query_text)
            YIELD node, score
            WHERE score > 0.3
            MATCH (d:Document)-[r:MENTIONS|HAS_TOPIC*1..2]-(node)
            WITH d, $cond_idx AS cond_idx,
                 score *
                 CASE type(r[0]) WHEN 'HAS_TOPIC' THEN 1.0 WHEN 'MENTIONS' THEN 0.5 ELSE 0.3 END *
                 CASE size(r) WHEN 1 THEN 1.0 WHEN 2 THEN 0.7 ELSE 0.5 END AS weighted_score
            RETURN d.id AS doc_id, cond_idx, max(weighted_score) AS score
            """

            entity_results = self.graph_repo.execute_query(
                entity_query,
                {"query_text": query_text, "cond_idx": idx}
            )
            topic_results = self.graph_repo.execute_query(
                topic_query,
                {"query_text": query_text, "cond_idx": idx}
            )

            all_results.extend(entity_results)
            all_results.extend(topic_results)

        # Python에서 결과 집계
        doc_scores = {}
        for result in all_results:
            doc_id = result["doc_id"]
            cond_idx = result["cond_idx"]
            score = result["score"]

            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    "doc_id": doc_id,
                    "match_score": 0,
                    "matched_conditions": set()
                }

            doc_scores[doc_id]["match_score"] += score
            doc_scores[doc_id]["matched_conditions"].add(cond_idx)

        # 정렬 및 상위 top_k 선택
        sorted_docs = sorted(
            [
                {
                    "doc_id": v["doc_id"],
                    "match_score": v["match_score"],
                    "matched_conditions": list(v["matched_conditions"])
                }
                for v in doc_scores.values()
                if len(v["matched_conditions"]) > 0
            ],
            key=lambda x: (x["match_score"], x["doc_id"]),
            reverse=True
        )

        return sorted_docs[:top_k]

    @staticmethod
    def _extract_doc_id(content) -> str | None:
        """Content에서 doc_id 추출 (Neo4j에서 가져온 검증된 데이터)"""
        if isinstance(content, dict):
            return content.get("id")

        if isinstance(content, str):
            try:
                content_dict = ast.literal_eval(content)
                return content_dict.get("id") if isinstance(content_dict, dict) else None
            except (ValueError, SyntaxError):
                match = re.search(r"'id':\s*'([^']+)'", content)
                return match.group(1) if match else None

        return None

    def _get_document_info(self, doc_id: str) -> Dict[str, Any] | None:
        """Document 상세 정보 조회"""
        result = self.graph_repo.execute_query(GET_DOCUMENT_INFO, {"doc_id": doc_id})

        if not result:
            return None

        return {
            "text": result[0].get("text", ""),
            "description": result[0].get("description", "")
        }

    def _get_related_documents(self, doc_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """문서의 연관 문서 조회

        Args:
            doc_id: 문서 ID
            limit: 최대 반환 개수

        Returns:
            연관 문서 목록
        """
        result = self.graph_repo.execute_query(GET_RELATED_DOCUMENTS, {"doc_id": doc_id, "limit": limit})

        return [
            {
                "target_doc_id": row.get("target_doc_id", ""),
                "target_description": row.get("target_description", ""),
                "relation_type": row.get("relation_type", ""),
                "reason": row.get("reason", ""),
                "confidence": row.get("confidence", 0.0)
            }
            for row in result
        ]

    def _extract_graph_structure(
            self,
            doc_ids: List[str],
            depth: int = 1
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Document와 연결된 그래프 구조 추출

        Args:
            doc_ids: Document ID 목록
            depth: 그래프 탐색 깊이

        Returns:
            그래프 구조 dict
        """
        query = self._build_graph_extraction_query(depth)
        result = self.graph_repo.execute_query(query, {"doc_ids": doc_ids})

        if not result:
            return {"nodes": [], "relationships": []}

        nodes = result[0].get("nodes", [])
        relationships = result[0].get("relationships", [])

        return {
            "nodes": self._format_nodes(nodes),
            "relationships": self._format_relationships(relationships)
        }

    @staticmethod
    def _build_graph_extraction_query(depth: int) -> str:
        """그래프 추출 Cypher 쿼리 생성"""
        return EXTRACT_GRAPH_2HOP if depth == 2 else EXTRACT_GRAPH_1HOP

    def _format_nodes(
            self,
            nodes: List[Dict[str, Any]],
            use_synthetic_id: bool = False
    ) -> List[Dict[str, Any]]:
        """노드 포맷팅"""
        formatted = []
        seen_ids = set()

        for node in nodes:
            if not node:
                continue

            labels = node.get("labels", [])
            props = node.get("properties", {})

            # synthetic ID 생성 (필요시)
            if use_synthetic_id:
                if "Entity" in labels:
                    node_id = f"entity_{props.get('name')}"
                elif "Category" in labels:
                    node_id = f"category_{props.get('name')}"
                elif "Topic" in labels:
                    node_id = f"topic_{props.get('name')}"
                else:
                    node_id = node.get("id")
            else:
                node_id = node.get("id")

            if not node_id or node_id in seen_ids:
                continue

            seen_ids.add(node_id)

            # properties 정리
            clean_props = self._sanitize_properties(props)
            clean_props.pop("graph_embedding", None)
            clean_props.pop("text_embedding", None)

            formatted.append({
                "id": node_id,
                "labels": labels,
                "properties": clean_props
            })

        return formatted

    def _format_relationships(
            self,
            relationships: List[Dict[str, Any]],
            id_mapping: Dict[str, str] = None
    ) -> List[Dict[str, Any]]:
        """관계 포맷팅

        Args:
            relationships: 원본 관계 목록
            id_mapping: 원본 ID -> synthetic ID 매핑 (옵셔널)

        Returns:
            포맷팅된 관계 목록
        """
        formatted = []
        seen = set()

        for rel in relationships:
            if not rel:
                continue

            rel_type = rel.get("type")
            start = rel.get("start")
            end = rel.get("end")

            if not start or not end or not rel_type:
                continue

            # ID 매핑 적용 (있으면)
            if id_mapping:
                start = id_mapping.get(start, start)
                end = id_mapping.get(end, end)

            key = (rel_type, start, end)
            if key in seen:
                continue

            seen.add(key)

            # properties 정리
            clean_props = self._sanitize_properties(rel.get("properties", {}))

            # 프론트엔드 호환성을 위해 주요 속성을 루트 레벨로 끌어올림
            rel_data = {
                "type": rel_type,
                "start": start,
                "end": end,
                "properties": clean_props
            }

            # reason, confidence, source, via 등 중요 속성은 루트 레벨에도 추가
            if "reason" in clean_props:
                rel_data["reason"] = clean_props["reason"]
            if "confidence" in clean_props:
                rel_data["confidence"] = clean_props["confidence"]
            if "source" in clean_props:
                rel_data["source"] = clean_props["source"]
            if "via" in clean_props:
                rel_data["via"] = clean_props["via"]
            if "wikidata_type" in clean_props:
                rel_data["wikidata_type"] = clean_props["wikidata_type"]

            formatted.append(rel_data)

        return formatted

    @staticmethod
    def _aggregate_metadata_counts(
            results: List[Dict[str, Any]],
            field_name: str,
            total_docs: int
    ) -> List[Dict[str, Any]]:
        """메타데이터 카운트 집계

        Args:
            results: 쿼리 결과 목록
            field_name: 집계할 필드명 ("categories" or "tags")
            total_docs: 전체 문서 수

        Returns:
            집계된 메타데이터 목록
        """
        counts = {}
        for row in results:
            for item in row.get(field_name, []):
                if item:
                    counts[item] = counts.get(item, 0) + 1

        return [
            {
                "name": name,
                "doc_count": count,
                "coverage": round(count / total_docs, 2)
            }
            for name, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)
        ]

    def _calculate_common_metadata(self, doc_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """검색된 Document들의 공통 Category/Entity 집계"""
        results = self.graph_repo.execute_query(CALCULATE_COMMON_METADATA, {"doc_ids": doc_ids})
        if not results:
            return {"categories": [], "common_tags": []}

        total_docs = len(doc_ids)
        categories = self._aggregate_metadata_counts(results, "categories", total_docs)
        common_tags = self._aggregate_metadata_counts(results, "entities", total_docs)

        logging.info(f"공통 메타데이터 집계: Category {len(categories)}개, Entity {len(common_tags)}개")

        return {"categories": categories, "common_tags": common_tags}

    def _get_document_nodes(self, doc_ids: List[str]) -> List[Dict[str, Any]]:
        """Document 노드 조회"""
        doc_results = self.graph_repo.execute_query(GET_DOCUMENT_NODES, {"doc_ids": doc_ids})

        return [
            {
                "id": doc["id"],
                "labels": ["Document"],
                "properties": {
                    "id": doc["id"],
                    "description": doc.get("description", "")
                }
            }
            for doc in doc_results
        ]

    @staticmethod
    def _filter_common_metadata(
            metadata_list: List[Dict[str, Any]],
            min_coverage: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Coverage 기준으로 메타데이터 필터링"""
        return [item for item in metadata_list if item.get("coverage", 0) >= min_coverage]

    @staticmethod
    def _create_metadata_nodes(
            categories: List[Dict[str, Any]],
            entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Category/Entity 노드 생성"""
        nodes = []

        for cat in categories:
            nodes.append({
                "id": f"category_{cat['name']}",
                "labels": ["Category"],
                "properties": {
                    "name": cat["name"],
                    "doc_count": cat["doc_count"],
                    "coverage": cat["coverage"]
                }
            })

        for entity in entities:
            nodes.append({
                "id": f"entity_{entity['name']}",
                "labels": ["Entity"],
                "properties": {
                    "name": entity["name"],
                    "doc_count": entity["doc_count"],
                    "coverage": entity["coverage"]
                }
            })

        return nodes

    def _get_category_relationships(
            self,
            doc_ids: List[str],
            category_names: List[str]
    ) -> List[Dict[str, Any]]:
        """Document-Category 관계 조회

        Args:
            doc_ids: Document ID 목록
            category_names: Category 이름 목록

        Returns:
            관계 목록
        """
        if not category_names:
            return []

        cat_rels = self.graph_repo.execute_query(
            GET_CATEGORY_RELATIONSHIPS,
            {"doc_ids": doc_ids, "category_names": category_names}
        )

        return [
            {
                "type": "BELONGS_TO",
                "start": rel["doc_id"],
                "end": f"category_{rel['category_name']}",
                "properties": {}
            }
            for rel in cat_rels
        ]

    def _get_entity_relationships(
            self,
            doc_ids: List[str],
            entity_names: List[str]
    ) -> List[Dict[str, Any]]:
        """Document-Entity 관계 조회

        Args:
            doc_ids: Document ID 목록
            entity_names: Entity 이름 목록

        Returns:
            관계 목록
        """
        if not entity_names:
            return []

        entity_rels = self.graph_repo.execute_query(
            GET_ENTITY_RELATIONSHIPS,
            {"doc_ids": doc_ids, "entity_names": entity_names}
        )

        return [
            {
                "type": "MENTIONS",
                "start": rel["doc_id"],
                "end": f"entity_{rel['entity_name']}",
                "properties": {}
            }
            for rel in entity_rels
        ]

    def _build_simple_graph(
            self,
            doc_ids: List[str],
            common_metadata: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """간단한 그래프 생성: Document + 공통 Category/Entity만

        Args:
            doc_ids: Document ID 목록
            common_metadata: 공통 메타데이터

        Returns:
            그래프 구조 dict
        """
        nodes = self._get_document_nodes(doc_ids)

        common_categories = self._filter_common_metadata(common_metadata.get("categories", []))
        common_tags = self._filter_common_metadata(common_metadata.get("common_tags", []))

        nodes.extend(self._create_metadata_nodes(common_categories, common_tags))

        category_names = [cat["name"] for cat in common_categories]
        entity_names = [entity["name"] for entity in common_tags]

        relationships = []
        relationships.extend(self._get_category_relationships(doc_ids, category_names))
        relationships.extend(self._get_entity_relationships(doc_ids, entity_names))

        logging.info(f"간단한 그래프 생성: {len(nodes)}개 노드, {len(relationships)}개 관계")

        return {"nodes": nodes, "relationships": relationships}

    @staticmethod
    def _build_explore_query(node_id: str) -> tuple[str, Dict[str, Any]]:
        """노드 탐색 쿼리 생성"""
        if node_id.startswith("entity_"):
            match_clause = "MATCH (center:Entity {name: $name})"
            params = {"name": node_id[7:]}
        elif node_id.startswith("category_"):
            match_clause = "MATCH (center:Category {name: $name})"
            params = {"name": node_id[9:]}
        elif node_id.startswith("topic_"):
            match_clause = "MATCH (center:Topic {name: $name})"
            params = {"name": node_id[6:]}
        else:
            match_clause = "MATCH (center) WHERE center.id = $node_id OR elementId(center) = $node_id"
            params = {"node_id": node_id}

        return EXPLORE_NODE_TEMPLATE.format(match_clause=match_clause), params

    @staticmethod
    def _build_id_mapping(nodes: List[Dict[str, Any]]) -> Dict[str, str]:
        """노드 ID 매핑 생성 (원본 ID -> synthetic ID)"""
        id_mapping = {}
        for node in nodes:
            if not node:
                continue

            labels = node.get("labels", [])
            props = node.get("properties", {})
            original_id = node.get("id")

            if "Entity" in labels:
                synthetic_id = f"entity_{props.get('name')}"
                id_mapping[original_id] = synthetic_id
                id_mapping[props.get('name')] = synthetic_id
            elif "Category" in labels:
                synthetic_id = f"category_{props.get('name')}"
                id_mapping[original_id] = synthetic_id
                id_mapping[props.get('name')] = synthetic_id
            elif "Topic" in labels:
                synthetic_id = f"topic_{props.get('name')}"
                id_mapping[original_id] = synthetic_id
                id_mapping[props.get('name')] = synthetic_id
            else:
                id_mapping[original_id] = original_id

        return id_mapping

    def _get_connected_documents(self, nodes: List[Dict[str, Any]], exclude_id: str) -> List[Dict[str, Any]]:
        """연결된 Document 정보 조회

        Args:
            nodes: 포맷팅된 노드 목록
            exclude_id: 제외할 노드 ID (중심 노드)

        Returns:
            연결된 Document 목록
        """
        doc_ids = [
            node["id"] for node in nodes
            if "Document" in node.get("labels", []) and node["id"] != exclude_id
        ]

        if not doc_ids:
            return []

        doc_results = self.graph_repo.execute_query(GET_DOCUMENTS_BY_IDS, {"doc_ids": doc_ids})

        return [
            {
                "doc_id": doc["doc_id"],
                "text": doc.get("text", ""),
                "description": doc.get("description", ""),
                "score": 0.0
            }
            for doc in doc_results
        ]

    def explore_node(self, node_id: str) -> Dict[str, Any]:
        """노드 중심 1-hop 그래프 탐색

        Args:
            node_id: 탐색할 노드 ID

        Returns:
            NodeExploreResponse를 dict로 변환한 결과
        """
        try:
            center_node = self._get_node_info(node_id)
            if not center_node:
                return NodeExploreResponse.empty(
                    error=f"노드를 찾을 수 없습니다: {node_id}"
                ).model_dump()

            graph_query, query_params = self._build_explore_query(node_id)
            result = self.graph_repo.execute_query(graph_query, query_params)

            if not result:
                return NodeExploreResponse.empty(
                    center_node=GraphNode(**center_node)
                ).model_dump()

            raw_nodes = result[0].get("nodes", [])
            raw_relationships = result[0].get("relationships", [])

            id_mapping = self._build_id_mapping(raw_nodes)
            nodes = self._format_nodes(raw_nodes, use_synthetic_id=True)
            relationships = self._format_relationships(raw_relationships, id_mapping=id_mapping)
            connected_documents = self._get_connected_documents(nodes, node_id)

            logging.info(
                f"노드 탐색 완료: {node_id}, "
                f"{len(connected_documents)}개 연결 Document, "
                f"{len(nodes)}개 노드, {len(relationships)}개 관계"
            )

            response = NodeExploreResponse(
                center_node=GraphNode(**center_node),
                connected_documents=connected_documents,
                graph=GraphStructure(nodes=nodes, relationships=relationships)
            )
            return response.model_dump()

        except (ServiceUnavailable, ClientError) as e:
            logging.exception(f"노드 탐색 실패 (node_id: {node_id}): {e}")
            return NodeExploreResponse.empty(error=str(e)).model_dump()

    def _get_node_info(self, node_id: str) -> Dict[str, Any] | None:
        """노드 정보 조회

        Args:
            node_id: 노드 ID (Document ID 또는 entity_xxx, category_xxx, topic_xxx 형식)

        Returns:
            노드 정보 또는 None
        """
        if node_id.startswith("entity_"):
            query, params = GET_NODE_BY_ENTITY, {"name": node_id[7:]}
        elif node_id.startswith("category_"):
            query, params = GET_NODE_BY_CATEGORY, {"name": node_id[9:]}
        elif node_id.startswith("topic_"):
            query, params = GET_NODE_BY_TOPIC, {"name": node_id[6:]}
        else:
            query, params = GET_NODE_BY_ID, {"node_id": node_id}

        result = self.graph_repo.execute_query(query, params)
        if not result:
            return None

        node = result[0]
        final_id = node_id if node_id.startswith(("entity_", "category_", "topic_")) else (
                node.get("properties", {}).get("id") or node.get("id"))

        return {
            "id": final_id,
            "labels": node.get("labels", []),
            "properties": self._sanitize_properties(node.get("properties", {}))
        }
