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
    EXTRACT_GRAPH_1HOP,
    EXTRACT_GRAPH_2HOP,
    EXTRACT_GRAPH_FROM_VALUES,
    GET_NODE_BY_ID,
    GET_DOCUMENTS_BY_IDS
)
from app.repositories.graph_repository import GraphRepository
from app.schemas.graphrag_api import (
    GraphRAGSearchResponse,
    GraphStructure,
    CommonMetadata
)
from app.src.graphrag.graph_pattern_matcher import GraphPatternMatcher
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
        self.pattern_matcher = GraphPatternMatcher(graph_repo)

    @property
    def reranker(self) -> BGEReranker:
        """Reranker lazy initialization (첫 사용 시에만 로딩)"""
        if self._reranker is None:
            self._reranker = BGEReranker()
        return self._reranker

    @staticmethod
    def _parse_node_id(node_id: str) -> tuple[str, str | None]:
        """Synthetic ID를 파싱하여 실제 ID와 타입 추출

        Args:
            node_id: 노드 ID (synthetic ID일 수 있음)

        Returns:
            (actual_id, node_type_hint): 실제 ID와 타입 힌트

        Examples:
            "Key:병상수" -> ("병상수", "Key")
            "Value:doc_001_병상수_100" -> ("doc_001_병상수_100", "Value")
            "doc_001" -> ("doc_001", None)
        """
        if ":" in node_id:
            parts = node_id.split(":", 1)
            if len(parts) == 2 and parts[0] in ["Key", "Value", "NormalizedValue", "Centrality"]:
                return parts[1], parts[0]
        return node_id, None

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
            top_k: int = 5,
            use_graph_pattern: bool = True
    ) -> Dict[str, Any]:
        """사용자 질의 기반 Graph RAG 검색 (Key-Value 구조)

        Args:
            query: 사용자 쿼리
            top_k: 반환할 Document 개수
            use_graph_pattern: Graph Pattern Matching 사용 여부 (기본: True)
        """
        try:
            # 1. Graph Pattern Matching 시도
            value_ids = []
            if use_graph_pattern:
                logging.info(f"🔍 Query: '{query}'")
                search_result = self.pattern_matcher.search_by_graph_pattern(query, top_k * 2)
                doc_ids = search_result.get("doc_ids", [])
                matched_values = search_result.get("matched_values", [])
                value_ids = [v["value_id"] for v in matched_values]

                if doc_ids:
                    logging.info(f"✓ Pattern matching: {len(doc_ids)}개 문서, {len(value_ids)}개 Value")
                    # matched_values에서 doc_id별 최고 점수 추출
                    doc_scores = self._extract_doc_scores(doc_ids[:top_k], matched_values)
                    documents = self._get_documents_by_ids(doc_ids[:top_k], doc_scores)
                else:
                    logging.info(f"  └─ Pattern matching 실패, 벡터 검색으로 fallback")
                    documents = self._search_documents(query, top_k)
            else:
                logging.info(f"🔍 Query: '{query}' (벡터 검색 모드)")
                documents = self._search_documents(query, top_k)

            if not documents:
                return GraphRAGSearchResponse.empty(query).model_dump()

            # 2. 그래프 구성: Value 중심으로 (검색 조건이 있으면)
            if value_ids:
                graph = self._extract_graph_from_values(value_ids[:top_k])
            else:
                # Fallback: Document 중심 그래프
                doc_ids = [doc["doc_id"] for doc in documents]
                graph = self._extract_graph_structure(doc_ids, depth=1)

            logging.info(
                f"✓ 검색 완료: {len(documents)}개 문서, {len(graph['nodes'])}개 노드, {len(graph['relationships'])}개 관계"
            )

            response = GraphRAGSearchResponse(
                query=query,
                documents=documents,
                common_metadata=CommonMetadata(categories=[], common_tags=[]),
                graph=GraphStructure(**graph)
            )
            return response.model_dump()

        except (ServiceUnavailable, TransientError, ClientError, ValueError, KeyError, AttributeError) as e:
            logging.exception(f"Graph RAG 검색 실패: {e}")
            return GraphRAGSearchResponse.empty(query, error=str(e)).model_dump()

    def _search_documents(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """텍스트 임베딩 + Reranker + LLM 검증 기반 검색"""
        # 1. 텍스트 임베딩 기반 검색
        vector_results = self._search_by_text_embedding(query, top_k * 10)
        if not vector_results:
            return []

        # 2. Document 정보 추가
        enriched_results = self._enrich_vector_results(vector_results)

        # 3. 경량 Reranker 필터링 (BGE Reranker)
        reranked_results = self.reranker.rerank(query, enriched_results, top_k * 3)
        logging.info(f"→ Reranker: {len(enriched_results)}개 → {len(reranked_results)}개")

        # 4. LLM 검증 (상위 후보만, early stopping 적용)
        verified_results = self._verify_results_with_llm(query, reranked_results, top_k)
        logging.info(f"→ LLM 검증: {len(verified_results)}개 통과")

        return verified_results[:top_k]

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

        for result in results:
            # Early stopping: top_k개 확보하면 중단
            if len(verified) >= top_k:
                break

            try:
                # 원본 text 전체 사용 (최대 10000자)
                document = result.get("text", "")
                if len(document) > 10000:
                    document = document[:10000]

                # LLM 검증
                verification = LLMResponse("document_relevance_verification").generate_llm_response(
                    variables={"query": query, "document": document}
                )

                if isinstance(verification, dict):
                    result["llm_relevant"] = verification.get("is_relevant", False)
                    result["llm_reason"] = verification.get("reason", "")

                    if result["llm_relevant"]:
                        verified.append(result)
                else:
                    result["llm_relevant"] = True
                    verified.append(result)

            except Exception as e:
                logging.error(f"LLM 검증 중 에러: {e}")
                result["llm_relevant"] = True
                verified.append(result)

        return verified

    def _enrich_vector_results(self, vector_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """벡터 검색 결과에 Document 정보 추가"""
        enriched = []

        for r in vector_results:
            doc_id = r["doc_id"]
            doc_info = self._get_document_info(doc_id)

            if not doc_info:
                continue

            enriched.append({
                "doc_id": doc_id,
                "text": doc_info.get("text", ""),
                "description": doc_info.get("description", ""),
                "score": r.get("vector_score", 0),
                "vector_score": r.get("vector_score", 0)
            })

        return enriched

    def _search_by_text_embedding(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """텍스트 임베딩 기반 검색"""
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

    def _extract_doc_scores(
            self,
            doc_ids: List[str],
            matched_values: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """matched_values에서 doc_id별 최고 점수 추출

        Args:
            doc_ids: Document ID 목록
            matched_values: Value 매칭 결과 (value_id, score 포함)

        Returns:
            {doc_id: max_score} 매핑
        """
        # Value ID에서 Document ID 매핑 조회
        value_ids = [v["value_id"] for v in matched_values]
        if not value_ids:
            return {doc_id: 1.0 for doc_id in doc_ids}

        query = """
        UNWIND $value_ids as vid
        MATCH (v:Value {id: vid})-[:EXTRACTED_FROM]->(d:Document)
        RETURN v.id as value_id, d.id as doc_id
        """

        try:
            results = self.graph_repo.execute_query(query, {"value_ids": value_ids})
            value_to_doc = {row["value_id"]: row["doc_id"] for row in results}

            # doc_id별 최고 점수 계산
            doc_scores: Dict[str, float] = {}
            for value_info in matched_values:
                value_id = value_info["value_id"]
                score = value_info.get("score", 1.0)
                doc_id = value_to_doc.get(value_id)

                if doc_id and doc_id in doc_ids:
                    doc_scores[doc_id] = max(doc_scores.get(doc_id, 0), score)

            # 매핑되지 않은 doc_id는 기본값 1.0
            for doc_id in doc_ids:
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 1.0

            return doc_scores

        except Exception as e:
            logging.warning(f"점수 추출 실패: {e}, 기본값 1.0 사용")
            return {doc_id: 1.0 for doc_id in doc_ids}

    def _get_documents_by_ids(
            self,
            doc_ids: List[str],
            doc_scores: Dict[str, float] = None
    ) -> List[Dict[str, Any]]:
        """Document ID 리스트로 Document 정보 조회 (Graph Pattern Matching용)

        Args:
            doc_ids: Document ID 목록
            doc_scores: doc_id별 점수 (없으면 1.0)
        """
        if not doc_ids:
            return []

        results = self.graph_repo.execute_query(
            GET_DOCUMENTS_BY_IDS,
            {"doc_ids": doc_ids}
        )

        return [
            {
                "doc_id": row["doc_id"],
                "text": row.get("text", ""),
                "description": row.get("description", ""),
                "score": doc_scores.get(row["doc_id"], 1.0) if doc_scores else 1.0
            }
            for row in results
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

    def _extract_graph_from_values(
            self,
            value_ids: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Value 노드를 중심으로 그래프 구조 추출

        Args:
            value_ids: Value ID 목록 (검색 조건에 매칭된 Value들)

        Returns:
            그래프 구조 dict
        """
        result = self.graph_repo.execute_query(EXTRACT_GRAPH_FROM_VALUES, {"value_ids": value_ids})

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
        """노드 포맷팅 (Key-Value 구조)"""
        formatted = []
        seen_ids = set()

        for node in nodes:
            if not node:
                continue

            labels = node.get("labels", [])
            props = node.get("properties", {})

            # synthetic ID 생성 (필요시)
            if use_synthetic_id:
                if "Key" in labels or "Value" in labels or "Centrality" in labels:
                    node_id = f"{labels[0].lower()}_{props.get('name', props.get('id', ''))}"
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

    def explore_node_onehop(self, node_id: str) -> Dict[str, Any]:
        """노드 ID를 중심으로 1-hop 관계 탐색 (노드 타입별 다른 동작)

        Args:
            node_id: 탐색할 중심 노드 ID (synthetic ID일 수 있음: "Key:xxx", "Value:xxx")

        Returns:
            {
                "center_node": {...},
                "connected_documents": [...],
                "graph": {"nodes": [...], "relationships": [...]}
            }

        노드 타입별 동작:
            - Document: Document <- Value, Value <- Key
            - Value: Value <- Key, Value -> NormalizedValue
            - Key: Key -> Value, Key -> Centrality
            - NormalizedValue: NormalizedValue <- Document
        """
        try:
            logging.info(f"🔍 Explore: node_id='{node_id}'")

            # Synthetic ID 파싱 (예: "Key:병상수" -> "병상수")
            actual_node_id, node_type_hint = self._parse_node_id(node_id)
            logging.info(f"  ├─ Parsed ID: '{actual_node_id}'")
            logging.info(f"  └─ Type hint: {node_type_hint}")

            # 1. 중심 노드 정보 조회 (실제 ID 또는 name으로)
            center_node_result = self.graph_repo.execute_query(
                GET_NODE_BY_ID,
                {"node_id": actual_node_id}
            )

            if not center_node_result:
                logging.warning(f"  └─ 노드를 찾을 수 없음: {actual_node_id}")
                return {
                    "center_node": None,
                    "connected_documents": [],
                    "graph": {"nodes": [], "relationships": []}
                }

            center_node = dict(center_node_result[0])
            center_labels = center_node.get("labels", [])
            node_type = node_type_hint or (center_labels[0] if center_labels else None)

            logging.info(f"  └─ 노드 타입: {node_type}")

            # 2. 노드 타입별 쿼리 선택 (실제 ID 사용)
            if node_type == "Document":
                query = self._build_document_explore_query()
            elif node_type == "Value":
                query = self._build_value_explore_query()
            elif node_type == "Key":
                query = self._build_key_explore_query()
            elif node_type == "NormalizedValue":
                query = self._build_normalizedvalue_explore_query()
            else:
                # 기본 동작: 모든 1-hop 연결
                query = self._build_default_explore_query()

            result = self.graph_repo.execute_query(query, {"node_id": actual_node_id})

            if not result:
                return {
                    "center_node": center_node,
                    "connected_documents": [],
                    "graph": {"nodes": [center_node], "relationships": []}
                }

            # 3. 결과 처리
            result_dict = dict(result[0])
            connected_nodes = result_dict.get("connected_nodes", [])
            relationships = result_dict.get("relationships", [])

            logging.info(f"  ├─ 연결된 노드: {len(connected_nodes)}개")
            logging.info(f"  └─ 관계: {len(relationships)}개")

            # 4. 노드 변환
            all_nodes = self._convert_nodes_to_dicts([center_node] + connected_nodes)

            logging.info(f"  ├─ 변환된 노드: {len(all_nodes)}개")
            for node in all_nodes:
                labels = node.get('labels', [])
                label = labels[0] if labels else 'Unknown'
                logging.info(f"      - {label}: id={node.get('id', 'N/A')}, props keys={list(node.get('properties', {}).keys())}")

            # 5. Document 목록 수집
            doc_ids = [
                n["id"] for n in all_nodes
                if "Document" in n.get("labels", []) and n["id"]
            ]
            documents = self._get_documents_by_ids(doc_ids) if doc_ids else []

            # 6. 관계 변환
            formatted_relationships = [
                rel for rel in relationships
                if isinstance(rel, dict) and rel.get("type") and rel.get("start") and rel.get("end")
            ]

            # 7. 그래프 구조 포맷팅
            formatted_graph_nodes = self._format_nodes(all_nodes)

            logging.info(f"  ├─ 포맷팅된 노드: {len(formatted_graph_nodes)}개")
            for node in formatted_graph_nodes:
                labels = node.get('labels', [])
                label = labels[0] if labels else 'Unknown'
                logging.info(f"      - {label}: id={node.get('id', 'N/A')}, has name={('name' in node.get('properties', {}))}")

            graph_structure = {
                "nodes": formatted_graph_nodes,
                "relationships": self._format_relationships(formatted_relationships)
            }

            logging.info(
                f"✓ Explore: {len(formatted_graph_nodes)}개 노드, {len(formatted_relationships)}개 관계 반환"
            )

            return {
                "center_node": center_node,
                "connected_documents": documents,
                "graph": graph_structure
            }

        except (ServiceUnavailable, TransientError, ClientError, ValueError, KeyError, AttributeError) as e:
            logging.exception(f"1-hop 탐색 실패: {e}")
            return {
                "center_node": None,
                "connected_documents": [],
                "graph": {"nodes": [], "relationships": []},
                "error": str(e)
            }

    @staticmethod
    def _build_document_explore_query() -> str:
        """Document 노드 탐색 쿼리: Document <- Value <- Key"""
        return """
        MATCH (center:Document)
        WHERE elementId(center) = $node_id OR center.id = $node_id

        OPTIONAL MATCH (value:Value)-[r1:EXTRACTED_FROM]->(center)
        OPTIONAL MATCH (key:Key)-[r2:HAS_VALUE]->(value)

        WITH center,
             collect(DISTINCT value) + collect(DISTINCT key) as connected_nodes,
             collect(DISTINCT r1) + collect(DISTINCT r2) as rels

        RETURN center,
               [node IN connected_nodes WHERE node IS NOT NULL | {
                   id: COALESCE(node.id, node.name),
                   labels: labels(node),
                   properties: properties(node)
               }] as connected_nodes,
               [rel IN rels WHERE rel IS NOT NULL | {
                   type: type(rel),
                   start: COALESCE(startNode(rel).id, startNode(rel).name),
                   end: COALESCE(endNode(rel).id, endNode(rel).name),
                   properties: properties(rel)
               }] as relationships
        """

    @staticmethod
    def _build_value_explore_query() -> str:
        """Value 노드 탐색 쿼리: Value <- Key, Value -> NormalizedValue"""
        return """
        MATCH (center:Value)
        WHERE elementId(center) = $node_id OR center.id = $node_id OR center.name = $node_id

        OPTIONAL MATCH (key:Key)-[r1:HAS_VALUE]->(center)
        OPTIONAL MATCH (center)-[r2:SAME_AS]->(nv:NormalizedValue)

        WITH center,
             collect(DISTINCT key) + collect(DISTINCT nv) as connected_nodes,
             collect(DISTINCT r1) + collect(DISTINCT r2) as rels

        RETURN center,
               [node IN connected_nodes WHERE node IS NOT NULL | {
                   id: COALESCE(node.id, node.name),
                   labels: labels(node),
                   properties: properties(node)
               }] as connected_nodes,
               [rel IN rels WHERE rel IS NOT NULL | {
                   type: type(rel),
                   start: COALESCE(startNode(rel).id, startNode(rel).name),
                   end: COALESCE(endNode(rel).id, endNode(rel).name),
                   properties: properties(rel)
               }] as relationships
        """

    @staticmethod
    def _build_key_explore_query() -> str:
        """Key 노드 탐색 쿼리: Key -> Value만 (Centrality 관계 없음)"""
        return """
        MATCH (center:Key)
        WHERE elementId(center) = $node_id OR center.name = $node_id

        OPTIONAL MATCH (center)-[r:HAS_VALUE]->(value:Value)

        WITH center,
             collect(DISTINCT value) as connected_nodes,
             collect(DISTINCT r) as rels

        RETURN center,
               [node IN connected_nodes WHERE node IS NOT NULL | {
                   id: COALESCE(node.id, node.name),
                   labels: labels(node),
                   properties: properties(node)
               }] as connected_nodes,
               [rel IN rels WHERE rel IS NOT NULL | {
                   type: type(rel),
                   start: COALESCE(startNode(rel).id, startNode(rel).name),
                   end: COALESCE(endNode(rel).id, endNode(rel).name),
                   properties: properties(rel)
               }] as relationships
        """

    @staticmethod
    def _build_normalizedvalue_explore_query() -> str:
        """NormalizedValue 노드 탐색 쿼리: NormalizedValue <- Value <- Document"""
        return """
        MATCH (center:NormalizedValue)
        WHERE elementId(center) = $node_id OR center.name = $node_id

        OPTIONAL MATCH (value:Value)-[r1:SAME_AS]->(center)
        OPTIONAL MATCH (value)-[r2:EXTRACTED_FROM]->(doc:Document)

        WITH center,
             collect(DISTINCT value) + collect(DISTINCT doc) as connected_nodes,
             collect(DISTINCT r1) + collect(DISTINCT r2) as rels

        RETURN center,
               [node IN connected_nodes WHERE node IS NOT NULL | {
                   id: COALESCE(node.id, node.name),
                   labels: labels(node),
                   properties: properties(node)
               }] as connected_nodes,
               [rel IN rels WHERE rel IS NOT NULL | {
                   type: type(rel),
                   start: COALESCE(startNode(rel).id, startNode(rel).name),
                   end: COALESCE(endNode(rel).id, endNode(rel).name),
                   properties: properties(rel)
               }] as relationships
        """

    @staticmethod
    def _build_default_explore_query() -> str:
        """기본 탐색 쿼리: 모든 1-hop 연결"""
        return """
        MATCH (center)
        WHERE elementId(center) = $node_id OR center.id = $node_id OR center.name = $node_id

        OPTIONAL MATCH (center)-[r]-(connected)

        WITH center,
             collect(DISTINCT connected) as connected_nodes,
             collect(DISTINCT {
                 type: type(r),
                 start: COALESCE(startNode(r).id, startNode(r).name),
                 end: COALESCE(endNode(r).id, endNode(r).name),
                 properties: properties(r)
             }) as relationships

        RETURN center,
               connected_nodes,
               relationships
        """

    def _convert_nodes_to_dicts(self, nodes: List[Any]) -> List[Dict[str, Any]]:
        """Neo4j Node 객체들을 dict로 변환"""
        result = []
        for node in nodes:
            if not node:
                continue

            if hasattr(node, 'labels'):
                # Neo4j Node 객체
                labels = list(node.labels)
                props = dict(node)
                node_id = props.get("id") or props.get("name")

                result.append({
                    "id": node_id,
                    "labels": labels,
                    "properties": props
                })
                logging.debug(f"  [Neo4j Node] {labels[0] if labels else 'Unknown'}: id={node_id}")
            elif isinstance(node, dict):
                # 이미 dict인 경우
                labels = node.get("labels", [])
                props = node.get("properties", {})
                node_id = props.get("id") or props.get("name") or node.get("id")

                result.append({
                    "id": node_id,
                    "labels": labels,
                    "properties": props
                })
                logging.debug(f"  [Dict Node] {labels[0] if labels else 'Unknown'}: id={node_id}")

        return result

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

            # dict가 아닌 경우 스킵
            if not isinstance(rel, dict):
                logging.warning(f"[_format_relationships] dict가 아닌 관계: {type(rel)}")
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
