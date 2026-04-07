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
                    documents = self._get_documents_by_ids(doc_ids[:top_k])
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

    def _get_documents_by_ids(self, doc_ids: List[str]) -> List[Dict[str, Any]]:
        """Document ID 리스트로 Document 정보 조회 (Graph Pattern Matching용)"""
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
                "score": 1.0  # Pattern matching은 binary (match or not)
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
        """노드 ID를 중심으로 1-hop 관계만 탐색

        Args:
            node_id: 탐색할 중심 노드 ID

        Returns:
            {
                "center_node": {...},
                "connected_documents": [...],
                "graph": {"nodes": [...], "relationships": [...]}
            }
        """
        try:
            logging.info(f"🔍 Explore: node_id='{node_id}'")

            # 1. 중심 노드 정보 조회
            center_node_result = self.graph_repo.execute_query(
                GET_NODE_BY_ID,
                {"node_id": node_id}
            )

            if not center_node_result:
                return {
                    "center_node": None,
                    "connected_documents": [],
                    "graph": {"nodes": [], "relationships": []}
                }

            # Neo4j Result를 dict로 변환
            center_node = dict(center_node_result[0])

            # 2. 1-hop 연결 노드 및 관계 조회
            query = """
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
                   relationships,
                   connected_nodes
            """

            logging.info(f"  └─ Cypher: MATCH (center) WHERE elementId(center) = '{node_id}' ... OPTIONAL MATCH (center)-[r]-(connected)")
            result = self.graph_repo.execute_query(query, {"node_id": node_id})

            if not result:
                return {
                    "center_node": center_node,
                    "connected_documents": [],
                    "graph": {"nodes": [center_node], "relationships": []}
                }

            # 3. 연결된 노드 추출 및 변환
            result_dict = dict(result[0])
            connected_nodes = result_dict.get("connected_nodes", [])
            relationships = result_dict.get("relationships", [])

            logging.info(f"  ├─ 연결된 노드: {len(connected_nodes)}개")
            logging.info(f"  └─ 관계: {len(relationships)}개")

            doc_ids = []
            all_nodes = []

            # Center 노드 추가
            center_labels = center_node.get("labels", [])
            center_props = center_node.get("properties", {})
            center_id = center_props.get("id") or center_props.get("name") or center_node.get("id")

            all_nodes.append({
                "id": center_id,
                "labels": center_labels,
                "properties": center_props
            })

            # 연결된 노드들 추가
            for node in connected_nodes:
                if not node:
                    continue

                # Neo4j Node 객체를 dict로 변환
                if hasattr(node, 'labels'):
                    labels = list(node.labels)
                    props = dict(node)
                    node_id = props.get("id") or props.get("name")

                    node_dict = {
                        "id": node_id,
                        "labels": labels,
                        "properties": props
                    }
                else:
                    # 이미 dict인 경우
                    node_dict = node
                    labels = node_dict.get("labels", [])
                    node_id = node_dict.get("properties", {}).get("id") or node_dict.get("properties", {}).get("name")

                all_nodes.append(node_dict)

                # Document 노드인 경우 doc_id 수집
                if "Document" in labels:
                    if node_id:
                        doc_ids.append(node_id)

            # 4. Document 상세 정보 조회
            documents = []
            if doc_ids:
                documents = self._get_documents_by_ids(doc_ids)

            # 5. 관계 변환 (이미 dict로 변환됨)
            formatted_relationships = []
            for rel in relationships:
                if not rel:
                    continue

                # 쿼리에서 이미 dict로 변환했으므로 그대로 사용
                if isinstance(rel, dict):
                    # null 값 필터링
                    if rel.get("type") and rel.get("start") and rel.get("end"):
                        formatted_relationships.append(rel)
                        logging.debug(f"    관계: {rel.get('start')} -[{rel.get('type')}]-> {rel.get('end')}")
                else:
                    logging.warning(f"    dict가 아닌 관계: {type(rel)}")

            # 6. 그래프 구조 포맷팅
            graph_structure = {
                "nodes": self._format_nodes(all_nodes),
                "relationships": self._format_relationships(formatted_relationships)
            }

            logging.info(
                f"✓ Explore: {len(all_nodes)}개 노드, {len(formatted_relationships)}개 관계"
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
