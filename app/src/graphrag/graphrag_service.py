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
    GET_TAG_RELATIONSHIPS,
    GET_NODE_BY_TAG,
    GET_NODE_BY_CATEGORY,
    GET_NODE_BY_ID,
    EXPLORE_NODE_TEMPLATE,
    GET_DOCUMENT_DETAILS
)
from app.repositories.graph_repository import GraphRepository
from app.schemas.graphrag_api import (
    GraphRAGSearchResponse,
    NodeExploreResponse,
    GraphStructure,
    CommonMetadata,
    GraphNode
)


class GraphRAGService:
    """Graph RAG 검색 서비스 - 사용자 질의 기반 그래프 구조 반환"""

    def __init__(self, graph_repo: GraphRepository):
        self.graph_repo = graph_repo
        self.driver = graph_repo.driver
        self.embedder = SentenceTransformerEmbeddings(model=settings.EMBEDDING_MODEL_NAME)

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
        """텍스트 임베딩 기반 Document 검색"""
        try:
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

                doc_info = self._get_document_info(doc_id)
                if not doc_info:
                    continue

                score = item.metadata.get("score", 0) if hasattr(item, 'metadata') and item.metadata else 0
                related_docs = self._get_related_documents(doc_id, limit=5)

                documents.append({
                    "doc_id": doc_id,
                    "text": doc_info.get("text", ""),
                    "description": doc_info.get("description", ""),
                    "score": score,
                    "related_documents": related_docs
                })

            return documents

        except (ServiceUnavailable, ClientError, ValueError, AttributeError) as e:
            logging.exception(f"Document 검색 실패: {e}")
            return []

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
                if "Tag" in labels:
                    node_id = f"tag_{props.get('name')}"
                elif "Category" in labels:
                    node_id = f"category_{props.get('name')}"
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

            formatted.append({
                "type": rel_type,
                "start": start,
                "end": end,
                "properties": clean_props
            })

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
        """검색된 Document들의 공통 Category/Tag 집계"""
        results = self.graph_repo.execute_query(CALCULATE_COMMON_METADATA, {"doc_ids": doc_ids})
        if not results:
            return {"categories": [], "common_tags": []}

        total_docs = len(doc_ids)
        categories = self._aggregate_metadata_counts(results, "categories", total_docs)
        common_tags = self._aggregate_metadata_counts(results, "tags", total_docs)

        logging.info(f"공통 메타데이터 집계: Category {len(categories)}개, Tag {len(common_tags)}개")

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
        """coverage 기준으로 메타데이터 필터링"""
        return [item for item in metadata_list if item.get("coverage", 0) >= min_coverage]

    @staticmethod
    def _create_metadata_nodes(
            categories: List[Dict[str, Any]],
            tags: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Category/Tag 노드 생성"""
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

        for tag in tags:
            nodes.append({
                "id": f"tag_{tag['name']}",
                "labels": ["Tag"],
                "properties": {
                    "name": tag["name"],
                    "doc_count": tag["doc_count"],
                    "coverage": tag["coverage"]
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

    def _get_tag_relationships(
            self,
            doc_ids: List[str],
            tag_names: List[str]
    ) -> List[Dict[str, Any]]:
        """Document-Tag 관계 조회

        Args:
            doc_ids: Document ID 목록
            tag_names: Tag 이름 목록

        Returns:
            관계 목록
        """
        if not tag_names:
            return []

        tag_rels = self.graph_repo.execute_query(
            GET_TAG_RELATIONSHIPS,
            {"doc_ids": doc_ids, "tag_names": tag_names}
        )

        return [
            {
                "type": "HAS_TAG",
                "start": rel["doc_id"],
                "end": f"tag_{rel['tag_name']}",
                "properties": {}
            }
            for rel in tag_rels
        ]

    def _build_simple_graph(
            self,
            doc_ids: List[str],
            common_metadata: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """간단한 그래프 생성: Document + 공통 Category/Tag만

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
        tag_names = [tag["name"] for tag in common_tags]

        relationships = []
        relationships.extend(self._get_category_relationships(doc_ids, category_names))
        relationships.extend(self._get_tag_relationships(doc_ids, tag_names))

        logging.info(f"간단한 그래프 생성: {len(nodes)}개 노드, {len(relationships)}개 관계")

        return {"nodes": nodes, "relationships": relationships}

    @staticmethod
    def _build_explore_query(node_id: str) -> tuple[str, Dict[str, Any]]:
        """노드 탐색 쿼리 생성"""
        if node_id.startswith("tag_"):
            match_clause = "MATCH (center:Tag {name: $name})"
            params = {"name": node_id[4:]}
        elif node_id.startswith("category_"):
            match_clause = "MATCH (center:Category {name: $name})"
            params = {"name": node_id[9:]}
        else:
            match_clause = "MATCH (center) WHERE center.id = $node_id OR elementId(center) = $node_id"
            params = {"node_id": node_id}

        return EXPLORE_NODE_TEMPLATE.format(match_clause=match_clause), params

    def _build_id_mapping(self, nodes: List[Dict[str, Any]]) -> Dict[str, str]:
        """노드 ID 매핑 생성 (원본 ID -> synthetic ID)"""
        id_mapping = {}
        for node in nodes:
            if not node:
                continue

            labels = node.get("labels", [])
            props = node.get("properties", {})
            original_id = node.get("id")

            if "Tag" in labels:
                synthetic_id = f"tag_{props.get('name')}"
                id_mapping[original_id] = synthetic_id
                id_mapping[props.get('name')] = synthetic_id
            elif "Category" in labels:
                synthetic_id = f"category_{props.get('name')}"
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

        doc_results = self.graph_repo.execute_query(GET_DOCUMENT_DETAILS, {"doc_ids": doc_ids})

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
            node_id: 노드 ID (Document ID 또는 tag_xxx, category_xxx 형식)

        Returns:
            노드 정보 또는 None
        """
        if node_id.startswith("tag_"):
            query, params = GET_NODE_BY_TAG, {"name": node_id[4:]}
        elif node_id.startswith("category_"):
            query, params = GET_NODE_BY_CATEGORY, {"name": node_id[9:]}
        else:
            query, params = GET_NODE_BY_ID, {"node_id": node_id}

        result = self.graph_repo.execute_query(query, params)
        if not result:
            return None

        node = result[0]
        final_id = node_id if node_id.startswith(("tag_", "category_")) else (
                node.get("properties", {}).get("id") or node.get("id"))

        return {
            "id": final_id,
            "labels": node.get("labels", []),
            "properties": self._sanitize_properties(node.get("properties", {}))
        }
