import ast
import json
import re
from typing import Dict, Any, List

import logging
from neo4j.time import DateTime
from neo4j.exceptions import (
    ClientError,
    ServiceUnavailable,
    TransientError,
    CypherSyntaxError
)
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever

from app.config import settings
from app.repositories.graph_repository import GraphRepository
from app.exceptions.graphrag import handle_graph_error


class GraphRAGService:
    """Graph RAG 검색 서비스 - 사용자 질의 기반 그래프 구조 반환"""

    def __init__(self, graph_repo: GraphRepository):
        """GraphRAGService 초기화

        Args:
            graph_repo: GraphRepository 인스턴스
        """
        self.graph_repo = graph_repo
        self.driver = graph_repo.driver

        self.embedder = SentenceTransformerEmbeddings(
            model=settings.EMBEDDING_MODEL_NAME
        )
        logging.info("GraphRAGService 초기화 완료")

    @staticmethod
    def _sanitize_properties(properties: Dict[str, Any]) -> Dict[str, Any]:
        """Neo4j 속성을 JSON 직렬화 가능한 형태로 변환

        Args:
            properties: Neo4j 노드/관계 속성

        Returns:
            직렬화 가능한 속성 dict
        """
        sanitized = {}
        for key, value in properties.items():
            if isinstance(value, DateTime):
                # Neo4j DateTime을 ISO 형식 문자열로 변환
                sanitized[key] = value.isoformat()
            elif isinstance(value, (list, tuple)):
                # 리스트 내부의 DateTime도 변환
                sanitized[key] = [
                    v.isoformat() if isinstance(v, DateTime) else v
                    for v in value
                ]
            else:
                sanitized[key] = value
        return sanitized

    def search_graph(
            self,
            query: str,
            top_k: int = 5
    ) -> Dict[str, Any]:
        """사용자 질의 기반 Graph RAG 검색

        Args:
            query: 사용자 질의
            top_k: 반환할 Document 개수 (기본: 5)

        Returns:
            {
                "query": "사용자 질의",
                "documents": [...],  # 검색된 Document 목록
                "common_metadata": {
                    "categories": [...],  # 공통 Category 집계
                    "common_tags": [...]  # 공통 Tag 집계
                }
            }
        """
        try:
            # 벡터 검색으로 관련 Document 찾기
            documents = self._search_documents(query, top_k)

            if not documents:
                return {
                    "query": query,
                    "documents": [],
                    "common_metadata": {"categories": [], "common_tags": []},
                    "graph": {"nodes": [], "relationships": []}
                }

            # Document ID 목록 추출
            doc_ids = [doc["doc_id"] for doc in documents]

            # 공통 메타데이터 계산
            common_metadata = self._calculate_common_metadata(doc_ids)

            # 간단한 그래프 구조 생성 (Document + 공통 메타데이터만)
            graph = self._build_simple_graph(doc_ids, common_metadata)

            logging.info(
                f"Graph RAG 검색 완료: {len(documents)}개 Document, "
                f"{len(graph['nodes'])}개 노드, {len(graph['relationships'])}개 관계"
            )

            return {
                "query": query,
                "documents": documents,
                "common_metadata": common_metadata,
                "graph": graph
            }

        except (ServiceUnavailable, TransientError, ClientError, ValueError, KeyError, AttributeError) as e:
            handle_graph_error(e, "Graph RAG 검색")
            return {
                "query": query,
                "documents": [],
                "common_metadata": {"categories": [], "common_tags": []},
                "graph": {"nodes": [], "relationships": []},
                "error": str(e)
            }

    def _search_documents(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """텍스트 임베딩 기반 Document 검색

        Args:
            query: 사용자 질의
            top_k: 반환할 Document 개수

        Returns:
            Document 목록
        """
        try:
            retriever = VectorRetriever(
                driver=self.driver,
                index_name="text_embedding_index",
                embedder=self.embedder
            )
            result = retriever.search(query_text=query, top_k=top_k)
            logging.info(f"VectorRetriever 검색 결과: {len(result.items)}개")

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

            logging.info(f"Document 검색 완료: {len(documents)}개")
            return documents

        except (ServiceUnavailable, ClientError, ValueError, AttributeError) as e:
            handle_graph_error(e, "Document 검색")
            return []

    def _extract_doc_id(self, content) -> str | None:
        """content에서 doc_id 추출"""
        try:
            if isinstance(content, dict):
                doc_id = content.get("id")
                logging.info(f"Extracted doc_id from dict: {doc_id}")
                return doc_id

            if isinstance(content, str):
                # 방법 1: 정규표현식으로 'id' 필드 추출 (가장 빠르고 안정적)
                match = re.search(r"'id':\s*'([^']+)'", content)
                if match:
                    doc_id = match.group(1)
                    logging.info(f"Extracted doc_id via regex: {doc_id}")
                    return doc_id

                # 방법 2: JSON 파싱 시도
                try:
                    # Python dict literal을 JSON으로 변환 ('를 "로 변경)
                    json_str = content.replace("'", '"')
                    content_dict = json.loads(json_str)
                    doc_id = content_dict.get("id")
                    logging.info(f"Extracted doc_id via JSON: {doc_id}")
                    return doc_id
                except:
                    pass

                # 방법 3: ast.literal_eval 시도
                try:
                    content_dict = ast.literal_eval(content)
                    doc_id = content_dict.get("id")
                    logging.info(f"Extracted doc_id via ast: {doc_id}")
                    return doc_id
                except:
                    pass

                logging.warning(f"All extraction methods failed for content")
                return None

            logging.warning(f"Unknown content type: {type(content)}")
            return None

        except Exception as e:
            logging.error(f"Failed to extract doc_id: {e}")
            return None

    def _get_document_info(self, doc_id: str) -> Dict[str, Any] | None:
        """Document 상세 정보 조회"""
        try:
            query = """
            MATCH (d:Document {id: $doc_id})
            RETURN d.text as text, d.description as description
            """
            result = self.graph_repo.execute_query(query, {"doc_id": doc_id})

            if result:
                return {
                    "text": result[0].get("text", ""),
                    "description": result[0].get("description", "")
                }
            return None

        except (ServiceUnavailable, ClientError) as e:
            handle_graph_error(e, f"Document 정보 조회 (doc_id: {doc_id})")
            return None

    def _get_related_documents(self, doc_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """문서의 연관 문서 조회 (함께 보면 좋은 문서)

        Args:
            doc_id: 문서 ID
            limit: 최대 반환 개수

        Returns:
            연관 문서 목록 [{target_doc_id, target_description, relation_type, reason, confidence}]
        """
        try:
            query = """
            MATCH (source:Document {id: $doc_id})-[r]->(target:Document)
            WHERE type(r) IN ['SIMILAR_TO', 'EXTENDS', 'CONTRADICTS', 'PREREQUISITE_OF', 'IMPLEMENTS', 'CO_OCCURS']
            RETURN target.id as target_doc_id,
                   target.description as target_description,
                   type(r) as relation_type,
                   r.reason as reason,
                   r.confidence as confidence
            ORDER BY r.confidence DESC
            LIMIT $limit
            """
            result = self.graph_repo.execute_query(query, {"doc_id": doc_id, "limit": limit})

            related_docs = []
            for row in result:
                related_docs.append({
                    "target_doc_id": row.get("target_doc_id", ""),
                    "target_description": row.get("target_description", ""),
                    "relation_type": row.get("relation_type", ""),
                    "reason": row.get("reason", ""),
                    "confidence": row.get("confidence", 0.0)
                })

            logging.info(f"문서 {doc_id}의 연관 문서: {len(related_docs)}개")
            return related_docs

        except (ServiceUnavailable, ClientError) as e:
            handle_graph_error(e, f"연관 문서 조회 (doc_id: {doc_id})")
            return []

    def _extract_graph_structure(
            self,
            doc_ids: List[str],
            depth: int = 1
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Document와 연결된 그래프 구조 추출

        Args:
            doc_ids: Document ID 목록
            depth: 그래프 탐색 깊이 (1 = 1-hop, 2 = 2-hop)

        Returns:
            {
                "nodes": [...],
                "relationships": [...]
            }
        """
        try:
            # Cypher 쿼리로 그래프 구조 추출
            query = self._build_graph_extraction_query(depth)

            result = self.graph_repo.execute_query(
                query,
                {"doc_ids": doc_ids}
            )

            if not result:
                return {"nodes": [], "relationships": []}

            # 결과 파싱
            nodes = result[0].get("nodes", [])
            relationships = result[0].get("relationships", [])

            # 노드 포맷팅
            formatted_nodes = self._format_nodes(nodes)

            # 관계 포맷팅
            formatted_relationships = self._format_relationships(relationships)

            return {
                "nodes": formatted_nodes,
                "relationships": formatted_relationships
            }

        except (ServiceUnavailable, ClientError) as e:
            handle_graph_error(e, "그래프 구조 추출")
            return {"nodes": [], "relationships": []}

    def _build_graph_extraction_query(self, depth: int) -> str:
        """그래프 추출 Cypher 쿼리 생성

        Args:
            depth: 그래프 탐색 깊이

        Returns:
            Cypher 쿼리
        """
        if depth == 1:
            # 1-hop: Document와 직접 연결된 노드/관계
            return """
            MATCH (d:Document)
            WHERE d.id IN $doc_ids
            OPTIONAL MATCH path = (d)-[r]-(n)
            WHERE n:Topic OR n:Tag OR n:Category OR n:Document
            WITH collect(DISTINCT d) + collect(DISTINCT n) as nodes,
                 collect(DISTINCT r) as rels
            RETURN [node IN nodes | {
                id: COALESCE(node.id, elementId(node)),
                labels: labels(node),
                properties: properties(node)
            }] as nodes,
            [rel IN rels WHERE rel IS NOT NULL | {
                type: type(rel),
                start: COALESCE(startNode(rel).id, elementId(startNode(rel))),
                end: COALESCE(endNode(rel).id, elementId(endNode(rel))),
                properties: properties(rel)
            }] as relationships
            """
        elif depth == 2:
            # 2-hop: 2단계 연결까지 포함
            return """
            MATCH (d:Document)
            WHERE d.id IN $doc_ids
            OPTIONAL MATCH path = (d)-[r1]-(n1)-[r2]-(n2)
            WHERE (n1:Topic OR n1:Tag OR n1:Category OR n1:Document)
              AND (n2:Topic OR n2:Tag OR n2:Category OR n2:Document)
            WITH collect(DISTINCT d) + collect(DISTINCT n1) + collect(DISTINCT n2) as nodes,
                 collect(DISTINCT r1) + collect(DISTINCT r2) as rels
            RETURN [node IN nodes | {
                id: COALESCE(node.id, elementId(node)),
                labels: labels(node),
                properties: properties(node)
            }] as nodes,
            [rel IN rels WHERE rel IS NOT NULL | {
                type: type(rel),
                start: COALESCE(startNode(rel).id, elementId(startNode(rel))),
                end: COALESCE(endNode(rel).id, elementId(endNode(rel))),
                properties: properties(rel)
            }] as relationships
            """
        else:
            # 기본: 1-hop
            return self._build_graph_extraction_query(1)

    def _format_nodes(
            self,
            nodes: List[Dict[str, Any]],
            use_synthetic_id: bool = False
    ) -> List[Dict[str, Any]]:
        """노드 포맷팅

        Args:
            nodes: 원본 노드 목록
            use_synthetic_id: Tag/Category에 synthetic ID 사용 여부

        Returns:
            포맷팅된 노드 목록
        """
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

    def _calculate_common_metadata(self, doc_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """검색된 Document들의 공통 Category/Tag 집계

        Args:
            doc_ids: Document ID 목록

        Returns:
            {
                "categories": [{"name": "카테고리명", "doc_count": 3, "coverage": 0.6}],
                "common_tags": [{"name": "태그명", "doc_count": 4, "coverage": 0.8}]
            }
        """
        try:
            query = """
            MATCH (d:Document)
            WHERE d.id IN $doc_ids

            // Category 수집 (Topic을 통해)
            OPTIONAL MATCH (d)-[:HAS_TOPIC]->(t:Topic)-[:BELONGS_TO]->(c:Category)
            WITH d, collect(DISTINCT c.name) as categories

            // Tag 수집
            OPTIONAL MATCH (d)-[:HAS_TAG]->(tag:Tag)
            WITH d, categories, collect(DISTINCT tag.name) as tags

            RETURN d.id as doc_id, categories, tags
            """

            results = self.graph_repo.execute_query(query, {"doc_ids": doc_ids})

            if not results:
                return {"categories": [], "common_tags": []}

            total_docs = len(doc_ids)

            # Category 집계
            category_counts = {}
            for row in results:
                for cat in row.get("categories", []):
                    if cat:
                        category_counts[cat] = category_counts.get(cat, 0) + 1

            categories = [
                {
                    "name": cat,
                    "doc_count": count,
                    "coverage": round(count / total_docs, 2)
                }
                for cat, count in sorted(
                    category_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            ]

            # Tag 집계
            tag_counts = {}
            for row in results:
                for tag in row.get("tags", []):
                    if tag:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1

            common_tags = [
                {
                    "name": tag,
                    "doc_count": count,
                    "coverage": round(count / total_docs, 2)
                }
                for tag, count in sorted(
                    tag_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            ]

            logging.info(
                f"공통 메타데이터 집계: Category {len(categories)}개, Tag {len(common_tags)}개"
            )

            return {
                "categories": categories,
                "common_tags": common_tags
            }

        except (ServiceUnavailable, ClientError) as e:
            handle_graph_error(e, "공통 메타데이터 계산")
            return {"categories": [], "common_tags": []}

    def _build_simple_graph(
            self,
            doc_ids: List[str],
            common_metadata: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """간단한 그래프 생성: Document + 공통 Category/Tag만

        Args:
            doc_ids: Document ID 목록
            common_metadata: 공통 메타데이터 (categories, common_tags)

        Returns:
            {
                "nodes": [...],
                "relationships": [...]
            }
        """
        try:
            nodes = []
            relationships = []

            # 1. Document 노드 조회
            doc_query = """
            MATCH (d:Document)
            WHERE d.id IN $doc_ids
            RETURN d.id as id, d.description as description
            """
            doc_results = self.graph_repo.execute_query(doc_query, {"doc_ids": doc_ids})

            for doc in doc_results:
                nodes.append({
                    "id": doc["id"],
                    "labels": ["Document"],
                    "properties": {
                        "id": doc["id"],
                        "description": doc.get("description", "")
                    }
                })

            # 2. 공통 Category 노드 (coverage >= 50%)
            common_categories = [
                cat for cat in common_metadata.get("categories", [])
                if cat["coverage"] >= 0.5
            ]

            for cat in common_categories:
                nodes.append({
                    "id": f"category_{cat['name']}",
                    "labels": ["Category"],
                    "properties": {
                        "name": cat["name"],
                        "doc_count": cat["doc_count"],
                        "coverage": cat["coverage"]
                    }
                })

            # 3. 공통 Tag 노드 (coverage >= 50%)
            common_tags = [
                tag for tag in common_metadata.get("common_tags", [])
                if tag["coverage"] >= 0.5
            ]

            for tag in common_tags:
                nodes.append({
                    "id": f"tag_{tag['name']}",
                    "labels": ["Tag"],
                    "properties": {
                        "name": tag["name"],
                        "doc_count": tag["doc_count"],
                        "coverage": tag["coverage"]
                    }
                })

            # 4. Document-Category 관계 조회
            if common_categories:
                category_names = [cat["name"] for cat in common_categories]
                cat_rel_query = """
                MATCH (d:Document)-[:HAS_TOPIC]->(t:Topic)-[:BELONGS_TO]->(c:Category)
                WHERE d.id IN $doc_ids AND c.name IN $category_names
                RETURN DISTINCT d.id as doc_id, c.name as category_name
                """
                cat_rels = self.graph_repo.execute_query(
                    cat_rel_query,
                    {"doc_ids": doc_ids, "category_names": category_names}
                )

                for rel in cat_rels:
                    relationships.append({
                        "type": "BELONGS_TO",
                        "start": rel["doc_id"],
                        "end": f"category_{rel['category_name']}",
                        "properties": {}
                    })

            # 5. Document-Tag 관계 조회
            if common_tags:
                tag_names = [tag["name"] for tag in common_tags]
                tag_rel_query = """
                MATCH (d:Document)-[:HAS_TAG]->(tag:Tag)
                WHERE d.id IN $doc_ids AND tag.name IN $tag_names
                RETURN DISTINCT d.id as doc_id, tag.name as tag_name
                """
                tag_rels = self.graph_repo.execute_query(
                    tag_rel_query,
                    {"doc_ids": doc_ids, "tag_names": tag_names}
                )

                for rel in tag_rels:
                    relationships.append({
                        "type": "HAS_TAG",
                        "start": rel["doc_id"],
                        "end": f"tag_{rel['tag_name']}",
                        "properties": {}
                    })

            logging.info(
                f"간단한 그래프 생성: {len(nodes)}개 노드, {len(relationships)}개 관계"
            )

            return {
                "nodes": nodes,
                "relationships": relationships
            }

        except (ServiceUnavailable, ClientError) as e:
            handle_graph_error(e, "간단한 그래프 생성")
            return {"nodes": [], "relationships": []}

    def _build_explore_query(self, node_id: str) -> tuple[str, Dict[str, Any]]:
        """노드 탐색 쿼리 생성

        Args:
            node_id: 노드 ID

        Returns:
            (쿼리 문자열, 파라미터) 튜플
        """
        # 공통 쿼리 템플릿
        query_template = """
        {match_clause}
        OPTIONAL MATCH (center)-[r]-(connected)
        WHERE connected:Document OR connected:Topic OR connected:Tag OR connected:Category
        WITH center, collect(DISTINCT connected) as connected_nodes, collect(DISTINCT r) as relationships
        WITH [center] + connected_nodes as all_nodes, relationships
        RETURN [node IN all_nodes WHERE node IS NOT NULL | {{
            id: COALESCE(node.id, node.name, elementId(node)),
            labels: labels(node),
            properties: properties(node)
        }}] as nodes,
        [rel IN relationships WHERE rel IS NOT NULL | {{
            type: type(rel),
            start: COALESCE(startNode(rel).id, startNode(rel).name, elementId(startNode(rel))),
            end: COALESCE(endNode(rel).id, endNode(rel).name, elementId(endNode(rel))),
            properties: properties(rel)
        }}] as relationships
        """

        if node_id.startswith("tag_"):
            match_clause = "MATCH (center:Tag {name: $name})"
            params = {"name": node_id[4:]}
        elif node_id.startswith("category_"):
            match_clause = "MATCH (center:Category {name: $name})"
            params = {"name": node_id[9:]}
        else:
            match_clause = "MATCH (center) WHERE center.id = $node_id OR elementId(center) = $node_id"
            params = {"node_id": node_id}

        return query_template.format(match_clause=match_clause), params

    def _build_id_mapping(self, nodes: List[Dict[str, Any]]) -> Dict[str, str]:
        """노드 ID 매핑 생성 (원본 ID -> synthetic ID)

        Args:
            nodes: 노드 목록

        Returns:
            ID 매핑 딕셔너리
        """
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

        doc_query = """
        MATCH (d:Document)
        WHERE d.id IN $doc_ids
        RETURN d.id as doc_id, d.text as text, d.description as description
        """
        doc_results = self.graph_repo.execute_query(doc_query, {"doc_ids": doc_ids})

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
            {
                "center_node": {...},
                "connected_documents": [...],
                "graph": {
                    "nodes": [...],
                    "relationships": [...]
                }
            }
        """
        try:
            center_node = self._get_node_info(node_id)
            if not center_node:
                return {
                    "center_node": None,
                    "connected_documents": [],
                    "graph": {"nodes": [], "relationships": []},
                    "error": f"노드를 찾을 수 없습니다: {node_id}"
                }

            graph_query, query_params = self._build_explore_query(node_id)
            result = self.graph_repo.execute_query(graph_query, query_params)

            if not result:
                return {
                    "center_node": center_node,
                    "connected_documents": [],
                    "graph": {"nodes": [], "relationships": []}
                }

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

            return {
                "center_node": center_node,
                "connected_documents": connected_documents,
                "graph": {
                    "nodes": nodes,
                    "relationships": relationships
                }
            }

        except (ServiceUnavailable, ClientError) as e:
            handle_graph_error(e, f"노드 탐색 (node_id: {node_id})")
            return {
                "center_node": None,
                "connected_documents": [],
                "graph": {"nodes": [], "relationships": []},
                "error": str(e)
            }

    def _get_node_info(self, node_id: str) -> Dict[str, Any] | None:
        """노드 정보 조회

        Args:
            node_id: 노드 ID (Document ID 또는 tag_xxx, category_xxx 형식)

        Returns:
            노드 정보 또는 None
        """
        try:
            # Tag/Category synthetic ID 처리
            if node_id.startswith("tag_"):
                tag_name = node_id[4:]  # "tag_" 제거
                query = """
                MATCH (n:Tag {name: $name})
                RETURN elementId(n) as id, labels(n) as labels, properties(n) as properties
                LIMIT 1
                """
                result = self.graph_repo.execute_query(query, {"name": tag_name})

                if result:
                    node = result[0]
                    return {
                        "id": node_id,  # synthetic ID 유지
                        "labels": node.get("labels", []),
                        "properties": self._sanitize_properties(node.get("properties", {}))
                    }
                return None

            elif node_id.startswith("category_"):
                category_name = node_id[9:]  # "category_" 제거
                query = """
                MATCH (n:Category {name: $name})
                RETURN elementId(n) as id, labels(n) as labels, properties(n) as properties
                LIMIT 1
                """
                result = self.graph_repo.execute_query(query, {"name": category_name})

                if result:
                    node = result[0]
                    return {
                        "id": node_id,  # synthetic ID 유지
                        "labels": node.get("labels", []),
                        "properties": self._sanitize_properties(node.get("properties", {}))
                    }
                return None

            # 일반 노드 (Document 등)
            else:
                query = """
                MATCH (n)
                WHERE n.id = $node_id OR elementId(n) = $node_id
                RETURN elementId(n) as id, labels(n) as labels, properties(n) as properties
                LIMIT 1
                """
                result = self.graph_repo.execute_query(query, {"node_id": node_id})

                if result:
                    node = result[0]
                    return {
                        "id": node.get("properties", {}).get("id") or node.get("id"),
                        "labels": node.get("labels", []),
                        "properties": self._sanitize_properties(node.get("properties", {}))
                    }
                return None

        except (ServiceUnavailable, ClientError) as e:
            handle_graph_error(e, f"노드 정보 조회 (node_id: {node_id})")
            return None
