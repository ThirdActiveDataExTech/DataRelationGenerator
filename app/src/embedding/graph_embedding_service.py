from typing import List, Dict, Optional

import logging
from graphdatascience import GraphDataScience
from neo4j.exceptions import ServiceUnavailable, ClientError

from app.config import settings
from app.repositories.graph_repository import GraphRepository


class GraphEmbeddingService:
    """Neo4j GDS Node2Vec 기반 Graph Embedding 생성 서비스"""

    def __init__(self, graph_repo: GraphRepository):
        """초기화

        Args:
            graph_repo: GraphRepository 인스턴스 (driver 재사용)
        """
        self.graph_repo = graph_repo

        # GDS 클라이언트 초기화
        self.gds = GraphDataScience(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )

    def generate_all_graph_embeddings(
            self,
            embedding_dim: int = 128,
            walk_length: int = 80,
            walks_per_node: int = 10
    ) -> Dict[str, List[float]]:
        """전체 그래프에 대해 Node2Vec 실행하여 모든 Document 노드의 임베딩 생성

        Args:
            embedding_dim: 임베딩 차원 (기본: 128)
            walk_length: Random walk 길이 (기본: 80)
            walks_per_node: 노드당 walk 횟수 (기본: 10)

        Returns:
            {doc_id: embedding} dict
        """
        graph_name = "full_graph_for_embedding"

        try:
            if self._graph_exists(graph_name):
                self.gds.graph.drop(graph_name)

            existing_rel_types = self._get_existing_relationship_types()
            logging.info(f"존재하는 관계 타입: {existing_rel_types}")

            # 3. 관계 projection 설정 (존재하는 것만)
            relationship_projection = {}

            # 메타데이터 관계
            if "HAS_TOPIC" in existing_rel_types:
                relationship_projection["HAS_TOPIC"] = {"orientation": "UNDIRECTED"}
            if "HAS_TAG" in existing_rel_types:
                relationship_projection["HAS_TAG"] = {"orientation": "UNDIRECTED"}
            if "BELONGS_TO" in existing_rel_types:
                relationship_projection["BELONGS_TO"] = {"orientation": "UNDIRECTED"}

            # Document 간 관계
            if "SIMILAR_TO" in existing_rel_types:
                relationship_projection["SIMILAR_TO"] = {"orientation": "UNDIRECTED"}
            if "CONTRADICTS" in existing_rel_types:
                relationship_projection["CONTRADICTS"] = {"orientation": "UNDIRECTED"}
            if "EXTENDS" in existing_rel_types:
                relationship_projection["EXTENDS"] = {"orientation": "NATURAL"}
            if "PREREQUISITE_OF" in existing_rel_types:
                relationship_projection["PREREQUISITE_OF"] = {"orientation": "NATURAL"}
            if "IMPLEMENTS" in existing_rel_types:
                relationship_projection["IMPLEMENTS"] = {"orientation": "NATURAL"}
            if "CO_OCCURS" in existing_rel_types:
                relationship_projection["CO_OCCURS"] = {"orientation": "UNDIRECTED"}

            if not relationship_projection:
                raise ValueError("그래프에 관계가 하나도 없습니다")

            G, result = self.gds.graph.project(
                graph_name,
                ["Document", "Topic", "Tag", "Category"],
                relationship_projection
            )

            logging.info(
                f"Graph projection 완료: {result['nodeCount']} nodes, "
                f"{result['relationshipCount']} relationships"
            )

            node2vec_result = self.gds.node2vec.mutate(
                G,
                embeddingDimension=embedding_dim,
                walkLength=walk_length,
                walksPerNode=walks_per_node,
                mutateProperty="graph_embedding"
            )

            logging.info(f"Node2Vec 완료: {node2vec_result}")

            embeddings = self._extract_all_document_embeddings(G)
            logging.info(f"총 {len(embeddings)}개 Document 임베딩 추출 완료")

            self.gds.graph.drop(G)

            return embeddings

        except (ServiceUnavailable, ClientError) as e:
            logging.exception(f"전체 그래프 임베딩 생성 중 DB 오류")
            if self._graph_exists(graph_name):
                self.gds.graph.drop(graph_name)
            return {}
        except (ValueError, RuntimeError, KeyError) as e:
            logging.exception(f"전체 그래프 임베딩 생성 중 처리 오류")
            if self._graph_exists(graph_name):
                self.gds.graph.drop(graph_name)
            return {}

    def _graph_exists(self, graph_name: str) -> bool:
        """In-memory graph 존재 여부 확인"""
        try:
            graphs = self.gds.graph.list()
            return graph_name in graphs["graphName"].values
        except (ValueError, KeyError, AttributeError):
            return False

    def _get_existing_relationship_types(self) -> list:
        """Neo4j 그래프에 존재하는 관계 타입 조회

        Returns:
            관계 타입 리스트
        """
        try:
            query = """
            CALL db.relationshipTypes() YIELD relationshipType
            RETURN collect(relationshipType) as types
            """
            result = self.graph_repo.execute_query(query)

            if result and "types" in result[0]:
                return result[0]["types"]

            return []

        except (ServiceUnavailable, ClientError) as e:
            logging.exception(f"관계 타입 조회 실패 (DB 오류)")
            return []

    def _extract_all_document_embeddings(self, G) -> Dict[str, List[float]]:
        """In-memory graph에서 모든 Document 노드의 임베딩 추출

        Args:
            G: GDS Graph 객체

        Returns:
            {doc_id: embedding} dict
        """
        embeddings = {}

        try:
            # GDS stream을 사용하여 임베딩 추출
            result = self.gds.graph.nodeProperty.stream(
                G,
                "graph_embedding",
                ["Document"]
            )

            # 각 노드의 doc_id와 임베딩 매핑
            for _, row in result.iterrows():
                node_props = self._get_node_properties(row["nodeId"])
                if node_props and "id" in node_props:
                    doc_id = node_props["id"]
                    embedding = row["propertyValue"]

                    if embedding is not None and isinstance(embedding, list):
                        embeddings[doc_id] = embedding

            return embeddings

        except (ValueError, KeyError, AttributeError, RuntimeError) as e:
            logging.exception(f"임베딩 추출 실패")
            return {}

    def _get_node_properties(self, node_id: int) -> Optional[Dict]:
        """Neo4j internal ID로 노드 속성 조회

        Args:
            node_id: Neo4j internal node ID

        Returns:
            노드 속성 dict 또는 None
        """
        try:
            query = "MATCH (n) WHERE id(n) = $node_id RETURN properties(n) as props"
            result = self.graph_repo.execute_query(query, {"node_id": node_id})

            if result:
                return result[0]["props"]
            return None

        except Exception as e:
            logging.error(f"노드 속성 조회 실패 (node_id: {node_id}): {e}")
            return None
