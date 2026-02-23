from typing import List, Dict, Optional

import logging
from graphdatascience import GraphDataScience
from neo4j.exceptions import ServiceUnavailable, ClientError

from app.config import settings
from app.constants import (
    GRAPH_EMBEDDING_DIM,
    HASHGNN_ITERATIONS,
    HASHGNN_EMBEDDING_DENSITY
)
from app.repositories.graph_repository import GraphRepository


class GraphEmbeddingService:
    """Neo4j GDS HashGNN 기반 Graph Embedding 생성 서비스"""

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
            embedding_dim: int = GRAPH_EMBEDDING_DIM
    ) -> Dict[str, List[float]]:
        """전체 그래프에 대해 HashGNN으로 Graph Embedding 생성

        Args:
            embedding_dim: 임베딩 차원 (기본: 128)

        Returns:
            {doc_id: embedding} dict
        """
        graph_name = "hashgnn_graph"

        try:
            if self._graph_exists(graph_name):
                self.gds.graph.drop(graph_name)

            existing_rel_types = self._get_existing_relationship_types()
            logging.info(f"[HashGNN] 존재하는 관계 타입: {existing_rel_types}")

            relationship_projection = self._build_relationship_projection(existing_rel_types)

            if not relationship_projection:
                raise ValueError("그래프에 관계가 하나도 없습니다")

            G, result = self.gds.graph.project(
                graph_name,
                ["Document", "Entity", "Topic", "Category"],
                relationship_projection
            )

            logging.info(
                f"[HashGNN] Graph projection 완료: {result['nodeCount']} nodes, "
                f"{result['relationshipCount']} relationships"
            )

            # HashGNN 실행 (관계 타입별 다른 해시 함수 사용)
            # Note: generateFeatures로 자동 feature 생성 (Topic/Category에는 임베딩이 없음)
            hashgnn_result = self.gds.beta.hashgnn.mutate(
                G,
                embeddingDensity=HASHGNN_EMBEDDING_DENSITY,
                iterations=HASHGNN_ITERATIONS,
                generateFeatures={
                    "dimension": GRAPH_EMBEDDING_DIM,
                    "densityLevel": 3
                },
                heterogeneous=True,  # 관계 타입별 다른 해시 함수 ⭐
                mutateProperty="graph_embedding",
                randomSeed=42
            )

            logging.info(f"[HashGNN] 완료: {hashgnn_result}")

            embeddings = self._extract_all_document_embeddings(G)
            logging.info(f"[HashGNN] 총 {len(embeddings)}개 Document 임베딩 추출 완료")

            self.gds.graph.drop(G)

            return embeddings

        except (ServiceUnavailable, ClientError):
            logging.exception("[HashGNN] 전체 그래프 임베딩 생성 중 DB 오류")
            if self._graph_exists(graph_name):
                self.gds.graph.drop(graph_name)
            return {}
        except (ValueError, RuntimeError, KeyError):
            logging.exception("[HashGNN] 전체 그래프 임베딩 생성 중 처리 오류")
            if self._graph_exists(graph_name):
                self.gds.graph.drop(graph_name)
            return {}

    @staticmethod
    def _build_relationship_projection(existing_rel_types: list) -> dict:
        """관계 projection 설정 빌드"""
        relationship_projection = {}

        # 메타데이터 관계
        if "HAS_TOPIC" in existing_rel_types:
            relationship_projection["HAS_TOPIC"] = {"orientation": "UNDIRECTED"}
        if "BELONGS_TO" in existing_rel_types:
            relationship_projection["BELONGS_TO"] = {"orientation": "UNDIRECTED"}
        if "MENTIONS" in existing_rel_types:
            relationship_projection["MENTIONS"] = {"orientation": "UNDIRECTED"}
        if "RELATED_TO" in existing_rel_types:
            relationship_projection["RELATED_TO"] = {"orientation": "UNDIRECTED"}

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

        return relationship_projection

    def _graph_exists(self, graph_name: str) -> bool:
        """In-memory graph 존재 여부 확인"""
        graphs = self.gds.graph.list()
        return graph_name in graphs["graphName"].values

    def _get_existing_relationship_types(self) -> list:
        """Neo4j 그래프에 존재하는 관계 타입 조회"""
        query = """
        CALL db.relationshipTypes() YIELD relationshipType
        RETURN collect(relationshipType) as types
        """
        result = self.graph_repo.execute_query(query)

        if result and "types" in result[0]:
            return result[0]["types"]

        return []

    def _extract_all_document_embeddings(self, G) -> Dict[str, List[float]]:
        """In-memory graph에서 모든 Document 노드의 임베딩 추출

        Args:
            G: GDS Graph 객체

        Returns:
            {doc_id: embedding} dict
        """
        embeddings = {}

        result = self.gds.graph.nodeProperty.stream(
            G,
            "graph_embedding",
            ["Document"]
        )

        for _, row in result.iterrows():
            node_props = self._get_node_properties(row["nodeId"])
            if node_props and "id" in node_props:
                doc_id = node_props["id"]
                embedding = row["propertyValue"]

                if embedding is not None and isinstance(embedding, list):
                    embeddings[doc_id] = embedding

        return embeddings

    def _get_node_properties(self, node_id: int) -> Optional[Dict]:
        """Neo4j internal ID로 노드 속성 조회

        Args:
            node_id: Neo4j internal node ID

        Returns:
            노드 속성 dict 또는 None
        """
        query = "MATCH (n) WHERE id(n) = $node_id RETURN properties(n) as props"
        result = self.graph_repo.execute_query(query, {"node_id": node_id})

        if result:
            return result[0]["props"]
        return None
