from typing import List, Dict, Optional
import logging

from graphdatascience import GraphDataScience

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

        logging.info("GraphEmbeddingService 초기화 완료")

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
            # 1. 기존 그래프 삭제 (있다면)
            if self._graph_exists(graph_name):
                self.gds.graph.drop(graph_name)
                logging.info(f"기존 그래프 '{graph_name}' 삭제 완료")

            # 2. 존재하는 관계 타입 확인
            existing_rel_types = self._get_existing_relationship_types()
            logging.info(f"존재하는 관계 타입: {existing_rel_types}")

            # 3. 관계 projection 설정 (존재하는 것만)
            relationship_projection = {}

            # 기본 관계 (항상 존재)
            if "HAS_TOPIC" in existing_rel_types:
                relationship_projection["HAS_TOPIC"] = {"orientation": "UNDIRECTED"}
            if "HAS_TAG" in existing_rel_types:
                relationship_projection["HAS_TAG"] = {"orientation": "UNDIRECTED"}
            if "BELONGS_TO" in existing_rel_types:
                relationship_projection["BELONGS_TO"] = {"orientation": "UNDIRECTED"}

            # 의미적 관계 (Step 3 이후 존재)
            if "EXTENDS" in existing_rel_types:
                relationship_projection["EXTENDS"] = {"orientation": "NATURAL"}
            if "USES" in existing_rel_types:
                relationship_projection["USES"] = {"orientation": "NATURAL"}
            if "PREREQUISITE_OF" in existing_rel_types:
                relationship_projection["PREREQUISITE_OF"] = {"orientation": "NATURAL"}
            if "COMPARES_WITH" in existing_rel_types:
                relationship_projection["COMPARES_WITH"] = {"orientation": "UNDIRECTED"}
            if "APPLIES_TO" in existing_rel_types:
                relationship_projection["APPLIES_TO"] = {"orientation": "NATURAL"}
            if "SAME_DOMAIN" in existing_rel_types:
                relationship_projection["SAME_DOMAIN"] = {"orientation": "UNDIRECTED"}

            if not relationship_projection:
                raise ValueError("그래프에 관계가 하나도 없습니다")

            # 4. 전체 그래프 projection 생성
            logging.info(f"전체 그래프 projection 생성 시작 (관계 타입: {list(relationship_projection.keys())})")

            G, result = self.gds.graph.project(
                graph_name,
                ["Document", "Topic", "Tag", "Category"],
                relationship_projection
            )

            logging.info(
                f"Graph projection 완료: {result['nodeCount']} nodes, "
                f"{result['relationshipCount']} relationships"
            )

            # 3. Node2Vec 실행 (전체 그래프)
            logging.info("Node2Vec 실행 시작 (전체 그래프)")

            node2vec_result = self.gds.node2vec.mutate(
                G,
                embeddingDimension=embedding_dim,
                walkLength=walk_length,
                walksPerNode=walks_per_node,
                mutateProperty="graph_embedding"
            )

            logging.info(f"Node2Vec 완료: {node2vec_result}")

            # 4. 모든 Document 노드의 임베딩 추출
            embeddings = self._extract_all_document_embeddings(G)

            logging.info(f"총 {len(embeddings)}개 Document 임베딩 추출 완료")

            # 5. Graph 삭제 (메모리 절약)
            self.gds.graph.drop(G)

            return embeddings

        except Exception as e:
            logging.error(f"전체 그래프 임베딩 생성 중 오류: {e}")
            import traceback
            logging.error(traceback.format_exc())

            # 에러 발생 시 그래프 삭제
            if self._graph_exists(graph_name):
                self.gds.graph.drop(graph_name)

            return {}

    def generate_graph_embedding(
        self,
        doc_id: str,
        embedding_dim: int = 128,
        walk_length: int = 80,
        walks_per_node: int = 10
    ) -> Optional[List[float]]:
        """특정 Document 노드에 대한 Graph Embedding 생성 (전체 그래프 기반)

        Args:
            doc_id: Document ID
            embedding_dim: 임베딩 차원 (기본: 128)
            walk_length: Random walk 길이 (기본: 80)
            walks_per_node: 노드당 walk 횟수 (기본: 10)

        Returns:
            Graph embedding 벡터 또는 None

        Note:
            개별 문서보다는 generate_all_graph_embeddings() 사용 권장
        """
        try:
            # 전체 그래프 임베딩 생성 후 특정 문서만 추출
            all_embeddings = self.generate_all_graph_embeddings(
                embedding_dim=embedding_dim,
                walk_length=walk_length,
                walks_per_node=walks_per_node
            )

            return all_embeddings.get(doc_id)

        except Exception as e:
            logging.error(f"Graph embedding 생성 중 오류 (doc_id: {doc_id}): {e}")
            return None

    def _graph_exists(self, graph_name: str) -> bool:
        """In-memory graph 존재 여부 확인"""
        try:
            graphs = self.gds.graph.list()
            return graph_name in graphs["graphName"].values
        except Exception:
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

        except Exception as e:
            logging.error(f"관계 타입 조회 실패: {e}")
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

            logging.info(f"총 {len(embeddings)}개 Document 임베딩 추출 완료")
            return embeddings

        except Exception as e:
            logging.error(f"임베딩 추출 실패: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return {}

    def _extract_node_embedding(self, graph_name: str, doc_id: str) -> Optional[List[float]]:
        """In-memory graph에서 특정 노드의 임베딩 추출 (레거시)

        Args:
            graph_name: In-memory graph 이름
            doc_id: Document ID

        Returns:
            임베딩 벡터 또는 None
        """
        try:
            # GDS stream을 사용하여 임베딩 추출
            result = self.gds.graph.nodeProperty.stream(
                graph_name,
                "graph_embedding",
                ["Document"]
            )

            # doc_id에 해당하는 노드 찾기
            for _, row in result.iterrows():
                # nodeId는 Neo4j internal ID이므로, 실제 id 속성과 매칭해야 함
                node_props = self._get_node_properties(row["nodeId"])
                if node_props and node_props.get("id") == doc_id:
                    embedding_value = row["propertyValue"]
                    # numpy array인 경우 list로 변환
                    if hasattr(embedding_value, "tolist"):
                        return embedding_value.tolist()
                    if isinstance(embedding_value, list):
                        return embedding_value
                    return None

            return None

        except Exception as e:
            logging.error(f"임베딩 추출 실패 (graph: {graph_name}, doc_id: {doc_id}): {e}")
            return None

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
