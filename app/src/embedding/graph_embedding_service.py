from typing import List, Dict, Optional

import logging
from graphdatascience import GraphDataScience
from neo4j.exceptions import ServiceUnavailable, ClientError
from tqdm import tqdm

from app.config import settings
from app.constants import (
    GRAPH_EMBEDDING_DIM,
    HASHGNN_ITERATIONS,
    HASHGNN_EMBEDDING_DENSITY
)
from app.repositories.graph_repository import GraphRepository
from app.src.embedding.embedding_service import embedding_service


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
        """전체 그래프에 대해 GraphSAGE로 Graph Embedding 생성

        Args:
            embedding_dim: 임베딩 차원 (기본: 128)

        Returns:
            {doc_id: embedding} dict
        """
        graph_name = "graphsage_graph"

        try:
            # 1. Key, Value 노드에 임베딩 추가
            logging.info("[GraphSAGE] Key, Value 노드 임베딩 생성 시작")
            self._add_key_embeddings()
            self._add_value_embeddings()
            logging.info("[GraphSAGE] Key, Value 노드 임베딩 생성 완료")

            # 2. 모든 노드에 공통 속성 이름으로 복사 (GDS 요구사항)
            logging.info("[GraphSAGE] 공통 feature_embedding 속성 생성 중")
            self._copy_to_common_feature_property()
            logging.info("[GraphSAGE] 공통 feature_embedding 속성 생성 완료")

            if self._graph_exists(graph_name):
                self.gds.graph.drop(graph_name)

            existing_rel_types = self._get_existing_relationship_types()
            logging.info(f"[GraphSAGE] 존재하는 관계 타입: {existing_rel_types}")

            relationship_projection = self._build_relationship_projection(existing_rel_types)

            if not relationship_projection:
                raise ValueError("그래프에 관계가 하나도 없습니다")

            # 노드 projection with properties (모든 노드에 feature_embedding 보장됨)
            node_projection = {
                "Document": {"properties": ["feature_embedding"]},
                "Key": {"properties": ["feature_embedding"]},
                "Value": {"properties": ["feature_embedding"]}
            }

            G, result = self.gds.graph.project(
                graph_name,
                node_projection,
                relationship_projection
            )

            logging.info(
                f"[GraphSAGE] Graph projection 완료: {result['nodeCount']} nodes, "
                f"{result['relationshipCount']} relationships"
            )

            # GraphSAGE 훈련
            logging.info("[GraphSAGE] 모델 훈련 시작")
            model_name = "graphsage_model"

            train_result = self.gds.beta.graphSage.train(
                G,
                modelName=model_name,
                featureProperties=["feature_embedding"],  # 모든 노드에 공통된 속성 사용
                embeddingDimension=embedding_dim,
                aggregator="mean",
                activationFunction="relu",
                sampleSizes=[25, 10],
                epochs=1,
                batchSize=100,
                randomSeed=42
            )

            logging.info(f"[GraphSAGE] 훈련 완료: {train_result}")

            # GraphSAGE mutate (임베딩 생성)
            logging.info("[GraphSAGE] 임베딩 생성 시작")
            mutate_result = self.gds.beta.graphSage.mutate(
                G,
                modelName=model_name,
                mutateProperty="graph_embedding"
            )

            logging.info(f"[GraphSAGE] mutate 완료: {mutate_result}")

            embeddings = self._extract_all_document_embeddings(G)
            logging.info(f"[GraphSAGE] 총 {len(embeddings)}개 Document 임베딩 추출 완료")

            # 모델 삭제
            try:
                self.gds.model.drop(model_name)
            except Exception:
                pass

            self.gds.graph.drop(G)

            return embeddings

        except (ServiceUnavailable, ClientError):
            logging.exception("[GraphSAGE] 전체 그래프 임베딩 생성 중 DB 오류")
            if self._graph_exists(graph_name):
                self.gds.graph.drop(graph_name)
            return {}
        except (ValueError, RuntimeError, KeyError):
            logging.exception("[GraphSAGE] 전체 그래프 임베딩 생성 중 처리 오류")
            if self._graph_exists(graph_name):
                self.gds.graph.drop(graph_name)
            return {}

    @staticmethod
    def _build_relationship_projection(existing_rel_types: list) -> dict:
        """관계 projection 설정 빌드 (Key-Value-Document 구조만)

        Centrality 노드는 임베딩이 없으므로 제외:
        - HAS_VALUE: Key -[:HAS_VALUE]-> Value
        - EXTRACTED_FROM: Value -[:EXTRACTED_FROM]-> Document
        """
        relationship_projection = {}

        # HAS_KEY는 Centrality -[:HAS_KEY]-> Key 이므로 제외
        if "HAS_VALUE" in existing_rel_types:
            relationship_projection["HAS_VALUE"] = {"orientation": "UNDIRECTED"}
        if "EXTRACTED_FROM" in existing_rel_types:
            relationship_projection["EXTRACTED_FROM"] = {"orientation": "UNDIRECTED"}

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

    def _add_key_embeddings(self):
        """모든 Key 노드에 name_embedding 추가"""
        # 1. 모든 Key 노드 조회
        query = "MATCH (k:Key) RETURN k.name as name, elementId(k) as id"
        keys = self.graph_repo.execute_query(query)

        logging.info(f"[GraphSAGE] {len(keys)}개 Key 노드 임베딩 생성 중")

        # 2. 배치 처리로 임베딩 생성 및 저장
        batch_size = 100
        with tqdm(total=len(keys), desc="Key 임베딩 생성", unit="keys") as pbar:
            for i in range(0, len(keys), batch_size):
                batch = keys[i:i + batch_size]

                # 임베딩 생성
                embeddings_data = []
                for key in batch:
                    name = key["name"]
                    key_id = key["id"]

                    try:
                        embedding = embedding_service.generate_embedding(name, show_progress_bar=False)
                        embeddings_data.append({"id": key_id, "embedding": embedding})
                    except Exception as e:
                        logging.warning(f"Key '{name}' 임베딩 생성 실패: {e}")
                        continue

                # 배치 업데이트
                if embeddings_data:
                    update_query = """
                    UNWIND $data AS row
                    MATCH (k:Key)
                    WHERE elementId(k) = row.id
                    SET k.name_embedding = row.embedding
                    """
                    self.graph_repo.execute_query(update_query, {"data": embeddings_data})

                pbar.update(len(batch))

        logging.info(f"[GraphSAGE] Key 노드 임베딩 생성 완료")

    def _add_value_embeddings(self):
        """모든 Value 노드에 value_embedding 추가"""
        # 1. 모든 Value 노드 조회
        query = "MATCH (v:Value) RETURN v.value as value, elementId(v) as id"
        values = self.graph_repo.execute_query(query)

        logging.info(f"[GraphSAGE] {len(values)}개 Value 노드 임베딩 생성 중")

        # 2. 배치 처리로 임베딩 생성 및 저장
        batch_size = 100
        with tqdm(total=len(values), desc="Value 임베딩 생성", unit="values") as pbar:
            for i in range(0, len(values), batch_size):
                batch = values[i:i + batch_size]

                # 임베딩 생성
                embeddings_data = []
                for value in batch:
                    value_text = str(value["value"])  # 숫자도 문자열로 변환
                    value_id = value["id"]

                    try:
                        embedding = embedding_service.generate_embedding(value_text, show_progress_bar=False)
                        embeddings_data.append({"id": value_id, "embedding": embedding})
                    except Exception as e:
                        logging.warning(f"Value '{value_text}' 임베딩 생성 실패: {e}")
                        continue

                # 배치 업데이트
                if embeddings_data:
                    update_query = """
                    UNWIND $data AS row
                    MATCH (v:Value)
                    WHERE elementId(v) = row.id
                    SET v.value_embedding = row.embedding
                    """
                    self.graph_repo.execute_query(update_query, {"data": embeddings_data})

                pbar.update(len(batch))

        logging.info(f"[GraphSAGE] Value 노드 임베딩 생성 완료")

    def _copy_to_common_feature_property(self):
        """모든 노드 타입의 임베딩을 공통 feature_embedding 속성으로 복사

        GDS GraphSAGE는 모든 노드에 동일한 속성 이름이 필요하므로:
        - Document.binary_text_embedding → Document.feature_embedding (없으면 0 벡터)
        - Key.name_embedding → Key.feature_embedding (없으면 0 벡터)
        - Value.value_embedding → Value.feature_embedding (없으면 0 벡터)
        """
        zero_vector = [0.0] * 768  # 768차원 0 벡터

        # Document 노드 (모든 노드에 보장)
        query_doc = """
        MATCH (d:Document)
        SET d.feature_embedding = CASE
            WHEN d.binary_text_embedding IS NOT NULL THEN d.binary_text_embedding
            ELSE $zero_vector
        END
        """
        self.graph_repo.execute_query(query_doc, {"zero_vector": zero_vector})

        # Key 노드 (모든 노드에 보장)
        query_key = """
        MATCH (k:Key)
        SET k.feature_embedding = CASE
            WHEN k.name_embedding IS NOT NULL THEN k.name_embedding
            ELSE $zero_vector
        END
        """
        self.graph_repo.execute_query(query_key, {"zero_vector": zero_vector})

        # Value 노드 (모든 노드에 보장)
        query_value = """
        MATCH (v:Value)
        SET v.feature_embedding = CASE
            WHEN v.value_embedding IS NOT NULL THEN v.value_embedding
            ELSE $zero_vector
        END
        """
        self.graph_repo.execute_query(query_value, {"zero_vector": zero_vector})

        logging.info("[GraphSAGE] 모든 노드에 feature_embedding 속성 복사 완료 (누락 노드는 0 벡터)")
