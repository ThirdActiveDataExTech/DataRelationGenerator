import uuid
from typing import Dict, Any, List

import logging

from app.repositories.graph_repository import GraphRepository
from app.src.embedding.embedding_service import EmbeddingService
from app.src.meta_extract.entity_resolution_service import EntityResolutionService


class GraphBuilder:
    """Neo4j Graph DB 구축 서비스 클래스"""

    def __init__(self, graph_repo: GraphRepository = None):
        """Init 매서드 - GraphRepository 인스턴스 생성"""
        self.graph_repo = graph_repo or GraphRepository()
        self.embedding_service = EmbeddingService()
        self.entity_resolution_service = EntityResolutionService(graph_repo=self.graph_repo)

    def build_graph(
            self,
            metadata: Dict[str, Any],
            text: str,
            categories: List[Dict[str, Any]],
            text_embedding: List[float],
            graph_embedding: List[float] = None
    ) -> Dict[str, Any]:
        """메타데이터로부터 Graph 구축 (Document + Embeddings + Categories)

        Args:
            metadata: 메타데이터
            text: 문서 텍스트
            categories: 카테고리 목록
            text_embedding: 텍스트 기반 임베딩 (768차원)
            graph_embedding: 그래프 기반 임베딩 (128차원, optional)
        """
        doc_id = self._create_document(metadata, text, text_embedding, graph_embedding)
        self._create_categories(categories)

        # Entity Resolution 수행
        entities = metadata.get("entities", [])
        total_relations = self._resolve_entities(entities)

        return {
            "document_id": doc_id,
            "categories_created": len(categories),
            "entity_relations_created": total_relations
        }

    def _create_document(
            self,
            metadata: Dict[str, Any],
            text: str,
            text_embedding: List[float],
            graph_embedding: List[float] = None
    ) -> str:
        """Document 노드 및 Entity/Topic 관계 생성

        Args:
            metadata: 메타데이터
            text: 문서 텍스트
            text_embedding: 텍스트 기반 임베딩
            graph_embedding: 그래프 기반 임베딩 (optional)
        """
        doc_id = f"doc_{uuid.uuid4().hex[:12]}"

        # Binary text embedding 생성 (HashGNN용)
        binary_text_embedding = self.embedding_service.binarize_embedding(
            text_embedding, method="threshold"
        )

        # Entity별 임베딩 생성
        entities = metadata.get("entities", [])
        entities_with_embeddings = []
        for entity in entities:
            entity_emb = self.embedding_service.generate_embedding(entity)
            entity_binary_emb = self.embedding_service.binarize_embedding(
                entity_emb, method="threshold"
            )
            entities_with_embeddings.append({
                "name": entity,
                "binary_embedding": entity_binary_emb
            })

        # graph_embedding이 있으면 함께 저장
        if graph_embedding:
            doc_query = """
            CREATE (d:Document {
                id: $doc_id,
                text: $text,
                description: $description,
                text_embedding: $text_embedding,
                binary_text_embedding: $binary_text_embedding,
                graph_embedding: $graph_embedding,
                created_at: datetime()
            })
            WITH d
            UNWIND $topics AS topic_name
            MERGE (t:Topic {name: topic_name})
            MERGE (d)-[:HAS_TOPIC]->(t)
            WITH d
            UNWIND $entities_with_embeddings AS entity_data
            MERGE (e:Entity {name: entity_data.name})
            ON CREATE SET
                e.binary_embedding = entity_data.binary_embedding,
                e.created_at = datetime()
            MERGE (d)-[:MENTIONS]->(e)
            WITH d, e
            UNWIND $topics AS topic_name
            MERGE (t:Topic {name: topic_name})
            MERGE (e)-[:HAS_TOPIC]->(t)
            WITH d
            RETURN d.id as document_id
            """
            doc_params = {
                "doc_id": doc_id,
                "text": text,
                "description": metadata.get("description", ""),
                "topics": metadata.get("topics", []),
                "entities_with_embeddings": entities_with_embeddings,
                "text_embedding": text_embedding,
                "binary_text_embedding": binary_text_embedding,
                "graph_embedding": graph_embedding
            }
        else:
            # graph_embedding이 없으면 text_embedding, binary_text_embedding만 저장
            doc_query = """
            CREATE (d:Document {
                id: $doc_id,
                text: $text,
                description: $description,
                text_embedding: $text_embedding,
                binary_text_embedding: $binary_text_embedding,
                created_at: datetime()
            })
            WITH d
            UNWIND $topics AS topic_name
            MERGE (t:Topic {name: topic_name})
            MERGE (d)-[:HAS_TOPIC]->(t)
            WITH d
            UNWIND $entities_with_embeddings AS entity_data
            MERGE (e:Entity {name: entity_data.name})
            ON CREATE SET
                e.binary_embedding = entity_data.binary_embedding,
                e.created_at = datetime()
            MERGE (d)-[:MENTIONS]->(e)
            WITH d, e
            UNWIND $topics AS topic_name
            MERGE (t:Topic {name: topic_name})
            MERGE (e)-[:HAS_TOPIC]->(t)
            WITH d
            RETURN d.id as document_id
            """
            doc_params = {
                "doc_id": doc_id,
                "text": text,
                "description": metadata.get("description", ""),
                "topics": metadata.get("topics", []),
                "entities_with_embeddings": entities_with_embeddings,
                "text_embedding": text_embedding,
                "binary_text_embedding": binary_text_embedding
            }

        self.graph_repo.execute_query(doc_query, doc_params)
        logging.info(f"Document 생성 완료: {doc_id}, Entities: {len(entities)}")

        return doc_id

    def _create_categories(self, categories: List[Dict[str, Any]]) -> None:
        """Category 노드 및 관계 생성"""
        for cat in categories:
            cat_query = self._build_category_query(cat)
            self.graph_repo.execute_query(cat_query)

        logging.info(f"Category {len(categories)}개 생성 완료")

    @staticmethod
    def _build_category_query(category: Dict[str, Any]) -> str:
        """Category 생성 Cypher 쿼리 빌드"""
        name = category.get("name", "")
        topics = category.get("topics", [])

        parts = [f"MERGE (c:Category {{name: '{name}'}})"]
        for topic in topics:
            relation = topic.get("relation", "BELONGS_TO")
            topic_name = topic.get("topic", "").replace("'", "\\'")
            parts.extend([
                f"WITH c\nMERGE (t:Topic {{name: '{topic_name}'}})",
                f"MERGE (t)-[:{relation}]->(c)"
            ])

        return "\n".join(parts)

    def build_entity_relations(self) -> Dict[str, Any]:
        """Entity 간 RELATED_TO 관계 생성 (구조 기반)

        Score = (공통 Topic 수 × 2) + (공통 Document 수 × 1)
        Threshold: score >= 3
        """
        query = """
        MATCH (e1:Entity)
        MATCH (e2:Entity)
        WHERE id(e1) < id(e2)

        // 공통 Topic 계산
        OPTIONAL MATCH (e1)-[:HAS_TOPIC]->(t:Topic)<-[:HAS_TOPIC]-(e2)
        WITH e1, e2, count(DISTINCT t) as common_topics

        // 공통 Document 계산
        OPTIONAL MATCH (e1)<-[:MENTIONS]-(d:Document)-[:MENTIONS]->(e2)
        WITH e1, e2, common_topics, count(DISTINCT d) as common_docs

        // Score 계산
        WITH e1, e2,
             common_topics * 2 + common_docs * 1 as score,
             common_topics,
             common_docs
        WHERE score >= 3

        // RELATED_TO 관계 생성
        MERGE (e1)-[r:RELATED_TO]-(e2)
        SET r.score = score,
            r.common_topics = common_topics,
            r.common_docs = common_docs,
            r.updated_at = datetime()

        RETURN count(r) as relations_created
        """

        result = self.graph_repo.execute_query(query)
        count = result[0]["relations_created"] if result else 0

        logging.info(f"Entity RELATED_TO 관계 생성 완료: {count}개")
        return {"relations_created": count}

    def _resolve_entities(self, entities: List[str]) -> int:
        """Entity Resolution 수행 - 새 Entity와 기존 그래프 연결

        Args:
            entities: 새로 추출된 Entity 목록

        Returns:
            생성된 관계 개수
        """
        total_relations = 0

        for entity in entities:
            candidates = self.entity_resolution_service.resolve_entity(new_entity=entity)

            if candidates:
                count = self.graph_repo.create_entity_relations(entity, candidates)
                total_relations += count

        logging.info(f"Entity Resolution 완료: {total_relations}개 관계 생성")
        return total_relations
