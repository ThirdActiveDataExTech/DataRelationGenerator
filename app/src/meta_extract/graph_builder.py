import uuid
from typing import Dict, Any, List

import logging

from app.repositories.graph_repository import GraphRepository


class GraphBuilder:
    """Neo4j Graph DB 구축 서비스 클래스"""

    def __init__(self, graph_repo: GraphRepository = None):
        """Init 매서드 - GraphRepository 인스턴스 생성"""
        self.graph_repo = graph_repo or GraphRepository()

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

        return {
            "document_id": doc_id,
            "categories_created": len(categories)
        }

    def _create_document(
            self,
            metadata: Dict[str, Any],
            text: str,
            text_embedding: List[float],
            graph_embedding: List[float] = None
    ) -> str:
        """Document 노드 및 Topic/Tag 관계 생성

        Args:
            metadata: 메타데이터
            text: 문서 텍스트
            text_embedding: 텍스트 기반 임베딩
            graph_embedding: 그래프 기반 임베딩 (optional)
        """
        doc_id = f"doc_{uuid.uuid4().hex[:12]}"

        # graph_embedding이 있으면 함께 저장
        if graph_embedding:
            doc_query = """
            CREATE (d:Document {
                id: $doc_id,
                text: $text,
                description: $description,
                text_embedding: $text_embedding,
                graph_embedding: $graph_embedding,
                created_at: datetime()
            })
            WITH d
            UNWIND $topics AS topic_name
            MERGE (t:Topic {name: topic_name})
            MERGE (d)-[:HAS_TOPIC]->(t)
            WITH d
            UNWIND $tags AS tag_name
            MERGE (tag:Tag {name: tag_name})
            MERGE (d)-[:HAS_TAG]->(tag)
            RETURN d.id as document_id
            """
            doc_params = {
                "doc_id": doc_id,
                "text": text,
                "description": metadata.get("description", ""),
                "topics": metadata.get("topics", []),
                "tags": metadata.get("tags", []),
                "text_embedding": text_embedding,
                "graph_embedding": graph_embedding
            }
        else:
            # graph_embedding이 없으면 text_embedding만 저장
            doc_query = """
            CREATE (d:Document {
                id: $doc_id,
                text: $text,
                description: $description,
                text_embedding: $text_embedding,
                created_at: datetime()
            })
            WITH d
            UNWIND $topics AS topic_name
            MERGE (t:Topic {name: topic_name})
            MERGE (d)-[:HAS_TOPIC]->(t)
            WITH d
            UNWIND $tags AS tag_name
            MERGE (tag:Tag {name: tag_name})
            MERGE (d)-[:HAS_TAG]->(tag)
            RETURN d.id as document_id
            """
            doc_params = {
                "doc_id": doc_id,
                "text": text,
                "description": metadata.get("description", ""),
                "topics": metadata.get("topics", []),
                "tags": metadata.get("tags", []),
                "text_embedding": text_embedding
            }

        self.graph_repo.execute_query(doc_query, doc_params)
        logging.info(f"Document 생성 완료: {doc_id}")

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
