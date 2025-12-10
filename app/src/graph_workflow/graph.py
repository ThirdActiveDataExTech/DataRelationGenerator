from typing import List, Dict, Any

from app.repositories.graph_repository import GraphRepository


class GraphService:
    """그래프 데이터베이스 비즈니스 로직 서비스"""

    def __init__(self):
        self.graph_repository = GraphRepository()

    def create_document_with_metadata_transaction(self, doc_id: str, sqlite_id: int, text: str,
                                                  description: str, topics: List[str], tags: List[str]) -> Dict[
        str, Any]:
        """Document와 모든 관련 메타데이터를 한 번의 트랜잭션으로 생성"""
        # 단일 쿼리로 Document, Topic, Tag를 모두 생성
        query = """
        CREATE (d:Document {
            id: $doc_id,
            sqlite_id: $sqlite_id,
            text: $text,
            description: $description,
            created_at: datetime()
        })
        WITH d
        UNWIND $topics as topic_name
        MERGE (t:Topic {name: topic_name})
        CREATE (d)-[:HAS_TOPIC]->(t)
        WITH d, collect(DISTINCT t.name) as created_topics
        UNWIND $tags as tag_name
        MERGE (tag:Tag {name: tag_name})
        CREATE (d)-[:HAS_TAG]->(tag)
        WITH d, created_topics, collect(DISTINCT tag.name) as created_tags
        RETURN d.id as document_id, created_topics, created_tags
        """

        result = self.graph_repository.execute_query(query, {
            "doc_id": doc_id,
            "sqlite_id": sqlite_id,
            "text": text,
            "description": description,
            "topics": topics or [],
            "tags": tags or []
        })

        if result:
            return {
                "document_id": result[0]["document_id"],
                "topics": result[0]["created_topics"],
                "tags": result[0]["created_tags"]
            }
        else:
            return {
                "document_id": doc_id,
                "topics": [],
                "tags": []
            }

    def create_document_with_metadata(self, doc_id: str, sqlite_id: int, text: str,
                                      description: str, topics: List[str], tags: List[str]) -> Dict[str, Any]:
        """Document와 모든 관련 메타데이터를 한 번에 생성하는 비즈니스 로직"""
        return self.create_document_with_metadata_transaction(doc_id, sqlite_id, text, description, topics, tags)

    def execute_queries_transaction(self, queries: List[str]) -> bool:
        """여러 쿼리를 트랜잭션으로 실행"""
        return self.graph_repository.execute_queries_transaction(queries)
