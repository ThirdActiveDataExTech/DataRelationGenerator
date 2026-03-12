from typing import List, Dict, Any, Optional

import logging
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError, CypherSyntaxError

from app.config import settings
from app.exceptions.service import ApplicationError


class GraphRepository:
    """그래프 데이터베이스 도메인 레포지토리"""

    def __init__(self):
        self.driver = None

        try:
            logging.info(f"Neo4j 연결 시도: {settings.NEO4J_URI}")
            self.driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )
            self.driver.verify_connectivity()
            logging.info("Neo4j 연결 성공")

        except ServiceUnavailable as e:
            logging.error(f"Neo4j 서버가 사용 불가능합니다: {e}")
            raise ApplicationError(
                code=503,
                message="Neo4j 서버에 연결할 수 없습니다",
                result={"error_type": "service_unavailable", "detail": str(e)}
            )
        except AuthError as e:
            logging.error(f"Neo4j 인증 실패: {e}")
            raise ApplicationError(
                code=401,
                message="Neo4j 인증 실패 - 사용자명 또는 비밀번호를 확인해주세요",
                result={"error_type": "authentication_failed", "detail": str(e)}
            )

    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """단일 쿼리 실행"""
        if self.driver is None:
            raise ApplicationError(
                code=500,
                message="Neo4j connection not available",
                result={"error_type": "connection_error"}
            )
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except CypherSyntaxError as e:
            logging.error(f"Cypher 쿼리 문법 오류: {e}")
            raise ApplicationError(
                code=400,
                message="잘못된 Cypher 쿼리입니다",
                result={"error_type": "cypher_syntax_error", "detail": str(e)}
            )

    def create_vector_index(self) -> None:
        """Document 노드의 binary_text_embedding property에 대한 vector index 생성"""
        query = """
        CREATE VECTOR INDEX text_embedding_index IF NOT EXISTS
        FOR (d:Document)
        ON d.binary_text_embedding
        OPTIONS {
            indexConfig: {
                `vector.dimensions`: 768,
                `vector.similarity_function`: 'cosine'
            }
        }
        """
        self.execute_query(query)

    def create_graph_embedding_index(self, dimensions: int = 128) -> None:
        """Document 노드의 graph_embedding property에 대한 vector index 생성

        Args:
            dimensions: 그래프 임베딩 차원 (기본: 128)
        """
        query = f"""
        CREATE VECTOR INDEX graph_embedding_index IF NOT EXISTS
        FOR (d:Document)
        ON d.graph_embedding
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {dimensions},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
        self.execute_query(query)

    def create_entity_embedding_index(self) -> None:
        """Entity 노드의 binary_embedding property에 대한 vector index 생성"""
        query = """
        CREATE VECTOR INDEX entity_embedding_index IF NOT EXISTS
        FOR (e:Entity)
        ON e.binary_embedding
        OPTIONS {
            indexConfig: {
                `vector.dimensions`: 768,
                `vector.similarity_function`: 'cosine'
            }
        }
        """
        self.execute_query(query)

    def create_fulltext_index_entity(self) -> None:
        """Entity 노드의 name property에 대한 Full-Text Index 생성 (CJK 분석기)"""
        query = """
        CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
        FOR (n:Entity) ON EACH [n.name]
        OPTIONS {indexConfig: {`fulltext.analyzer`: 'cjk'}}
        """
        self.execute_query(query)

    def create_fulltext_index_topic(self) -> None:
        """Topic 노드의 name property에 대한 Full-Text Index 생성 (CJK 분석기)"""
        query = """
        CREATE FULLTEXT INDEX topic_fulltext IF NOT EXISTS
        FOR (n:Topic) ON EACH [n.name]
        OPTIONS {indexConfig: {`fulltext.analyzer`: 'cjk'}}
        """
        self.execute_query(query)

    def create_document_relations(
            self,
            source_doc_id: str,
            relations: List[Dict[str, Any]]
    ) -> int:
        """Document 간 의미적 관계 생성 (중복 방지)

        Args:
            source_doc_id: 출발 문서 ID
            relations: 관계 정보 리스트
                [{
                    "target_doc_id": "doc_xxx",
                    "relation_type": "EXTENDS",
                    "reason": "이유",
                    "confidence": 0.85
                }]

        Returns:
            생성된 관계 개수
        """
        if not relations:
            return 0

        # APOC 사용 시도 (MERGE로 중복 방지)
        query = """
        UNWIND $relations AS rel
        MATCH (source:Document {id: $source_doc_id})
        MATCH (target:Document {id: rel.target_doc_id})
        WITH source, target, rel
        CALL apoc.merge.relationship(
            source,
            rel.relation_type,
            {},
            {
                reason: rel.reason,
                confidence: rel.confidence,
                updated_at: datetime()
            },
            target,
            {}
        ) YIELD rel AS created_rel
        RETURN count(created_rel) as count
        """

        parameters = {
            "source_doc_id": source_doc_id,
            "relations": relations
        }

        try:
            results = self.execute_query(query, parameters)
            count = results[0]["count"] if results else 0
            logging.info(f"Document {source_doc_id}에 대해 {count}개의 관계 처리 완료")
            return count
        except Exception as e:
            logging.error(f"Document 관계 생성 실패 (APOC): {e}")
            # APOC이 없을 경우 대체 쿼리 시도
            return self._create_document_relations_without_apoc(source_doc_id, relations)

    def _create_document_relations_without_apoc(
            self,
            source_doc_id: str,
            relations: List[Dict[str, Any]]
    ) -> int:
        """APOC 없이 Document 관계 생성 (동적 관계 타입 대체)

        Args:
            source_doc_id: 출발 문서 ID
            relations: 관계 정보 리스트

        Returns:
            생성된 관계 개수
        """
        count = 0

        for rel in relations:
            relation_type = rel["relation_type"]
            query = f"""
            MATCH (source:Document {{id: $source_doc_id}})
            MATCH (target:Document {{id: $target_doc_id}})
            MERGE (source)-[r:{relation_type}]->(target)
            SET r.reason = $reason,
                r.confidence = $confidence,
                r.created_at = datetime()
            RETURN r
            """

            parameters = {
                "source_doc_id": source_doc_id,
                "target_doc_id": rel["target_doc_id"],
                "reason": rel["reason"],
                "confidence": rel["confidence"]
            }

            try:
                self.execute_query(query, parameters)
                count += 1
            except Exception as e:
                logging.error(
                    f"관계 생성 실패 ({source_doc_id} -[{relation_type}]-> {rel['target_doc_id']}): {e}"
                )

        logging.info(f"Document {source_doc_id}에 대해 {count}개의 관계 생성 완료 (APOC 없이)")
        return count

    def create_entity_relations(
            self,
            new_entity: str,
            candidates: List[Dict[str, Any]]
    ) -> int:
        """Entity 간 RELATED_TO 관계 생성"""
        if not candidates:
            return 0

        count = 0

        for candidate in candidates:
            query = """
            MATCH (new:Entity {name: $new_entity})
            MATCH (existing:Entity {name: $existing_entity})
            MERGE (new)-[r:RELATED_TO]-(existing)
            SET r.score = $score,
                r.created_at = datetime(),
                r.source = $source,
                r.wikidata_type = $wikidata_type
            RETURN r
            """

            parameters = {
                "new_entity": new_entity,
                "existing_entity": candidate["name"],
                "score": candidate["score"],
                "source": candidate.get("source", "vector"),
                "wikidata_type": candidate.get("wikidata_type")
            }

            try:
                self.execute_query(query, parameters)
                count += 1
                logging.info(
                    f"Entity 관계 생성: {new_entity} -[RELATED_TO]- {candidate['name']} (score: {candidate['score']:.2f})"
                )
            except Exception as e:
                logging.error(
                    f"Entity 관계 생성 실패 ({new_entity} -[RELATED_TO]- {candidate['name']}): {e}"
                )

        return count
