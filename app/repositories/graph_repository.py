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
        """Document 노드의 embedding property에 대한 vector index 생성"""
        query = """
        CREATE VECTOR INDEX document_embedding_index IF NOT EXISTS
        FOR (d:Document)
        ON d.embedding
        OPTIONS {
            indexConfig: {
                `vector.dimensions`: 768,
                `vector.similarity_function`: 'cosine'
            }
        }
        """
        self.execute_query(query)

    def find_and_create_similarity_relations(
            self,
            doc_id: str,
            embedding: List[float],
            threshold: float = settings.SIMILARITY_THRESHOLD,
            top_k: int = settings.SIMILARITY_TOP_K
    ) -> List[Dict[str, Any]]:
        """
        Vector search로 유사한 문서를 찾고 SIMILAR_TO 관계 생성

        Args:
            doc_id: 현재 문서 ID
            embedding: 현재 문서의 embedding 벡터
            threshold: 유사도 임계값 (0.0 ~ 1.0), None이면 settings에서 가져옴
            top_k: 최대 관계 생성 개수, None이면 settings에서 가져옴

        Returns:
            생성된 관계 정보 리스트
        """

        query = """
        MATCH (d1:Document {id: $doc_id})
        WITH d1
        CALL db.index.vector.queryNodes('document_embedding_index', $top_k, $embedding)
        YIELD node AS d2, score
        WHERE d2.id <> $doc_id AND score >= $threshold
        MERGE (d1)-[r:SIMILAR_TO]->(d2)
        SET r.similarity_score = score
        RETURN d2.id AS related_doc_id, score
        """

        parameters = {
            "doc_id": doc_id,
            "embedding": embedding,
            "threshold": threshold,
            "top_k": top_k
        }

        results = self.execute_query(query, parameters)
        logging.info(f"Document {doc_id}에 대해 {len(results)}개의 유사도 관계 생성")
        return results
