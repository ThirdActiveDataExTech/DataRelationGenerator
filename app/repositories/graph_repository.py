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
