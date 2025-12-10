from typing import List, Dict, Any, Optional

import logging
from langchain_neo4j import Neo4jGraph
from neo4j.exceptions import ServiceUnavailable, AuthError

from app.config import settings
from app.exceptions.service import ApplicationError


class GraphRepository:
    """그래프 데이터베이스 도메인 레포지토리"""

    def __init__(self):
        self.graph = None

        try:
            logging.info(f"Neo4j 연결 시도: {settings.NEO4J_URI}")
            self.graph = Neo4jGraph(
                url=settings.NEO4J_URI,
                username=settings.NEO4J_USER,
                password=settings.NEO4J_PASSWORD
            )
            # 연결 테스트
            self.graph.query("RETURN 1 as test")
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
        if self.graph is None:
            raise ApplicationError(
                code=500,
                message="Neo4j connection not available",
                result={"error_type": "connection_error"}
            )
        result = self.graph.query(query, parameters or {})
        return result

    def execute_queries_sequential(self, queries: List[str]) -> bool:
        """여러 쿼리를 순차적으로 실행"""
        if self.graph is None:
            raise ApplicationError(
                code=500,
                message="Neo4j connection not available",
                result={"error_type": "connection_error"}
            )
        for query in queries:
            self.graph.query(query)
        logging.info(f"순차 실행 성공: {len(queries)}개 쿼리 실행")
        return True
