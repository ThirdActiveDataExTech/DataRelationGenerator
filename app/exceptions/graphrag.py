"""GraphRAG 관련 커스텀 예외"""

import logging
from typing import Optional

from neo4j.exceptions import ServiceUnavailable, TransientError, ClientError


class GraphRAGError(Exception):
    """GraphRAG 기본 예외"""

    def __init__(self, message: str, original_error: Optional[Exception] = None):
        self.message = message
        self.original_error = original_error
        super().__init__(self.message)


class GraphQueryError(GraphRAGError):
    """Neo4j 쿼리 실행 오류 (CypherSyntaxError, ClientError 등)"""
    pass


class GraphConnectionError(GraphRAGError):
    """Neo4j 연결 오류 (ServiceUnavailable, TransientError 등)"""
    pass


class DocumentSearchError(GraphRAGError):
    """Document 검색 오류"""
    pass


class NodeNotFoundError(GraphRAGError):
    """노드를 찾을 수 없음"""

    def __init__(self, node_id: str):
        super().__init__(f"Node not found: {node_id}")
        self.node_id = node_id


class DataParsingError(GraphRAGError):
    """데이터 파싱/변환 오류"""
    pass


def handle_graph_error(e: Exception, context: str) -> None:
    """Neo4j 예외 로깅 (예외 타입별로 구분)

    Args:
        e: 발생한 예외
        context: 에러 발생 컨텍스트 (예: "Document 검색", "그래프 구조 추출")
    """
    if isinstance(e, (ServiceUnavailable, TransientError)):
        logging.exception(f"{context} - DB 연결 오류")
    elif isinstance(e, ClientError):
        logging.exception(f"{context} - 쿼리 실행 오류")
    elif isinstance(e, (ValueError, KeyError, AttributeError)):
        logging.exception(f"{context} - 데이터 처리 오류")
    else:
        logging.exception(f"{context} - 알 수 없는 오류")
