import uuid
from datetime import datetime
from typing import Any

import logging
from langchain_core.tools import tool

from app.exceptions.base import ApplicationError
from app.exceptions.meta_extract import MetadataDBError
from app.repositories.meta_repository import MetadataDBRepository
from app.src.graph_workflow.graph import GraphService
from app.src.meta_extract.llm_response import LLMResponse


@tool
def llm_meta_extract_tool(text: str, prompt_name: str, prompt_label: str) -> dict:
    """LLM을 사용하여 텍스트에서 메타데이터 추출"""
    try:
        logging.info("LLMNode: 텍스트에서 메타데이터 추출 시작")
        llm_response = LLMResponse()
        metadata = llm_response.extract_metadata_by_llm(text, prompt_name, prompt_label)

        return {
            "status": "success",
            "metadata": metadata,
            "error": "",
            "update_state": {"metadata": metadata, "llm_error": ""}
        }
    except ApplicationError as e:
        logging.error(f"Application error in metadata extraction: {e}")
        return {
            "status": "fail",
            "metadata": {},
            "error": f"Application error: {e.message}",
            "update_state": {"llm_error": f"Application error: {e.message}"}
        }


@tool
def llm_cypher_generate_tool(metadata: dict, text: str, prompt_name: str = "generate_cypher_query",
                             prompt_label: str = "production") -> dict:
    """LLM을 사용해 Neo4j Cypher 쿼리 생성"""
    try:
        logging.info("LLMNode: LLM을 사용해 Neo4j Cypher 쿼리 생성 시작")
        llm_response = LLMResponse()

        # LLM을 사용해 Cypher 쿼리 생성
        cypher_text = llm_response.generate_cypher_queries_by_llm(
            metadata=metadata,
            text=text,
            prompt_name=prompt_name,
            prompt_label=prompt_label
        )

        # 단일 쿼리 또는 ';'로 구분된 쿼리들을 리스트로 전환
        if ';' in cypher_text:
            queries = [q.strip() for q in cypher_text.split(';') if q.strip()]
        else:
            queries = [cypher_text.strip()] if cypher_text.strip() else []

        logging.info(f"LLMNode: Neo4j Cypher 쿼리 생성 완료: \n{queries}")

        return {
            "status": "success",
            "queries": queries,
            "error": "",
            "update_state": {"cypher_queries": queries, "llm_error": ""}
        }
    except AttributeError as e:
        logging.error(f"LLM response attribute error: {e}")
        return {
            "status": "fail",
            "queries": [],
            "error": f"LLM response format error: {str(e)}",
            "update_state": {"llm_error": f"LLM response format error: {str(e)}"}
        }


@tool
def sqlite_insert_tool(text: str, metadata: dict, db=Any) -> dict:
    """SQLite에 메타데이터 저장"""
    try:
        repository = MetadataDBRepository()
        saved_data_info = repository.save_metadata_to_db(
            origin_text=text,
            topics=metadata.get("topics", []),
            tags=metadata.get("tags", []),
            description=metadata.get("description", ""),
            extract_time=datetime.now(),
            db=db
        )
        return {
            "status": "success",
            "record_id": saved_data_info.id,
            "error": "",
            "update_state": {"sqlite_status": "success", "sqlite_error": ""}
        }
    except (ValueError, MetadataDBError) as e:
        logging.error(f"Data validation error in SQLite insert: {e}")
        return {
            "status": "fail",
            "record_id": 0,
            "error": f"Data validation error: {str(e)}",
            "update_state": {"sqlite_status": "fail", "sqlite_error": f"Data validation error: {str(e)}"}
        }


@tool
def cypher_execution_tool(queries: list) -> dict:
    """Neo4j Cypher 쿼리 실행"""
    try:
        logging.info("ToolNode: Cypher 쿼리 실행")
        graph_service = GraphService()
        success = graph_service.execute_queries_sequential(queries)

        if success:
            return {
                "status": "success",
                "error": "",
                "update_state": {"execution_status": "success", "cypher_error": ""}
            }
        else:
            return {
                "status": "fail",
                "error": "Neo4j transaction failed",
                "update_state": {"execution_status": "fail", "cypher_error": "Neo4j transaction failed"}
            }

    except ApplicationError as e:
        logging.error(f"Neo4j service error: {e}")
        return {
            "status": "fail",
            "error": f"Neo4j service error: {e.message}",
            "update_state": {"execution_status": "fail", "cypher_error": f"Neo4j service error: {e.message}"}
        }


@tool
def hardcoded_fallback_tool(metadata: dict, text: str) -> dict:
    """Neo4j 클라이언트의 하드코딩된 메서드 사용 (폴백 메커니즘)"""
    logging.info("ToolNode: Neo4j 클라이언트의 하드코딩된 메서드 사용")

    # 기본 Document 노드 생성
    doc_id = f"doc_fallback_{uuid.uuid4().hex[:8]}"

    # GraphService의 비즈니스 로직 메서드 사용
    topics = metadata.get('topics', []) if metadata else []
    tags = metadata.get('tags', []) if metadata else []
    description = metadata.get('description',
                               'Created by fallback mechanism') if metadata else 'Created by fallback mechanism'
    try:
        graph_service = GraphService()
        result = graph_service.create_document_with_metadata(
            doc_id=doc_id,
            sqlite_id=0,  # fallback이므로 0으로 설정
            text=text,
            description=description,
            topics=topics,
            tags=tags
        )

        return {
            "status": "success",
            "error": "",
            "update_state": {
                "execution_status": "success",
                "fallback_retry": True,
                "cypher_error": ""
            }
        }
    except ApplicationError as e:
        logging.error(f"Neo4j service error in fallback: {e}")
        return {
            "status": "fail",
            "error": f"Neo4j service error: {e.message}",
            "update_state": {
                "execution_status": "fail",
                "cypher_error": f"Neo4j service error: {e.message}"
            }
        }
