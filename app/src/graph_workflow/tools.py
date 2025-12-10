import uuid
from datetime import datetime
from typing import Any

import logging
from langchain_core.tools import tool

from app.exceptions.base import ApplicationError
from app.exceptions.meta_extract import MetadataDBError
from app.repositories.graph_repository import GraphRepository
from app.repositories.meta_repository import MetadataDBRepository
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
def sqlite_insert_tool(text: str, metadata: dict, db: Any) -> dict:
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
        graph_repository= GraphRepository()
        success = graph_repository.execute_queries_sequential(queries)

        if success:
            return {
                "status": "success",
                "error": "",
                "update_state": {"execution_status": "success", "cypher_error": ""}
            }
        else:
            return {
                "status": "fail",
                "error": "Neo4j execute queries failed",
                "update_state": {"execution_status": "fail", "cypher_error": "Neo4j execute queries failed"}
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
    """하드코딩된 Cypher 쿼리 생성 (폴백 메커니즘)"""
    logging.info("ToolNode: 하드코딩된 Cypher 쿼리 생성")

    # 기본 Document 노드 생성을 위한 하드코딩된 쿼리
    doc_id = f"doc_fallback_{uuid.uuid4().hex[:8]}"

    # 메타데이터 처리
    topics = metadata.get('topics', []) if metadata else []
    tags = metadata.get('tags', []) if metadata else []
    description = metadata.get('description',
                               'Created by fallback mechanism') if metadata else 'Created by fallback mechanism'

    # 텍스트 이스케이프 처리
    escaped_text = text.replace("'", "\\'").replace('"', '\\"') if text else ""
    escaped_description = description.replace("'", "\\'").replace('"', '\\"')

    # 하드코딩된 Cypher 쿼리 생성
    cypher_query = f"""
    CREATE (d:Document {{
        id: '{doc_id}',
        sqlite_id: 0,
        text: '{escaped_text}',
        description: '{escaped_description}',
        created_at: datetime()
    }})
    WITH d
    """

    # Topics 처리 - MERGE를 사용해서 중복 관계 방지
    if topics:
        # 중복 제거
        unique_topics = list(set(topics))
        topics_str = "', '".join([t.replace("'", "\\'") for t in unique_topics])
        cypher_query += f"""
        UNWIND ['{topics_str}'] as topic_name
        MERGE (t:Topic {{name: topic_name}})
        MERGE (d)-[:HAS_TOPIC]->(t)
        """

    # Tags 처리 - MERGE를 사용해서 중복 관계 방지  
    if tags:
        # 중복 제거
        unique_tags = list(set(tags))
        tags_str = "', '".join([t.replace("'", "\\'") for t in unique_tags])
        cypher_query += f"""
        WITH d
        UNWIND ['{tags_str}'] as tag_name
        MERGE (tag:Tag {{name: tag_name}})
        MERGE (d)-[:HAS_TAG]->(tag)
        """

    # Description 처리 - Description 노드 생성 및 관계 설정
    if description and description.strip():
        cypher_query += f"""
        WITH d
        MERGE (desc:Description {{content: '{escaped_description}'}})
        MERGE (d)-[:HAS_DESCRIPTION]->(desc)
        """

    # 최종 리턴
    cypher_query += "\nRETURN d.id as document_id"

    logging.info(f"Generated hardcoded cypher query: {cypher_query}")

    return {
        "status": "success",
        "queries": [cypher_query.strip()],
        "error": "",
        "update_state": {
            "cypher_queries": [cypher_query.strip()],
            "fallback_retry": True,
            "llm_error": ""
        }
    }
