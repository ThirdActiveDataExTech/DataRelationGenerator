import uuid
from typing import Any, Dict

import logging
from langchain_core.tools import tool

from app.src.meta_extract.llm_response import LLMResponse


@tool
def metadata_extract_tool(text: str) -> Dict[str, Any]:
    """텍스트에서 메타데이터를 추출하는 도구"""
    try:
        logging.info("Tool: 텍스트에서 메타데이터 추출")
        llm_response = LLMResponse("meta_extract")

        metadata: Dict[str, Any] | str = llm_response.generate_llm_response(
            variables={"data_text": text}
        )

        if not isinstance(metadata, dict):
            raise ValueError("LLM returned non-dict metadata")

        return {
            "status": "success",
            "metadata": metadata,
            "message": (
                f"메타데이터 추출 완료: "
                f"topics={len(metadata.get('topics', []))}, "
                f"tags={len(metadata.get('tags', []))}"
            )
        }

    except Exception as e:
        logging.error(f"메타데이터 추출 실패: {e}")
        return {
            "status": "error",
            "metadata": {},
            "message": f"메타데이터 추출 실패: {str(e)}"
        }


@tool
def cypher_generate_tool(metadata: Dict[str, Any], text: str) -> Dict[str, Any]:
    """메타데이터를 바탕으로 Neo4j Cypher 쿼리를 생성하는 도구"""
    try:
        logging.info("Tool: Neo4j Cypher 쿼리 생성")
        llm_response = LLMResponse("cypher_generate")

        cypher_text = llm_response.generate_llm_response(
            variables={"metadata": metadata, "text": text}
        )

        if isinstance(cypher_text, dict):
            cypher_text = cypher_text.get("cypher_query", "")

        queries = [q.strip().replace("apoc.create.uuid()", "randomUUID()")
                   for q in cypher_text.split(';') if q.strip()]

        return {
            "status": "success",
            "queries": queries,
            "message": f"Cypher 쿼리 생성 완료: {len(queries)}개 쿼리"
        }
    except Exception as e:
        logging.error(f"Cypher 쿼리 생성 실패: {e}")
        return {
            "status": "error",
            "queries": [],
            "message": f"Cypher 쿼리 생성 실패: {str(e)}"
        }


@tool
def hardcoded_fallback_tool(metadata: Dict[str, Any], text: str) -> Dict[str, Any]:
    """하드코딩된 Cypher 쿼리를 생성하는 폴백 도구"""
    try:
        logging.info("Tool: 하드코딩된 Cypher 쿼리 생성 (폴백)")

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

        # Topics 처리
        if topics:
            topics_str = "', '".join([t.replace("'", "\\'") for t in set(topics)])
            cypher_query += f"""
            UNWIND ['{topics_str}'] as topic_name
            MERGE (t:Topic {{name: topic_name}})
            MERGE (d)-[:HAS_TOPIC]->(t)
            """

        # Tags 처리
        if tags:
            tags_str = "', '".join([t.replace("'", "\\'") for t in set(tags)])
            cypher_query += f"""
            WITH d
            UNWIND ['{tags_str}'] as tag_name
            MERGE (tag:Tag {{name: tag_name}})
            MERGE (d)-[:HAS_TAG]->(tag)
            """

        # Description 처리
        if description.strip():
            cypher_query += f"""
            WITH d
            MERGE (desc:Description {{content: '{escaped_description}'}})
            MERGE (d)-[:HAS_DESCRIPTION]->(desc)
            """

        cypher_query += "\nRETURN d.id as document_id"

        return {
            "status": "success",
            "queries": [cypher_query.strip()],
            "message": "하드코딩된 폴백 쿼리 생성 완료"
        }

    except Exception as e:
        logging.error(f"폴백 쿼리 생성 실패: {e}")
        return {
            "status": "error",
            "queries": [],
            "message": f"폴백 쿼리 생성 실패: {str(e)}"
        }
