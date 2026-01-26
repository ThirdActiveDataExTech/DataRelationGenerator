import uuid
from typing import Any, Dict, List

import logging
from langchain_core.tools import tool

from app.config import settings
from app.src.embedding.embedding_service import embedding_service
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

        # Embedding 생성
        description = metadata.get("description", "")
        if not description or not description.strip():
            raise ValueError("Description이 비어있어 embedding을 생성할 수 없습니다")
        embedding = embedding_service.generate_embedding(description)
        logging.info(f"Embedding 생성 완료 (차원: {len(embedding)})")

        # Document 생성 쿼리
        doc_id = f"doc_{uuid.uuid4().hex[:12]}"
        queries = []

        doc_query = f"""
        CREATE (d:Document {{
            id: '{doc_id}',
            text: $text,
            description: $description,
            embedding: $embedding,
            created_at: datetime()
        }})
        WITH d
        UNWIND $topics AS topic_name
        MERGE (t:Topic {{name: topic_name}})
        MERGE (d)-[:HAS_TOPIC]->(t)
        WITH d
        UNWIND $tags AS tag_name
        MERGE (tag:Tag {{name: tag_name}})
        MERGE (d)-[:HAS_TAG]->(tag)
        RETURN d.id as document_id
        """
        queries.append(doc_query.strip())

        # 유사도 관계 생성 쿼리
        similarity_query = f"""
        MATCH (d1:Document {{id: '{doc_id}'}})
        CALL db.index.vector.queryNodes('document_embedding_index', {settings.SIMILARITY_TOP_K}, d1.embedding)
        YIELD node AS d2, score
        WHERE d2.id <> '{doc_id}' AND score >= {settings.SIMILARITY_THRESHOLD}
        MERGE (d1)-[r:SIMILAR_TO]->(d2)
        SET r.similarity_score = score
        RETURN d2.id AS related_doc_id, score
        """
        queries.append(similarity_query.strip())

        return {
            "status": "success",
            "queries": queries,
            "doc_id": doc_id,
            "embedding": embedding,
            "message": f"Cypher 쿼리 생성 완료: {len(queries)}개"
        }
    except Exception as e:
        logging.error(f"Cypher 쿼리 생성 실패: {e}")
        return {
            "status": "error",
            "queries": [],
            "message": f"Cypher 쿼리 생성 실패: {str(e)}"
        }


@tool
def category_generate_tool(topics: List[str], existing_categories: List[str]) -> Dict[str, Any]:
    """Topic 목록에서 Category를 생성하는 도구"""
    try:
        logging.info("Tool: Category 생성")

        category_result = LLMResponse("category_generate").generate_llm_response(
            variables={"topics": topics, "existing_categories": existing_categories}
        )

        if not isinstance(category_result, dict):
            raise ValueError("Category generation failed")

        categories = category_result.get("categories", [])

        queries = []
        for cat in categories:
            name = cat.get("name", "")
            topics_list = cat.get("topics", [])

            parts = [f"MERGE (c:Category {{name: '{name}'}})"]
            for topic in topics_list:
                relation = topic.get("relation", "BELONGS_TO")
                parts.extend([
                    f"WITH c\nMERGE (t:Topic {{name: '{topic.get('topic', '')}'}})",
                    f"MERGE (t)-[:{relation}]->(c)"
                ])
            queries.append("\n".join(parts))

        return {
            "status": "success",
            "categories": categories,
            "queries": queries,
            "message": f"Category {len(categories)}개 생성 완료"
        }
    except Exception as e:
        logging.error(f"Category 생성 실패: {e}")
        return {
            "status": "error",
            "categories": [],
            "queries": [],
            "message": f"Category 생성 실패: {str(e)}"
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
