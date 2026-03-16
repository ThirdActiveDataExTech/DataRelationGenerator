from typing import List, Dict, Any, Optional

import logging

from app.repositories.graph_repository import GraphRepository
from app.src.embedding.embedding_service import embedding_service
from app.src.meta_extract.llm_response import LLMResponse


class GraphPatternMatcher:
    """Graph Pattern Matching 서비스"""

    def __init__(self, graph_repo: GraphRepository):
        """초기화"""
        self.graph_repo = graph_repo

    def search_by_graph_pattern(
            self,
            query: str,
            top_k: int = 50
    ) -> List[str]:
        """Query를 Graph Pattern으로 변환하여 검색

        Args:
            query: 사용자 쿼리
            top_k: 반환할 Document ID 최대 개수

        Returns:
            Document ID 리스트
        """
        # 1. Query parsing (LLM)
        conditions = self._parse_query(query)

        if not conditions:
            logging.info("[Graph Pattern] 조건 없음, 전체 Document 반환")
            return self._get_all_documents(top_k)

        logging.info(f"[Graph Pattern] 파싱된 조건: {conditions}")

        # 2. Field → Key 매칭 (벡터 유사도)
        mapped_conditions = self._map_fields_to_keys(conditions)

        logging.info(f"[Graph Pattern] Key 매칭 완료: {mapped_conditions}")

        # 3. Cypher 쿼리 생성
        cypher_query = self._build_cypher_query(mapped_conditions, top_k)

        logging.info(f"[Graph Pattern] Cypher 쿼리:\n{cypher_query}")

        # 4. 실행
        results = self.graph_repo.execute_query(cypher_query)

        doc_ids = [row["doc_id"] for row in results]
        logging.info(f"[Graph Pattern] {len(doc_ids)}개 Document 매칭")

        return doc_ids

    def _parse_query(self, query: str) -> List[Dict[str, str]]:
        """Query를 조건 리스트로 파싱 (LLM)"""
        try:
            result = LLMResponse("graph_query_parse").generate_llm_response(
                variables={"query": query}
            )

            if isinstance(result, dict):
                return result.get("conditions", [])

            logging.warning(f"[Query Parse] 결과가 dict가 아님: {type(result)}")
            return []

        except Exception as e:
            logging.exception(f"[Query Parse] 실패: {e}")
            return []

    def _map_fields_to_keys(
            self,
            conditions: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """Field를 Key 이름으로 벡터 유사도 매칭"""
        mapped = []

        for cond in conditions:
            field = cond.get("field", "")
            operator = cond.get("operator", "=")
            value = cond.get("value", "")

            # Field 임베딩 생성
            field_embedding = embedding_service.generate_embedding(field)

            # Key.name_embedding과 유사도 검색 (top 1)
            matched_key = self._find_similar_key(field_embedding)

            if matched_key:
                mapped.append({
                    "original_field": field,
                    "matched_key": matched_key["key_name"],
                    "similarity": matched_key["score"],
                    "operator": operator,
                    "value": value
                })
                logging.info(
                    f"[Field→Key] '{field}' → '{matched_key['key_name']}' "
                    f"(score: {matched_key['score']:.3f})"
                )
            else:
                logging.warning(f"[Field→Key] '{field}'에 매칭되는 Key 없음")

        return mapped

    def _find_similar_key(
            self,
            field_embedding: List[float],
            threshold: float = 0.6
    ) -> Optional[Dict[str, Any]]:
        """Key.name_embedding과 벡터 유사도 검색

        Args:
            field_embedding: field 임베딩 벡터
            threshold: 최소 유사도 (0.6 미만은 매칭 안함)

        Returns:
            {key_name: str, score: float} 또는 None
        """
        # Neo4j vector index 사용
        query = """
        CALL db.index.vector.queryNodes('key_name_embedding_index', 1, $embedding)
        YIELD node, score
        RETURN node.name as key_name, score
        """

        try:
            results = self.graph_repo.execute_query(
                query,
                {"embedding": field_embedding}
            )

            if results and results[0]["score"] >= threshold:
                return {
                    "key_name": results[0]["key_name"],
                    "score": results[0]["score"]
                }

            return None

        except Exception as e:
            logging.warning(f"[Vector Search] Key 검색 실패: {e}")
            return None

    def _build_cypher_query(
            self,
            conditions: List[Dict[str, Any]],
            top_k: int
    ) -> str:
        """매칭된 조건으로 Cypher 쿼리 생성"""
        if not conditions:
            return f"MATCH (d:Document) RETURN d.id as doc_id LIMIT {top_k}"

        # WHERE 절 구성
        where_clauses = []

        for i, cond in enumerate(conditions):
            key_name = cond["matched_key"]
            operator = cond["operator"]
            value = cond["value"]

            if operator == "=":
                where_clauses.append(
                    f"v{i}.value = '{value}'"
                )
            elif operator == "!=":
                where_clauses.append(
                    f"v{i}.value <> '{value}'"
                )
            elif operator == "CONTAINS":
                where_clauses.append(
                    f"v{i}.value CONTAINS '{value}'"
                )
            elif operator in [">", ">=", "<", "<="]:
                # 숫자 비교
                where_clauses.append(
                    f"toInteger(v{i}.value) {operator} {value}"
                )

        # MATCH 절 구성
        match_clauses = []
        for i, cond in enumerate(conditions):
            key_name = cond["matched_key"]
            match_clauses.append(
                f"MATCH (d)<-[:EXTRACTED_FROM]-(v{i}:Value)<-[:HAS_VALUE]-(k{i}:Key {{name: '{key_name}'}})"
            )

        # 최종 쿼리
        query = "\n".join(match_clauses)
        query += f"\nWHERE {' AND '.join(where_clauses)}"
        query += f"\nRETURN DISTINCT d.id as doc_id LIMIT {top_k}"

        return query

    def _get_all_documents(self, top_k: int) -> List[str]:
        """조건 없을 때 전체 Document 반환"""
        query = f"MATCH (d:Document) RETURN d.id as doc_id LIMIT {top_k}"
        results = self.graph_repo.execute_query(query)
        return [row["doc_id"] for row in results]
