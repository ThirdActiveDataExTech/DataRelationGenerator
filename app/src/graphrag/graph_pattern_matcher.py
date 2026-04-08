from typing import List, Dict, Any, Optional

import logging

from app.repositories.graph_repository import GraphRepository
from app.src.embedding.embedding_service import embedding_service
from app.src.meta_extract.llm_response import LLMResponse


class GraphPatternMatcher:
    """Graph Pattern Matching 서비스"""

    # 교집합으로 인한 결과 부족을 고려한 검색 확장 계수
    SEARCH_EXPANSION_FACTOR = 5

    def __init__(self, graph_repo: GraphRepository):
        """초기화"""
        self.graph_repo = graph_repo

    def search_by_graph_pattern(
            self,
            query: str,
            top_k: int = 50
    ) -> Dict[str, Any]:
        """Query를 Graph Pattern으로 변환하여 검색

        Args:
            query: 사용자 쿼리
            top_k: 반환할 Document ID 최대 개수

        Returns:
            {
                "doc_ids": Document ID 리스트,
                "matched_values": 매칭된 Value 정보 리스트 [{"value_id": str, "key_name": str, "value_name": str}, ...]
            }
        """
        # 1. LLM에게 필요한 Key 추론 요청
        key_inference = self._infer_required_keys(query)

        if not key_inference or not key_inference.get("keys"):
            logging.info("  └─ Key 추론 실패")
            return {"doc_ids": [], "matched_values": []}

        required_keys = key_inference.get("keys", [])
        search_terms = key_inference.get("search_terms", {})

        logging.info(f"  ├─ 추론된 Key: {required_keys}")
        logging.info(f"  └─ 검색 조건: {search_terms}")

        # 2. Key 이름 → 실제 Key 노드 매칭 (벡터 검색)
        matched_keys = self._match_keys_by_name(required_keys)

        if not matched_keys:
            logging.info("  └─ Key 매칭 실패")
            return {"doc_ids": [], "matched_values": []}

        matched_key_names = [k["key_name"] for k in matched_keys]
        logging.info(f"  └─ 매칭된 Key: {matched_key_names}")

        # 3. 각 Key-검색어 조합으로 Document 검색 (AND 조건)
        # 교집합을 위해 각 Key별로 충분히 많은 결과를 가져옴
        search_limit = top_k * self.SEARCH_EXPANSION_FACTOR
        all_doc_ids: set[str] | None = None
        all_matched_values: List[Dict[str, Any]] = []

        logging.info(f"  ├─ 검색 시작 (AND 조건)")
        for idx, key_info in enumerate(matched_keys, 1):
            key_name = key_info["key_name"]
            original_key = key_info["original_key"]
            search_term = search_terms.get(original_key, query)  # fallback to full query

            # 검색 조건 포맷팅
            if isinstance(search_term, dict):
                condition_str = f"{search_term.get('operator', '==')} {search_term.get('value')}"
            else:
                condition_str = f"~= '{search_term}'"

            logging.info(f"  │  [{idx}] Key='{key_name}', Condition={condition_str}")
            search_result = self._search_documents_by_key_value(key_name, search_term, search_limit)
            doc_ids = search_result["doc_ids"]
            matched_values = search_result["matched_values"]

            logging.info(f"  │      → {len(doc_ids)}개 문서, {len(matched_values)}개 Value")

            # Value 정보 수집
            all_matched_values.extend(matched_values)

            # 첫 번째 Key는 그대로, 이후는 교집합(AND)
            if all_doc_ids is None:
                all_doc_ids = set(doc_ids)
            else:
                prev_count = len(all_doc_ids)
                all_doc_ids = all_doc_ids.intersection(doc_ids)
                logging.info(f"  │      → 교집합: {prev_count}개 ∩ {len(doc_ids)}개 = {len(all_doc_ids)}개")

                # 조기 종료: 교집합이 빈 집합이면 더 이상 검색 불필요
                if len(all_doc_ids) == 0:
                    logging.info(f"  └─ 교집합 0개, 조기 종료")
                    break

        result_doc_ids = list(all_doc_ids) if all_doc_ids is not None else []
        result_doc_ids = result_doc_ids[:top_k]

        # Value ID 중복 제거 (상위 top_k개만)
        unique_values = {v["value_id"]: v for v in all_matched_values}.values()
        result_values = list(unique_values)[:top_k]

        logging.info(f"  └─ 최종 결과: {len(result_doc_ids)}개 문서, {len(result_values)}개 고유 Value")

        return {
            "doc_ids": result_doc_ids,
            "matched_values": result_values
        }

    def _infer_required_keys(self, query: str) -> Dict[str, Any]:
        """LLM에게 필요한 Key들 추론 요청"""
        try:
            result = LLMResponse("query_key_inference").generate_llm_response(
                variables={"query": query}
            )

            if isinstance(result, dict):
                return result

            logging.warning(f"[Key Inference] 결과가 dict가 아님: {type(result)}")
            return {}

        except Exception as e:
            logging.exception(f"[Key Inference] 실패: {e}")
            return {}

    def _match_keys_by_name(
            self,
            key_names: List[str],
            threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """추론된 Key 이름들을 실제 Key 노드에 벡터 매칭

        Args:
            key_names: 추론된 Key 이름 목록 (예: ["병원명", "지역"])
            threshold: 최소 유사도 (기본: 0.6)

        Returns:
            매칭된 Key 정보 목록 [{original_key, key_name, score}, ...]
        """
        matched = []

        for key_name in key_names:
            # Key 이름 임베딩 생성
            key_embedding = embedding_service.generate_embedding(key_name)

            # Key.name_embedding과 유사도 검색
            result = self._find_similar_key(key_embedding, threshold)

            if result:
                matched.append({
                    "original_key": key_name,
                    "key_name": result["key_name"],
                    "score": result["score"]
                })

        return matched

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

    def _search_documents_by_key_value(
            self,
            key_name: str,
            search_term: Any,
            top_k: int = 50
    ) -> Dict[str, Any]:
        """특정 Key의 Value에서 검색하여 Document 반환

        Args:
            key_name: Key 이름 (예: "BIZPLC NM", "병상수")
            search_term: 검색어 또는 숫자 조건
                - 문자열: 벡터 유사도 검색 (예: "365매일한방병원")
                - dict: 숫자 조건 (예: {"operator": ">=", "value": 100})
            top_k: 최대 반환 개수

        Returns:
            {
                "doc_ids": Document ID 목록,
                "matched_values": 매칭된 Value 정보 [{"value_id": str, "key_name": str, "value_name": str}, ...]
            }
        """
        # 숫자 조건인 경우
        if isinstance(search_term, dict) and "operator" in search_term and "value" in search_term:
            return self._search_by_numeric_condition(key_name, search_term, top_k)

        # 텍스트 검색인 경우
        return self._search_by_text_similarity(key_name, str(search_term), top_k)

    def _search_by_numeric_condition(
            self,
            key_name: str,
            condition: Dict[str, Any],
            top_k: int = 50
    ) -> Dict[str, Any]:
        """숫자 조건으로 Document 검색

        Args:
            key_name: Key 이름
            condition: {"operator": ">=", "value": 100}
            top_k: 최대 반환 개수

        Returns:
            {
                "doc_ids": Document ID 목록,
                "matched_values": 매칭된 Value 정보
            }
        """
        operator = condition.get("operator", "==")
        value = condition.get("value")

        if value is None:
            logging.warning(f"[Numeric Search] value가 없음: {condition}")
            return {"doc_ids": [], "matched_values": []}

        # value를 float로 변환 (예외 처리)
        try:
            numeric_value = float(value)
        except (ValueError, TypeError) as e:
            logging.warning(f"[Numeric Search] value를 float로 변환 실패: {value}, {e}")
            return {"doc_ids": [], "matched_values": []}

        # Cypher 연산자 템플릿 (SQL injection 방지)
        query_templates = {
            ">=": "toFloat(clean_value) >= $value",
            ">": "toFloat(clean_value) > $value",
            "<=": "toFloat(clean_value) <= $value",
            "<": "toFloat(clean_value) < $value",
            "==": "toFloat(clean_value) = $value",
            "=": "toFloat(clean_value) = $value"
        }

        if operator not in query_templates:
            logging.warning(f"[Numeric Search] 잘못된 연산자: {operator}")
            return {"doc_ids": [], "matched_values": []}

        # 디버그: DEBUG 레벨일 때만 Value 샘플 조회
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            debug_query = """
            MATCH (k:Key {name: $key_name})-[:HAS_VALUE]->(v:Value)
            RETURN properties(v) as all_properties, labels(v) as labels
            LIMIT 5
            """
            try:
                debug_results = self.graph_repo.execute_query(debug_query, {"key_name": key_name})
                logging.debug(f"[DEBUG] Key '{key_name}'의 Value 노드 전체 속성:")
                for idx, row in enumerate(debug_results):
                    logging.debug(f"  [{idx}] {row}")
            except Exception as e:
                logging.debug(f"[DEBUG] Value 샘플 조회 실패: {e}")

        # Value.name을 숫자로 변환하여 비교
        query = f"""
        MATCH (k:Key {{name: $key_name}})-[:HAS_VALUE]->(v:Value)-[:EXTRACTED_FROM]->(d:Document)
        WITH v, d, replace(v.name, ',', '') as clean_value
        WHERE toFloatOrNull(clean_value) IS NOT NULL
          AND {query_templates[operator]}
        RETURN DISTINCT d.id as doc_id, v.id as value_id, v.name as value_name
        LIMIT $top_k
        """

        try:
            results = self.graph_repo.execute_query(
                query,
                {"key_name": key_name, "value": numeric_value, "top_k": top_k}
            )

            doc_ids = [row["doc_id"] for row in results]
            matched_values = [
                {
                    "value_id": row["value_id"],
                    "key_name": key_name,
                    "value_name": row["value_name"],
                    "score": 1.0  # 숫자 조건은 binary match
                }
                for row in results
            ]

            return {"doc_ids": doc_ids, "matched_values": matched_values}

        except Exception as e:
            logging.error(f"Numeric search 실패: {e}")
            return {"doc_ids": [], "matched_values": []}

    def _search_by_text_similarity(
            self,
            key_name: str,
            search_term: str,
            top_k: int = 50
    ) -> Dict[str, Any]:
        """텍스트 유사도로 Document 검색 (기존 벡터 검색 로직)

        Args:
            key_name: Key 이름
            search_term: 검색어
            top_k: 최대 반환 개수

        Returns:
            {
                "doc_ids": Document ID 목록,
                "matched_values": 매칭된 Value 정보
            }
        """
        # 1. 검색어 임베딩 생성
        search_embedding = embedding_service.generate_embedding(search_term)

        # 디버그: Key와 연결된 Value 개수 확인
        debug_query = """
        MATCH (k:Key {name: $key_name})-[:HAS_VALUE]->(v:Value)
        RETURN count(v) as total_values, collect(v.name)[0..5] as sample_values
        """
        debug_result = self.graph_repo.execute_query(debug_query, {"key_name": key_name})
        if debug_result:
            logging.info(f"  │  └─ Key '{key_name}'의 Value 개수: {debug_result[0]['total_values']}개, 샘플: {debug_result[0]['sample_values']}")

        # 2. Value 벡터 검색 + Key 필터링
        vector_limit = max(top_k * 3, 50)  # 최소 50개는 검색
        query = """
        MATCH (k:Key {name: $key_name})-[:HAS_VALUE]->(v:Value)
        WITH collect(elementId(v)) as valid_value_ids

        CALL db.index.vector.queryNodes('value_embedding_index', $vector_limit, $search_embedding)
        YIELD node, score
        WHERE elementId(node) IN valid_value_ids

        WITH node as v, score
        MATCH (v)-[:EXTRACTED_FROM]->(d:Document)
        RETURN DISTINCT d.id as doc_id, v.id as value_id, v.name as value_name, max(score) as max_score
        ORDER BY max_score DESC
        LIMIT $top_k
        """

        try:
            results = self.graph_repo.execute_query(
                query,
                {
                    "key_name": key_name,
                    "search_embedding": search_embedding,
                    "vector_limit": vector_limit,
                    "top_k": top_k
                }
            )

            if results:
                logging.info(f"  │      → 벡터 검색 결과: {len(results)}개 (최고 유사도: {results[0].get('max_score', 0):.3f})")
            else:
                logging.warning(f"  │      → 벡터 검색 결과: 0개")

            doc_ids = [row["doc_id"] for row in results]
            matched_values = [
                {
                    "value_id": row["value_id"],
                    "key_name": key_name,
                    "value_name": row["value_name"],
                    "score": row.get("max_score", 0.0)  # 벡터 유사도 포함
                }
                for row in results
            ]

            return {"doc_ids": doc_ids, "matched_values": matched_values}

        except Exception as e:
            logging.error(f"Text search 실패: {e}")
            return {"doc_ids": [], "matched_values": []}
