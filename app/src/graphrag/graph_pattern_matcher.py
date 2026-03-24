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
        # 1. LLM에게 필요한 Key 추론 요청
        key_inference = self._infer_required_keys(query)

        if not key_inference or not key_inference.get("keys"):
            logging.info("[Graph Pattern] Key 추론 실패, 빈 결과 반환")
            return []

        required_keys = key_inference.get("keys", [])
        search_terms = key_inference.get("search_terms", {})

        logging.info(f"[Graph Pattern] 추론된 Key: {required_keys}")
        logging.info(f"[Graph Pattern] 검색어: {search_terms}")

        # 2. Key 이름 → 실제 Key 노드 매칭 (벡터 검색)
        matched_keys = self._match_keys_by_name(required_keys)

        if not matched_keys:
            logging.info("[Graph Pattern] Key 매칭 실패, 빈 결과 반환")
            return []

        logging.info(f"[Graph Pattern] 매칭된 Key: {matched_keys}")

        # 3. 각 Key-검색어 조합으로 Document 검색
        all_doc_ids = set()

        for key_info in matched_keys:
            key_name = key_info["key_name"]
            original_key = key_info["original_key"]
            search_term = search_terms.get(original_key, query)  # fallback to full query

            doc_ids = self._search_documents_by_key_value(key_name, search_term, top_k)
            logging.info(
                f"[Graph Pattern] Key '{key_name}' + Term '{search_term}' → {len(doc_ids)}개 Document"
            )
            all_doc_ids.update(doc_ids)

        result_doc_ids = list(all_doc_ids)[:top_k]
        logging.info(f"[Graph Pattern] 총 {len(result_doc_ids)}개 Document 매칭")

        return result_doc_ids

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
                logging.info(
                    f"[Key 매칭] '{key_name}' → '{result['key_name']}' "
                    f"(score: {result['score']:.3f})"
                )
            else:
                logging.warning(f"[Key 매칭] '{key_name}'에 매칭되는 Key 없음 (threshold: {threshold})")

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
            search_term: str,
            top_k: int = 50
    ) -> List[str]:
        """특정 Key의 Value에서 벡터 유사도로 검색하여 Document 반환

        Args:
            key_name: Key 이름 (예: "BIZPLC NM")
            search_term: 검색어 (예: "365매일한방병원")
            top_k: 최대 반환 개수

        Returns:
            Document ID 목록
        """
        # 1. 검색어 임베딩 생성
        search_embedding = embedding_service.generate_embedding(search_term)

        # 2. Value 벡터 검색 + Key 필터링
        query = """
        MATCH (k:Key {name: $key_name})-[:HAS_VALUE]->(v:Value)
        WITH collect(elementId(v)) as valid_value_ids

        CALL db.index.vector.queryNodes('value_embedding_index', $top_k * 3, $search_embedding)
        YIELD node, score
        WHERE elementId(node) IN valid_value_ids

        WITH node as v, score
        MATCH (v)-[:EXTRACTED_FROM]->(d:Document)
        RETURN DISTINCT d.id as doc_id, max(score) as score
        ORDER BY score DESC
        LIMIT $top_k
        """

        try:
            results = self.graph_repo.execute_query(
                query,
                {
                    "key_name": key_name,
                    "search_embedding": search_embedding,
                    "top_k": top_k
                }
            )

            doc_ids = [row["doc_id"] for row in results]
            logging.info(
                f"[Value Vector Search] Key '{key_name}' + Term '{search_term}' → {len(doc_ids)}개 Document (벡터 검색)")
            return doc_ids

        except Exception as e:
            logging.exception(f"[Value Vector Search] 검색 실패: {e}")
            return []
