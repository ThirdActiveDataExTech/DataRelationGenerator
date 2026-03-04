from typing import List, Dict, Any

import logging

from app.repositories.graph_repository import GraphRepository
from app.src.embedding.embedding_service import EmbeddingService


class EntityResolutionService:
    """Entity Resolution 서비스 - 새 Entity와 기존 그래프 Entity 간 RELATED_TO 연결"""

    def __init__(self, graph_repo: GraphRepository = None):
        """Init 매서드 - GraphRepository 인스턴스 생성"""
        self.graph_repo = graph_repo or GraphRepository()
        self.embedding_service = EmbeddingService()

    def resolve_entity(
            self,
            new_entity: str,
            top_k: int = 5,
            similarity_threshold: float = 0.95
    ) -> List[Dict[str, Any]]:
        """새 Entity와 기존 그래프의 RELATED_TO 관계 생성

        Args:
            new_entity: 새로 추출된 Entity 이름
            top_k: 후보 탐색 개수
            similarity_threshold: 유사도 임계값

        Returns:
            관계 정보 리스트 [{"existing_entity": "판례", "score": 0.95}, ...]
        """
        candidates = self._find_candidate_entities(new_entity, top_k, similarity_threshold)

        if not candidates:
            logging.info(f"Entity '{new_entity}': 유사 후보 없음")
            return []

        logging.info(f"Entity '{new_entity}': {len(candidates)}개 RELATED_TO 관계 생성")
        return candidates

    def _find_candidate_entities(
            self,
            entity_name: str,
            top_k: int,
            threshold: float
    ) -> List[Dict[str, Any]]:
        """Full-Text + Vector로 후보 Entity 탐색

        Args:
            entity_name: Entity 이름
            top_k: 최대 후보 개수
            threshold: 유사도 임계값

        Returns:
            후보 Entity 목록 [{"name": "판례", "score": 0.95}, ...]
        """
        fulltext_results = self._search_by_fulltext(entity_name, top_k)
        vector_results = self._search_by_vector(entity_name, top_k)

        combined = {}
        for item in fulltext_results:
            name = item["name"]
            # 자기 자신 제외
            if name == entity_name:
                continue
            combined[name] = {"name": name, "score": item["score"]}

        for item in vector_results:
            name = item["name"]
            # 자기 자신 제외
            if name == entity_name:
                continue
            if name in combined:
                combined[name]["score"] = max(combined[name]["score"], item["score"])
            else:
                combined[name] = {"name": name, "score": item["score"]}

        candidates = [v for v in combined.values() if v["score"] >= threshold]
        candidates.sort(key=lambda x: x["score"], reverse=True)

        return candidates[:top_k]

    def _search_by_fulltext(self, entity_name: str, top_k: int) -> List[Dict[str, Any]]:
        """Full-Text Index로 Entity 검색 (Fuzzy 포함)"""
        import re

        # 특수문자 체크 (영문, 한글, 숫자 외)
        has_special = bool(re.search(r'[^\w\sㄱ-힣]', entity_name))

        # 특수문자 있으면 따옴표로 감싸기, 3글자 이상만 fuzzy
        if len(entity_name) >= 3:
            if has_special or ' ' in entity_name:
                # 내부 따옴표 이스케이프 후 감싸기
                escaped = entity_name.replace('"', '\\"')
                fuzzy_term = f'"{escaped}"~2'
            else:
                fuzzy_term = f"{entity_name}~2"
        else:
            # 짧은 단어는 정확 매칭만
            fuzzy_term = entity_name

        query = """
        CALL db.index.fulltext.queryNodes('entity_fulltext', $search_term)
        YIELD node, score
        RETURN node.name as name, score
        ORDER BY score DESC
        LIMIT $top_k
        """

        try:
            results = self.graph_repo.execute_query(
                query,
                {"search_term": fuzzy_term, "top_k": top_k}
            )
            return results
        except Exception as e:
            logging.warning(f"Full-Text 검색 실패: {e}")
            return []

    def _search_by_vector(self, entity_name: str, top_k: int) -> List[Dict[str, Any]]:
        """Vector Similarity로 Entity 검색"""
        entity_emb = self.embedding_service.generate_embedding(entity_name)
        binary_emb = self.embedding_service.binarize_embedding(entity_emb, method="threshold")

        query = """
        CALL db.index.vector.queryNodes('entity_embedding_index', $top_k, $embedding)
        YIELD node, score
        RETURN node.name as name, score
        """

        try:
            results = self.graph_repo.execute_query(
                query,
                {"top_k": top_k, "embedding": binary_emb}
            )
            return results
        except Exception as e:
            logging.warning(f"Vector 검색 실패: {e}")
            return []
