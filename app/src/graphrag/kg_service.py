import ast
from typing import Dict, Any, List

import logging
from neo4j.exceptions import ServiceUnavailable, ClientError
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever

from app.config import settings
from app.repositories.graph_repository import GraphRepository
from app.repositories.graph_queries import (
    GET_DOCUMENTS_BY_IDS,
    SEARCH_BY_GRAPH_EMBEDDING
)


class KnowledgeGraphService:
    """neo4j-graphrag-python 기반 Knowledge Graph 서비스"""

    def __init__(self, graph_repo: GraphRepository):
        """KnowledgeGraphService 초기화"""
        self.graph_repo = graph_repo
        self.driver = graph_repo.driver
        self.embedder = SentenceTransformerEmbeddings(model=settings.EMBEDDING_MODEL_NAME)

    def find_related_documents_hybrid(
            self,
            doc_id: str,
            doc_text: str,
            top_k: int = 10,
            text_weight: float = 0.6,
            graph_weight: float = 0.4
    ) -> List[Dict[str, Any]]:
        """하이브리드 검색: 텍스트 유사도 + 그래프 임베딩 유사도"""
        try:
            text_results = self._search_by_text_embedding(doc_text)
            graph_embedding_scores = self._search_by_graph_embedding(doc_id, top_k=50)

            combined_results = self._combine_search_scores(
                doc_id, text_results, graph_embedding_scores, text_weight, graph_weight
            )

            sorted_results = sorted(
                combined_results.values(),
                key=lambda x: x["combined_score"],
                reverse=True
            )[:top_k]

            sorted_results = KnowledgeGraphService._deduplicate_results(sorted_results)
            related_docs = self._enrich_with_document_info(sorted_results)
            related_docs = [doc for doc in related_docs if doc["doc_id"] != doc_id]

            return related_docs

        except (ServiceUnavailable, ClientError):
            logging.exception(f"하이브리드 검색 실패 (doc_id: {doc_id}) - DB 오류")
            return []
        except (ValueError, KeyError, AttributeError):
            logging.exception(f"하이브리드 검색 실패 (doc_id: {doc_id}) - 데이터 처리 오류")
            return []

    @staticmethod
    def _deduplicate_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """결과에서 중복 doc_id 제거"""
        seen = set()
        unique_results = []

        for result in results:
            doc_id = result["doc_id"]
            if doc_id not in seen:
                seen.add(doc_id)
                unique_results.append(result)

        return unique_results

    def _search_by_text_embedding(self, doc_text: str):
        """텍스트 임베딩 기반 검색"""
        text_retriever = VectorRetriever(
            driver=self.driver,
            index_name="text_embedding_index",
            embedder=self.embedder
        )
        return text_retriever.search(query_text=doc_text, top_k=50)

    def _combine_search_scores(
            self,
            doc_id: str,
            text_results,
            graph_embedding_scores: Dict[str, float],
            text_weight: float,
            graph_weight: float
    ) -> Dict[str, Dict[str, Any]]:
        """검색 점수 결합"""
        combined_results = {}

        for item in text_results.items:
            candidate_id = self._extract_doc_id(item.content)
            if not candidate_id or candidate_id == doc_id:
                continue

            text_score = item.metadata.get("score", 0) if hasattr(item, 'metadata') and item.metadata else 0

            combined_results[candidate_id] = {
                "doc_id": candidate_id,
                "text_score": text_score,
                "graph_embedding_score": 0,
                "combined_score": text_weight * text_score
            }

        for candidate_id, graph_emb_score in graph_embedding_scores.items():
            if candidate_id == doc_id:
                continue

            if candidate_id in combined_results:
                combined_results[candidate_id]["graph_embedding_score"] = graph_emb_score
                combined_results[candidate_id]["combined_score"] += graph_weight * graph_emb_score
            else:
                combined_results[candidate_id] = {
                    "doc_id": candidate_id,
                    "text_score": 0,
                    "graph_embedding_score": graph_emb_score,
                    "combined_score": graph_weight * graph_emb_score
                }

        return combined_results

    def _enrich_with_document_info(self, sorted_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """문서 상세 정보 추가"""
        if not sorted_results:
            return []

        doc_ids = [result["doc_id"] for result in sorted_results]
        doc_infos = self.graph_repo.execute_query(GET_DOCUMENTS_BY_IDS, {'doc_ids': doc_ids})
        doc_info_map = {doc["doc_id"]: doc for doc in doc_infos}

        return [
            {
                "doc_id": result["doc_id"],
                "text": doc_info_map.get(result["doc_id"], {}).get("text", ""),
                "description": doc_info_map.get(result["doc_id"], {}).get("description", ""),
                "combined_score": result.get("combined_score", 0),
                "text_score": result.get("text_score", 0),
                "graph_embedding_score": result.get("graph_embedding_score", 0)
            }
            for result in sorted_results
        ]

    @staticmethod
    def _extract_doc_id(content) -> str | None:
        """content에서 doc_id 추출 (Neo4j에서 가져온 검증된 데이터)"""
        if isinstance(content, dict):
            return content.get("id")

        if isinstance(content, str):
            try:
                content_dict = ast.literal_eval(content)
                return content_dict.get("id") if isinstance(content_dict, dict) else None
            except (ValueError, SyntaxError):
                import re
                match = re.search(r"'id':\s*'([^']+)'", content)
                return match.group(1) if match else None

        return None

    def _search_by_graph_embedding(self, doc_id: str, top_k: int = 50) -> Dict[str, float]:
        """그래프 임베딩 기반 유사 문서 검색"""
        try:
            results = self.graph_repo.execute_query(
                SEARCH_BY_GRAPH_EMBEDDING,
                {"doc_id": doc_id, "top_k": top_k}
            )

            return {row["doc_id"]: row["score"] for row in results}

        except (ServiceUnavailable, ClientError) as e:
            logging.warning(f"그래프 임베딩 검색 실패 (doc_id: {doc_id}) - DB 오류: {e}")
            return {}
        except (ValueError, KeyError) as e:
            logging.warning(f"그래프 임베딩 검색 실패 (doc_id: {doc_id}) - 데이터 오류: {e}")
            return {}
