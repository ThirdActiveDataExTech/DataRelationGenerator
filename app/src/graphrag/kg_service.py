import ast
from typing import Dict, Any, List

import logging
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever

from app.config import settings
from app.repositories.graph_repository import GraphRepository


class KnowledgeGraphService:
    """neo4j-graphrag-python 기반 Knowledge Graph 서비스"""

    def __init__(self, graph_repo: GraphRepository):
        self.graph_repo = graph_repo
        self.driver = graph_repo.driver

        self.embedder = SentenceTransformerEmbeddings(
            model=settings.EMBEDDING_MODEL_NAME
        )
        logging.info("KnowledgeGraphService 초기화 완료")

    def find_related_documents_hybrid(
            self,
            doc_id: str,
            doc_text: str,
            top_k: int = 10,
            text_weight: float = 0.6,
            graph_weight: float = 0.4,
            use_graph_structure: bool = True
    ) -> List[Dict[str, Any]]:
        """하이브리드 검색: 텍스트 유사도 + 그래프 임베딩 유사도 + 그래프 연결 정보"""
        try:
            text_results = self._search_by_text_embedding(doc_text)
            graph_embedding_scores = self._search_by_graph_embedding(doc_id, top_k=50)

            logging.info(f"텍스트 검색: {len(text_results.items)}개, 그래프 검색: {len(graph_embedding_scores)}개")

            combined_results = self._combine_search_scores(
                doc_id, text_results, graph_embedding_scores, text_weight, graph_weight
            )

            if use_graph_structure:
                self._add_graph_structure_info(doc_id, combined_results)

            sorted_results = sorted(
                combined_results.values(),
                key=lambda x: x["combined_score"],
                reverse=True
            )[:top_k]

            related_docs = self._enrich_with_document_info(sorted_results)

            logging.info(f"하이브리드 검색 완료: {len(related_docs)}개 (doc_id: {doc_id})")

            return related_docs

        except Exception as e:
            logging.error(f"하이브리드 검색 실패 (doc_id: {doc_id}): {e}")
            return []

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
                "graph_score": 0,
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
                    "graph_score": 0,
                    "combined_score": graph_weight * graph_emb_score
                }

        return combined_results

    def _add_graph_structure_info(self, doc_id: str, combined_results: Dict[str, Dict[str, Any]]):
        """그래프 구조 정보 추가"""
        for candidate_id in list(combined_results.keys()):
            graph_data = self._get_graph_connections(doc_id, candidate_id)
            combined_results[candidate_id]["shared_topics"] = graph_data["shared_topics"]
            combined_results[candidate_id]["shared_tags"] = graph_data["shared_tags"]
            combined_results[candidate_id]["graph_score"] = graph_data["graph_score"]

    def _enrich_with_document_info(self, sorted_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """문서 상세 정보 추가"""
        related_docs = []
        for result in sorted_results:
            doc_info = self.graph_repo.execute_query(
                'MATCH (d:Document {id: $doc_id}) RETURN d.text as text, d.description as description',
                {'doc_id': result["doc_id"]}
            )

            related_docs.append({
                **result,
                "text": doc_info[0]["text"] if doc_info else "",
                "description": doc_info[0]["description"] if doc_info else ""
            })

        return related_docs

    @staticmethod
    def _extract_doc_id(content) -> str | None:
        """content에서 doc_id 추출"""
        if isinstance(content, str):
            try:
                content_dict = ast.literal_eval(content)
                return content_dict.get("id")
            except Exception:
                return content
        elif isinstance(content, dict):
            return content.get("id")
        return None

    def _search_by_graph_embedding(self, doc_id: str, top_k: int = 50) -> Dict[str, float]:
        """그래프 임베딩 기반 유사 문서 검색"""
        try:
            query = """
            MATCH (source:Document {id: $doc_id})
            WHERE source.graph_embedding IS NOT NULL
            CALL db.index.vector.queryNodes('graph_embedding_index', $top_k, source.graph_embedding)
            YIELD node, score
            WHERE node.id <> $doc_id
            RETURN node.id as doc_id, score
            """

            results = self.graph_repo.execute_query(query, {"doc_id": doc_id, "top_k": top_k})

            return {row["doc_id"]: row["score"] for row in results}

        except Exception as e:
            logging.warning(f"그래프 임베딩 검색 실패 (doc_id: {doc_id}): {e}")
            return {}

    def _get_graph_connections(self, source_doc_id: str, target_doc_id: str) -> Dict[str, Any]:
        """두 문서 간의 그래프 연결 정보 조회"""
        query = """
        MATCH (source:Document {id: $source_id})
        MATCH (target:Document {id: $target_id})

        // 공유 Topic
        OPTIONAL MATCH (source)-[:HAS_TOPIC]->(t:Topic)<-[:HAS_TOPIC]-(target)
        WITH source, target, collect(DISTINCT t.name) as shared_topics

        // 공유 Tag
        OPTIONAL MATCH (source)-[:HAS_TAG]->(tag:Tag)<-[:HAS_TAG]-(target)
        WITH source, target, shared_topics, collect(DISTINCT tag.name) as shared_tags

        // 공유 Category (2-hop)
        OPTIONAL MATCH (source)-[:HAS_TOPIC]->(:Topic)-[:BELONGS_TO]->(c:Category)
                      <-[:BELONGS_TO]-(:Topic)<-[:HAS_TOPIC]-(target)
        WITH shared_topics, shared_tags, count(DISTINCT c) as shared_categories

        // Graph Score 계산 (Topic: 3점, Tag: 1점, Category: 2점)
        RETURN shared_topics,
               shared_tags,
               size(shared_topics) * 3 + size(shared_tags) * 1 + shared_categories * 2 as graph_score
        """

        result = self.graph_repo.execute_query(
            query,
            {'source_id': source_doc_id, 'target_id': target_doc_id}
        )

        if result:
            return {
                "shared_topics": result[0].get("shared_topics", []),
                "shared_tags": result[0].get("shared_tags", []),
                "graph_score": result[0].get("graph_score", 0)
            }

        return {"shared_topics": [], "shared_tags": [], "graph_score": 0}
