from functools import cached_property
from typing import Dict, Any, List

import logging

from neo4j.exceptions import ServiceUnavailable, ClientError

from app.exceptions.base import ApplicationError
from app.repositories.graph_repository import GraphRepository
from app.src.embedding.graph_embedding_service import GraphEmbeddingService
from app.src.graphrag.graphrag_service import GraphRAGService
from app.src.graphrag.kg_service import KnowledgeGraphService
from app.src.meta_extract.llm_response import LLMResponse


class GraphService:
    """Graph RAG 서비스 - 그래프 임베딩 및 관계 분석"""

    @cached_property
    def graph_repo(self) -> GraphRepository:
        """Lazy initialization: GraphRepository"""
        return GraphRepository()

    @cached_property
    def kg_service(self) -> KnowledgeGraphService:
        """Lazy initialization: KG 서비스"""
        return KnowledgeGraphService(graph_repo=self.graph_repo)

    @cached_property
    def graph_embedding_service(self) -> GraphEmbeddingService:
        """Lazy initialization: Graph Embedding 서비스 (HashGNN)"""
        return GraphEmbeddingService(graph_repo=self.graph_repo)

    @cached_property
    def graphrag_service(self) -> GraphRAGService:
        """Lazy initialization: GraphRAG 서비스"""
        return GraphRAGService(graph_repo=self.graph_repo)

    def check_graph_embedding_exists(self, doc_id: str) -> bool:
        """Document에 graph_embedding 존재 여부 확인"""
        query = "MATCH (d:Document {id: $doc_id}) RETURN d.graph_embedding IS NOT NULL as has_embedding"
        result = self.graph_repo.execute_query(query, {"doc_id": doc_id})
        return result[0]["has_embedding"] if result else False

    async def update_all_graph_embeddings(
            self,
            embedding_dim: int = 128
    ) -> Dict[str, Any]:
        """전체 Document 노드에 대해 HashGNN으로 그래프 임베딩 일괄 생성"""
        try:
            doc_ids = self._get_all_document_ids()
            if not doc_ids:
                return {"status": "no_documents", "message": "업데이트할 Document가 없습니다"}

            all_embeddings = self._generate_embeddings(embedding_dim)
            if not all_embeddings:
                return {
                    "status": "failed",
                    "message": "그래프 임베딩 생성에 실패했습니다",
                    "total": len(doc_ids),
                    "success": 0,
                    "failed": len(doc_ids)
                }

            result = self._update_document_embeddings(doc_ids, all_embeddings)
            logging.info(f"[HashGNN] 그래프 임베딩 완료: 성공 {result['success']}/{len(doc_ids)}")

            return {
                "status": "completed",
                "total": result["total"],
                "success": result["success"],
                "failed": result["failed"],
                "errors": result["errors"][:10]
            }

        except (ServiceUnavailable, ClientError) as e:
            logging.exception(f"[HashGNN] 그래프 임베딩 일괄 생성 실패 (DB 오류)")
            raise ApplicationError(
                code=500,
                message="그래프 임베딩 생성에 실패했습니다 (DB 연결 문제)",
                result={"error": str(e)}
            )
        except (ValueError, KeyError, AttributeError, RuntimeError) as e:
            logging.exception(f"[HashGNN] 그래프 임베딩 일괄 생성 실패 (데이터 처리 오류)")
            raise ApplicationError(
                code=500,
                message="그래프 임베딩 생성에 실패했습니다.",
                result={"error": str(e)}
            )

    def _get_all_document_ids(self) -> List[str]:
        """전체 Document ID 조회"""
        query = "MATCH (d:Document) RETURN d.id as doc_id"
        docs = self.graph_repo.execute_query(query)
        return [doc["doc_id"] for doc in docs]

    def _generate_embeddings(self, embedding_dim: int) -> Dict[str, List[float]]:
        """HashGNN으로 그래프 임베딩 생성"""
        return self.graph_embedding_service.generate_all_graph_embeddings(
            embedding_dim=embedding_dim
        )

    def _update_document_embeddings(
            self,
            doc_ids: List[str],
            all_embeddings: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """Document 노드에 임베딩 업데이트"""
        success_count = 0
        failed_count = 0
        errors = []

        update_query = """
        MATCH (d:Document {id: $doc_id})
        SET d.graph_embedding = $graph_embedding
        """

        for doc_id in doc_ids:
            try:
                if doc_id in all_embeddings:
                    self.graph_repo.execute_query(
                        update_query,
                        {"doc_id": doc_id, "graph_embedding": all_embeddings[doc_id]}
                    )
                    success_count += 1
                else:
                    failed_count += 1
                    errors.append({"doc_id": doc_id, "error": "임베딩 생성되지 않음"})
            except (ServiceUnavailable, ClientError, ValueError, KeyError) as e:
                failed_count += 1
                errors.append({"doc_id": doc_id, "error": str(e)})
                logging.exception(f"Document {doc_id} 임베딩 업데이트 실패")

        return {
            "total": len(doc_ids),
            "success": success_count,
            "failed": failed_count,
            "errors": errors
        }

    async def analyze_document_relations(
            self,
            doc_id: str = None,
            top_k: int = 4,
            text_weight: float = 0.4,
            graph_weight: float = 0.6
    ) -> Dict[str, Any]:
        """Document 간 관계 분석 및 생성"""
        try:
            docs = self._get_target_documents(doc_id)
            if not docs:
                return {"status": "no_documents", "message": "분석할 Document가 없습니다"}

            total_relations_created = 0
            results = []

            for doc in docs:
                current_doc_id = doc["doc_id"]
                current_doc_text = doc["text"]

                doc_result = self._analyze_single_document(
                    current_doc_id, current_doc_text, top_k, text_weight, graph_weight
                )

                total_relations_created += doc_result["relations_created"]
                results.append(doc_result)

            logging.info(f"관계 분석 완료: {len(docs)}개 문서, 총 {total_relations_created}개 관계 생성")

            return {
                "status": "completed",
                "total_documents": len(docs),
                "total_relations_created": total_relations_created,
                "results": results[:10]
            }

        except (ServiceUnavailable, ClientError) as e:
            logging.exception(f"관계 분석 실패 (DB 오류)")
            raise ApplicationError(
                code=500,
                message="관계 분석에 실패했습니다 (DB 연결 문제)",
                result={"error": str(e)}
            )
        except (ValueError, KeyError, AttributeError, RuntimeError) as e:
            logging.exception(f"관계 분석 실패 (데이터 처리 오류)")
            raise ApplicationError(
                code=500,
                message="관계 분석에 실패했습니다.",
                result={"error": str(e)}
            )

    def _get_target_documents(self, doc_id: str = None) -> List[Dict[str, str]]:
        """분석 대상 문서 조회"""
        if doc_id:
            query = "MATCH (d:Document {id: $doc_id}) RETURN d.id as doc_id, d.text as text"
            return self.graph_repo.execute_query(query, {"doc_id": doc_id})

        query = """
        MATCH (d:Document)
        WHERE NOT EXISTS {
            MATCH (d)-[r:SIMILAR_TO|CONTRADICTS|EXTENDS|PREREQUISITE_OF|IMPLEMENTS|CO_OCCURS]->(:Document)
        }
        RETURN d.id as doc_id, d.text as text
        """
        return self.graph_repo.execute_query(query)

    def _analyze_single_document(
            self,
            doc_id: str,
            doc_text: str,
            top_k: int,
            text_weight: float,
            graph_weight: float
    ) -> Dict[str, Any]:
        """단일 문서에 대한 관계 분석"""
        has_graph_emb = self.check_graph_embedding_exists(doc_id)

        related_docs = self.kg_service.find_related_documents_hybrid(
            doc_id=doc_id,
            doc_text=doc_text,
            top_k=top_k,
            text_weight=text_weight,
            graph_weight=graph_weight if has_graph_emb else 0,
            use_graph_structure=True
        )

        if not related_docs:
            return {
                "doc_id": doc_id,
                "relations_created": 0,
                "related_docs_found": 0
            }

        metadata = self._get_document_metadata(doc_id)
        relations = self._infer_document_relations(doc_id, doc_text, metadata, related_docs)

        created_count = 0
        if relations:
            created_count = self.graph_repo.create_document_relations(doc_id, relations)

        return {
            "doc_id": doc_id,
            "relations_created": created_count,
            "related_docs_found": len(related_docs),
            "graph_embedding_used": has_graph_emb
        }

    def _get_document_metadata(self, doc_id: str) -> Dict[str, Any]:
        """Document 메타데이터 조회"""
        query = """
        MATCH (d:Document {id: $doc_id})
        OPTIONAL MATCH (d)-[:HAS_TOPIC]->(t:Topic)
        OPTIONAL MATCH (d)-[:MENTIONS]->(e:Entity)
        RETURN d.description as description,
               collect(DISTINCT t.name) as topics,
               collect(DISTINCT e.name) as entities
        """
        result = self.graph_repo.execute_query(query, {"doc_id": doc_id})

        if result:
            return {
                "description": result[0]["description"],
                "topics": result[0]["topics"],
                "entities": result[0]["entities"]
            }
        return {"description": "", "topics": [], "entities": []}

    def _infer_document_relations(
            self,
            doc_id: str,
            doc_text: str,
            metadata: Dict[str, Any],
            related_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """LLM으로 Document 간 관계 추론"""
        if not related_docs:
            return []

        unique_related_docs = self._deduplicate_related_docs(related_docs)

        relations = []
        for related_doc in unique_related_docs:
            relation = self._infer_single_relation(doc_id, doc_text, metadata, related_doc)
            if relation:
                relations.append(relation)

        return relations

    @staticmethod
    def _deduplicate_related_docs(related_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """관련 문서 목록에서 중복 제거"""
        seen = set()
        unique_docs = []

        for doc in related_docs:
            doc_id = doc.get("doc_id")
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                unique_docs.append(doc)

        return unique_docs

    def _infer_single_relation(
            self,
            doc_id: str,
            doc_text: str,
            metadata: Dict[str, Any],
            related_doc: Dict[str, Any]
    ) -> Dict[str, Any] | None:
        """단일 문서 쌍에 대한 관계 추론"""
        related_doc_id = related_doc.get("doc_id", "unknown")
        if related_doc_id == doc_id:
            return None

        variables = self._build_relation_inference_variables(
            doc_id, doc_text, metadata, related_doc_id, related_doc
        )

        result = LLMResponse("document_relation_inference").generate_llm_response(
            variables=variables
        )

        if not isinstance(result, dict):
            logging.warning(f"관계 추론 결과가 dict가 아님: {type(result).__name__}")
            return None

        relations = result.get("relations", [])
        return relations[0] if relations else None

    @staticmethod
    def _build_relation_inference_variables(
            doc_id: str,
            doc_text: str,
            metadata: Dict[str, Any],
            related_doc_id: str,
            related_doc: Dict[str, Any]
    ) -> Dict[str, Any]:
        """관계 추론을 위한 LLM 입력 변수 구성"""
        return {
            "new_doc_id": doc_id,
            "new_doc_description": metadata.get("description", ""),
            "new_doc_entities": ", ".join(metadata.get("entities", [])),
            "new_doc_topics": ", ".join(metadata.get("topics", [])),
            "new_doc_text": doc_text[:1000],
            "related_doc_id": related_doc_id,
            "related_doc_description": related_doc.get("description", "N/A"),
            "shared_entities": ", ".join(related_doc.get("shared_entities", [])),
            "related_doc_topics": ", ".join(related_doc.get("shared_topics", [])),
            "graph_score": related_doc.get("graph_score", 0),
            "related_doc_text": related_doc.get("text", "")[:500]
        }


# 싱글톤 인스턴스
graph_service = GraphService()
