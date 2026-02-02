from functools import cached_property
from typing import Dict, Any, List

import logging

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
        """Lazy initialization: Graph Embedding 서비스"""
        return GraphEmbeddingService(graph_repo=self.graph_repo)

    @cached_property
    def graphrag_service(self) -> GraphRAGService:
        """Lazy initialization: GraphRAG 서비스"""
        return GraphRAGService(graph_repo=self.graph_repo)

    def check_graph_embedding_exists(self, doc_id: str) -> bool:
        """Document에 graph_embedding이 있는지 확인"""
        query = """
        MATCH (d:Document {id: $doc_id})
        RETURN d.graph_embedding IS NOT NULL as has_embedding
        """
        result = self.graph_repo.execute_query(query, {"doc_id": doc_id})
        return result[0]["has_embedding"] if result else False

    async def update_all_graph_embeddings(
            self,
            embedding_dim: int = 128,
            walk_length: int = 80,
            walks_per_node: int = 10
    ) -> Dict[str, Any]:
        """전체 Document 노드에 대해 그래프 임베딩 일괄 생성"""
        try:
            logging.info("전체 그래프 임베딩 생성 시작")

            doc_ids = self._get_all_document_ids()
            if not doc_ids:
                return {"status": "no_documents", "message": "업데이트할 Document가 없습니다"}

            all_embeddings = self._generate_embeddings(
                embedding_dim, walk_length, walks_per_node
            )
            if not all_embeddings:
                return {
                    "status": "failed",
                    "message": "그래프 임베딩 생성에 실패했습니다",
                    "total": len(doc_ids),
                    "success": 0,
                    "failed": len(doc_ids)
                }

            result = self._update_document_embeddings(doc_ids, all_embeddings)
            logging.info(f"그래프 임베딩 생성 완료: 성공 {result['success']}개, 실패 {result['failed']}개")

            return {
                "status": "completed",
                "total": result["total"],
                "success": result["success"],
                "failed": result["failed"],
                "errors": result["errors"][:10]
            }

        except Exception as e:
            logging.error(f"그래프 임베딩 일괄 생성 실패: {e}")
            raise ApplicationError(
                code=500,
                message="그래프 임베딩 생성에 실패했습니다.",
                result={"error": str(e)}
            )

    def _get_all_document_ids(self) -> List[str]:
        """전체 Document ID 조회"""
        query = "MATCH (d:Document) RETURN d.id as doc_id"
        docs = self.graph_repo.execute_query(query)
        doc_ids = [doc["doc_id"] for doc in docs]
        logging.info(f"총 {len(doc_ids)}개 Document 노드 발견")
        return doc_ids

    def _generate_embeddings(
            self,
            embedding_dim: int,
            walk_length: int,
            walks_per_node: int
    ) -> Dict[str, List[float]]:
        """Node2Vec으로 그래프 임베딩 생성"""
        logging.info("전체 그래프 Node2Vec 실행 중...")
        return self.graph_embedding_service.generate_all_graph_embeddings(
            embedding_dim=embedding_dim,
            walk_length=walk_length,
            walks_per_node=walks_per_node
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
                    if success_count % 10 == 0:
                        logging.info(f"[{success_count}/{len(doc_ids)}] 임베딩 업데이트 중...")
                else:
                    failed_count += 1
                    errors.append({"doc_id": doc_id, "error": "임베딩 생성되지 않음"})
            except Exception as e:
                failed_count += 1
                errors.append({"doc_id": doc_id, "error": str(e)})
                logging.error(f"Document {doc_id} 임베딩 업데이트 실패: {e}")

        return {
            "total": len(doc_ids),
            "success": success_count,
            "failed": failed_count,
            "errors": errors
        }

    async def analyze_document_relations(
            self,
            doc_id: str = None,
            top_k: int = 10,
            text_weight: float = 0.6,
            graph_weight: float = 0.4
    ) -> Dict[str, Any]:
        """Document 간 관계 분석 및 생성"""
        try:
            docs = self._get_target_documents(doc_id)
            if not docs:
                return {"status": "no_documents", "message": "분석할 Document가 없습니다"}

            logging.info(f"총 {len(docs)}개 Document 관계 분석 시작")

            total_relations_created = 0
            results = []

            for idx, doc in enumerate(docs, 1):
                current_doc_id = doc["doc_id"]
                current_doc_text = doc["text"]

                logging.info(f"[{idx}/{len(docs)}] {current_doc_id} 분석 중...")

                doc_result = self._analyze_single_document(
                    current_doc_id, current_doc_text, top_k, text_weight, graph_weight
                )

                total_relations_created += doc_result["relations_created"]
                results.append(doc_result)

                logging.info(f"{current_doc_id}: {doc_result['relations_created']}개 관계 생성")

            logging.info(f"관계 분석 완료: 총 {total_relations_created}개 관계 생성")

            return {
                "status": "completed",
                "total_documents": len(docs),
                "total_relations_created": total_relations_created,
                "results": results[:10]
            }

        except Exception as e:
            logging.error(f"관계 분석 실패: {e}")
            raise ApplicationError(
                code=500,
                message="관계 분석에 실패했습니다.",
                result={"error": str(e)}
            )

    def _get_target_documents(self, doc_id: str = None) -> List[Dict[str, str]]:
        """분석 대상 문서 조회 (이미 관계가 있는 문서 제외)"""
        if doc_id:
            # 특정 문서 지정 시: 해당 문서만 (이미 관계가 있어도 재분석)
            query = "MATCH (d:Document {id: $doc_id}) RETURN d.id as doc_id, d.text as text"
            return self.graph_repo.execute_query(query, {"doc_id": doc_id})
        else:
            # 전체 문서 분석 시: 아직 outgoing 관계가 없는 문서만
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
            logging.info(f"{doc_id}: 관련 문서 없음")
            return {
                "doc_id": doc_id,
                "relations_created": 0,
                "related_docs_found": 0
            }

        doc_ids_debug = []
        for d in related_docs:
            if isinstance(d, dict) and 'doc_id' in d:
                doc_ids_debug.append(d['doc_id'])
            else:
                doc_ids_debug.append(f"INVALID:{type(d).__name__}")
        logging.info(f"{doc_id}: 검색된 관련 문서 {len(related_docs)}개 - {doc_ids_debug}")

        metadata = self._get_document_metadata(doc_id)
        relations = self._infer_document_relations(doc_id, doc_text, metadata, related_docs)

        logging.info(f"{doc_id}: LLM이 추론한 관계 {len(relations)}개")
        for rel in relations:
            logging.info(f"  -> {rel.get('target_doc_id')}: {rel.get('relation_type')}")

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
        OPTIONAL MATCH (d)-[:HAS_TAG]->(tag:Tag)
        RETURN d.description as description,
               collect(DISTINCT t.name) as topics,
               collect(DISTINCT tag.name) as tags
        """
        result = self.graph_repo.execute_query(query, {"doc_id": doc_id})

        if result:
            return {
                "description": result[0]["description"],
                "topics": result[0]["topics"],
                "tags": result[0]["tags"]
            }
        return {"description": "", "topics": [], "tags": []}

    def _infer_document_relations(
            self,
            doc_id: str,
            doc_text: str,
            metadata: Dict[str, Any],
            related_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """LLM으로 Document 간 관계 추론 (각 문서 쌍 1:1로 처리)"""
        if not related_docs:
            logging.info(f"Document {doc_id}: 관련 문서 없음")
            return []

        # 중복 제거: doc_id 기준으로
        seen_doc_ids = set()
        unique_related_docs = []
        for doc in related_docs:
            related_doc_id = doc.get("doc_id")
            if related_doc_id and related_doc_id not in seen_doc_ids:
                seen_doc_ids.add(related_doc_id)
                unique_related_docs.append(doc)
            else:
                logging.warning(f"중복 관련 문서 제거: {related_doc_id}")

        logging.info(f"Document {doc_id}: 중복 제거 후 {len(unique_related_docs)}개 문서 처리")

        all_relations = []

        for related_doc in unique_related_docs:
            # 각 관련 문서와 1:1 관계 추론
            relation = self._infer_single_relation(
                doc_id=doc_id,
                doc_text=doc_text,
                metadata=metadata,
                related_doc=related_doc
            )

            if relation:
                all_relations.append(relation)

        logging.info(f"Document {doc_id}: 총 {len(all_relations)}개 관계 추출")
        return all_relations

    def _infer_single_relation(
            self,
            doc_id: str,
            doc_text: str,
            metadata: Dict[str, Any],
            related_doc: Dict[str, Any]
    ) -> Dict[str, Any] | None:
        """단일 문서 쌍에 대한 관계 추론"""
        # 관련 문서 ID 확인
        related_doc_id = related_doc.get("doc_id", "unknown")

        # 자기 자신과의 비교 방지 (최종 안전장치)
        if related_doc_id == doc_id:
            logging.warning(f"자기 자신과의 관계 추론 시도 방지: {doc_id}")
            return None

        variables = {
            "new_doc_id": doc_id,
            "new_doc_description": metadata.get("description", ""),
            "new_doc_topics": ", ".join(metadata.get("topics", [])),
            "new_doc_text": doc_text[:1000],
            "related_doc_id": related_doc_id,
            "related_doc_description": related_doc.get("description", "N/A"),
            "related_doc_topics": ", ".join(related_doc.get("shared_topics", [])),
            "related_doc_tags": ", ".join(related_doc.get("shared_tags", [])),
            "graph_score": related_doc.get("graph_score", 0),
            "related_doc_text": related_doc.get("text", "")[:500]
        }

        # 디버그: 프롬프트 변수 확인
        logging.debug(
            f"관계 추론 변수: {doc_id} -> {related_doc_id} | "
            f"Topics: {variables['related_doc_topics'][:50] if variables['related_doc_topics'] else '없음'} | "
            f"Tags: {variables['related_doc_tags'][:50] if variables['related_doc_tags'] else '없음'} | "
            f"Graph Score: {variables['graph_score']}"
        )

        result = LLMResponse("document_relation_inference").generate_llm_response(
            variables=variables
        )

        if not isinstance(result, dict):
            logging.warning(f"관계 추론 결과가 dict가 아님: {type(result).__name__}")
            return None

        relations = result.get("relations", [])

        # 단일 문서 쌍이므로 관계가 있으면 첫 번째 것만 반환
        if relations and len(relations) > 0:
            return relations[0]

        return None


# TODO: 이 내용 어떻게 전달할지 더 고민해보기

# 싱글톤 인스턴스
graph_service = GraphService()
