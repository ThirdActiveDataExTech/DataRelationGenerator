from typing import Dict, Any, List
from functools import cached_property
import logging

from app.exceptions.base import ApplicationError
from app.repositories.graph_repository import GraphRepository
from app.src.embedding.graph_embedding_service import GraphEmbeddingService
from app.src.graphrag.kg_service import KnowledgeGraphService


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

    def check_graph_embedding_exists(self, doc_id: str) -> bool:
        """Document에 graph_embedding이 있는지 확인

        Args:
            doc_id: Document ID

        Returns:
            graph_embedding 존재 여부
        """
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
        """전체 Document 노드에 대해 그래프 임베딩 일괄 생성

        Args:
            embedding_dim: 임베딩 차원 (기본: 128)
            walk_length: Random walk 길이 (기본: 80)
            walks_per_node: 노드당 walk 횟수 (기본: 10)

        Returns:
            업데이트 결과
        """
        try:
            logging.info("전체 그래프 임베딩 생성 시작")

            # 1. 전체 Document 노드 조회
            query = "MATCH (d:Document) RETURN d.id as doc_id"
            docs = self.graph_repo.execute_query(query)
            doc_ids = [doc["doc_id"] for doc in docs]

            logging.info(f"총 {len(doc_ids)}개 Document 노드 발견")

            if not doc_ids:
                return {
                    "status": "no_documents",
                    "message": "업데이트할 Document가 없습니다"
                }

            # 2. 전체 그래프에 대해 Node2Vec 실행 (한 번만)
            logging.info("전체 그래프 Node2Vec 실행 중...")

            all_embeddings = self.graph_embedding_service.generate_all_graph_embeddings(
                embedding_dim=embedding_dim,
                walk_length=walk_length,
                walks_per_node=walks_per_node
            )

            if not all_embeddings:
                return {
                    "status": "failed",
                    "message": "그래프 임베딩 생성에 실패했습니다",
                    "total": len(doc_ids),
                    "success": 0,
                    "failed": len(doc_ids)
                }

            # 3. 각 Document 노드에 임베딩 업데이트
            success_count = 0
            failed_count = 0
            errors = []

            for doc_id in doc_ids:
                try:
                    if doc_id in all_embeddings:
                        # Document 노드에 graph_embedding 추가
                        update_query = """
                        MATCH (d:Document {id: $doc_id})
                        SET d.graph_embedding = $graph_embedding
                        """
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

            logging.info(f"그래프 임베딩 생성 완료: 성공 {success_count}개, 실패 {failed_count}개")

            return {
                "status": "completed",
                "total": len(doc_ids),
                "success": success_count,
                "failed": failed_count,
                "errors": errors[:10]  # 최대 10개만 반환
            }

        except Exception as e:
            logging.error(f"그래프 임베딩 일괄 생성 실패: {e}")
            import traceback
            logging.error(traceback.format_exc())
            raise ApplicationError(
                code=500,
                message="그래프 임베딩 생성에 실패했습니다.",
                result={"error": str(e)}
            )

    async def analyze_document_relations(
        self,
        doc_id: str = None,
        top_k: int = 10,
        text_weight: float = 0.6,
        graph_weight: float = 0.4
    ) -> Dict[str, Any]:
        """Document 간 관계 분석 및 생성

        Args:
            doc_id: 분석할 Document ID (None이면 전체)
            top_k: 관련 문서 최대 개수
            text_weight: 텍스트 유사도 가중치
            graph_weight: 그래프 임베딩 유사도 가중치

        Returns:
            관계 분석 결과

        Note:
            - graph_embedding이 생성된 후에 실행해야 함
            - 하이브리드 검색 (text + graph embedding + graph structure)
            - LLM으로 관계 추론
        """
        try:
            # 1. 대상 문서 조회
            if doc_id:
                # 특정 문서만
                query = "MATCH (d:Document {id: $doc_id}) RETURN d.id as doc_id, d.text as text"
                docs = self.graph_repo.execute_query(query, {"doc_id": doc_id})
            else:
                # 전체 문서
                query = "MATCH (d:Document) RETURN d.id as doc_id, d.text as text"
                docs = self.graph_repo.execute_query(query)

            if not docs:
                return {
                    "status": "no_documents",
                    "message": "분석할 Document가 없습니다"
                }

            logging.info(f"총 {len(docs)}개 Document 관계 분석 시작")

            total_relations_created = 0
            results = []

            for idx, doc in enumerate(docs, 1):
                current_doc_id = doc["doc_id"]
                current_doc_text = doc["text"]

                logging.info(f"[{idx}/{len(docs)}] {current_doc_id} 분석 중...")

                # 2. graph_embedding 존재 여부 확인
                has_graph_emb = self.check_graph_embedding_exists(current_doc_id)

                # 3. 하이브리드 검색으로 관련 문서 탐색
                related_docs = self.kg_service.find_related_documents_hybrid(
                    doc_id=current_doc_id,
                    doc_text=current_doc_text,
                    top_k=top_k,
                    text_weight=text_weight,
                    graph_weight=graph_weight if has_graph_emb else 0,
                    use_graph_structure=True
                )

                if not related_docs:
                    logging.info(f"{current_doc_id}: 관련 문서 없음")
                    results.append({
                        "doc_id": current_doc_id,
                        "relations_created": 0,
                        "related_docs_found": 0
                    })
                    continue

                # 4. 메타데이터 조회 (LLM 추론용)
                metadata_query = """
                MATCH (d:Document {id: $doc_id})
                OPTIONAL MATCH (d)-[:HAS_TOPIC]->(t:Topic)
                OPTIONAL MATCH (d)-[:HAS_TAG]->(tag:Tag)
                RETURN d.description as description,
                       collect(DISTINCT t.name) as topics,
                       collect(DISTINCT tag.name) as tags
                """
                metadata_result = self.graph_repo.execute_query(
                    metadata_query,
                    {"doc_id": current_doc_id}
                )

                current_metadata = {
                    "description": metadata_result[0]["description"] if metadata_result else "",
                    "topics": metadata_result[0]["topics"] if metadata_result else [],
                    "tags": metadata_result[0]["tags"] if metadata_result else []
                }

                # 5. LLM으로 Document 간 관계 추론
                relations = self._infer_document_relations(
                    new_doc_id=current_doc_id,
                    new_doc_text=current_doc_text,
                    new_doc_metadata=current_metadata,
                    related_docs=related_docs
                )

                # 6. 관계 생성
                created_count = 0
                if relations:
                    created_count = self.graph_repo.create_document_relations(
                        source_doc_id=current_doc_id,
                        relations=relations
                    )
                    total_relations_created += created_count

                results.append({
                    "doc_id": current_doc_id,
                    "relations_created": created_count,
                    "related_docs_found": len(related_docs),
                    "graph_embedding_used": has_graph_emb
                })

                logging.info(f"{current_doc_id}: {created_count}개 관계 생성")

            logging.info(f"관계 분석 완료: 총 {total_relations_created}개 관계 생성")

            return {
                "status": "completed",
                "total_documents": len(docs),
                "total_relations_created": total_relations_created,
                "results": results[:10]  # 최대 10개만 반환
            }

        except Exception as e:
            logging.error(f"관계 분석 실패: {e}")
            import traceback
            logging.error(traceback.format_exc())
            raise ApplicationError(
                code=500,
                message="관계 분석에 실패했습니다.",
                result={"error": str(e)}
            )

    def _infer_document_relations(
        self,
        new_doc_id: str,
        new_doc_text: str,
        new_doc_metadata: Dict[str, Any],
        related_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """LLM으로 Document 간 관계 추론

        Args:
            new_doc_id: 새 문서 ID
            new_doc_text: 새 문서 텍스트
            new_doc_metadata: 새 문서 메타데이터 (topics, tags, description)
            related_docs: 관련 문서 목록

        Returns:
            추출된 관계 목록 [{"target_doc_id": "...", "relation_type": "...", ...}]
        """
        from app.src.meta_extract.llm_response import LLMResponse

        if not related_docs:
            logging.info(f"Document {new_doc_id}: 관련 문서 없음")
            return []

        # 관련 문서 정보 포맷팅
        related_docs_info = self._format_related_docs_for_prompt(related_docs)

        # LLM 프롬프트 변수 준비
        variables = {
            "new_doc_id": new_doc_id,
            "new_doc_description": new_doc_metadata.get("description", ""),
            "new_doc_topics": ", ".join(new_doc_metadata.get("topics", [])),
            "new_doc_text": new_doc_text[:1000],  # 텍스트는 처음 1000자만
            "related_docs_info": related_docs_info
        }

        # LLM 호출
        result = LLMResponse("document_relation_inference").generate_llm_response(
            variables=variables
        )

        if not isinstance(result, dict):
            logging.warning(f"관계 추론 결과가 dict가 아님: {type(result).__name__}")
            return []

        relations = result.get("relations", [])
        logging.info(f"Document {new_doc_id}: 관계 {len(relations)}개 추출")

        return relations

    @staticmethod
    def _format_related_docs_for_prompt(related_docs: List[Dict[str, Any]]) -> str:
        """관련 문서를 LLM 프롬프트용 텍스트로 포맷팅"""
        formatted = []

        for idx, doc in enumerate(related_docs, 1):
            doc_info = f"""[Document {idx}]
ID: {doc.get('doc_id', 'unknown')}
설명: {doc.get('description', 'N/A')}
공유 Topic: {', '.join(doc.get('shared_topics', []))}
공유 Tag: {', '.join(doc.get('shared_tags', []))}
Graph Score: {doc.get('graph_score', 0)}
내용: {doc.get('text', '')[:300]}...
"""
            formatted.append(doc_info)

        return "\n\n".join(formatted)


# 싱글톤 인스턴스
graph_service = GraphService()
