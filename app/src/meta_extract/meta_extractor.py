import json
from functools import cached_property
from typing import Dict, Any, List

import logging

from app.exceptions.base import ApplicationError
from app.repositories.graph_repository import GraphRepository
from app.src.embedding.embedding_service import embedding_service
from app.src.meta_extract.graph_builder import GraphBuilder
from app.src.meta_extract.llm_response import LLMResponse


class MetadataService:
    """메타데이터 추출 서비스 - 메타데이터 추출 및 문서 생성만 담당"""

    @cached_property
    def graph_repo(self) -> GraphRepository:
        """Lazy initialization: GraphRepository"""
        return GraphRepository()

    @cached_property
    def graph_builder(self) -> GraphBuilder:
        """Lazy initialization: GraphBuilder"""
        return GraphBuilder(graph_repo=self.graph_repo)

    @staticmethod
    def _extract_metadata(data_text: str) -> Dict[str, Any]:
        """메타데이터 추출"""
        metadata = LLMResponse("meta_extract").generate_llm_response(
            variables={"data_text": data_text}
        )
        if not isinstance(metadata, dict):
            raise ValueError(f"메타데이터 추출 결과가 dict가 아님: {type(metadata).__name__}")

        logging.info(f"메타데이터 추출 완료: topics={len(metadata.get('topics', []))}, tags={len(metadata.get('tags', []))}")
        return metadata

    def _generate_categories(self, topics: List[str]) -> List[Dict[str, Any]]:
        """기존 카테고리 조회 후 카테고리 생성"""
        existing_cats = self.graph_repo.execute_query("MATCH (c:Category) RETURN c.name as category")
        cat_names = [item["category"] for item in existing_cats if "category" in item]

        category_result = LLMResponse("category_generate").generate_llm_response(
            variables={"topics": topics, "existing_categories": cat_names}
        )
        if not isinstance(category_result, dict):
            raise ValueError(f"카테고리 생성 결과가 dict가 아님: {type(category_result).__name__}")

        categories = category_result.get("categories", [])
        logging.info(f"카테고리 생성: {len(categories)}개")
        return categories

    async def extract_and_load_to_graph(self, data_text: str) -> Dict[str, Any]:
        """메타데이터 추출 후 Graph DB 적재"""
        try:
            # 1. 메타데이터 추출 및 카테고리 생성
            metadata = self._extract_metadata(data_text)
            categories = self._generate_categories(metadata.get("topics", []))
            # 2. 텍스트 임베딩 생성
            logging.info("텍스트 임베딩 생성 시작")
            text_embedding = embedding_service.generate_embedding(data_text)
            logging.info("텍스트 임베딩 생성 완료")

            # 3. Vector index 생성 (최초 1회만 실행됨)
            logging.info("Vector index 생성 시작")
            self.graph_repo.create_vector_index()
            logging.info("Text embedding index 생성 완료")

            self.graph_repo.create_graph_embedding_index()
            logging.info("Graph embedding index 생성 완료")

            # 4. 고정 구조 그래프 생성 (Document, Topic, Tag, Category + 고정 관계)
            logging.info("그래프 생성 시작")
            build_result = self.graph_builder.build_graph(
                metadata=metadata,
                text=data_text,
                categories=categories,
                text_embedding=text_embedding
            )

            doc_id = build_result["document_id"]
            logging.info(f"Document 생성 완료: {doc_id}")
            return {
                "metadata": metadata,
                "categories": categories,
                "document_id": doc_id,
                "status": "success"
            }
        except ValueError as e:
            raise ApplicationError(
                code=400,
                message="메타데이터 또는 카테고리 생성에 실패했습니다.",
                result={"error": str(e)}
            )

    async def batch_extract_and_load_from_jsonl(self, content: str) -> Dict[str, Any]:
        """JSONL 콘텐츠를 배치 처리하여 메타데이터 추출 및 Graph DB 적재"""
        lines = [line for line in content.strip().split("\n") if line.strip()]
        success_count = 0
        errors = []

        logging.info(f"배치 처리 시작: 총 {len(lines)}개 라인")

        for idx, line in enumerate(lines, start=1):
            try:
                logging.info(f"[{idx}/{len(lines)}] 처리 중...")
                json.loads(line)
                await self.extract_and_load_to_graph(line)
                success_count += 1
                logging.info(f"[{idx}/{len(lines)}] 처리 완료")
            except json.JSONDecodeError as e:
                logging.error(f"[{idx}/{len(lines)}] JSON 파싱 실패: {e}")
                errors.append({"line": idx, "error": f"JSON 파싱 실패: {str(e)}"})
            except Exception as e:
                logging.error(f"[{idx}/{len(lines)}] 처리 실패: {e}")
                import traceback
                logging.error(traceback.format_exc())
                errors.append({"line": idx, "error": str(e)})

        logging.info(f"배치 처리 완료: 성공 {success_count}개, 실패 {len(errors)}개")
        return {"success": success_count, "failed": len(errors), "errors": errors}


meta_service = MetadataService()
