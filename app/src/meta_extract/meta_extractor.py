import json
from functools import cached_property
from typing import Dict, Any, List

import logging

from app.exceptions.base import ApplicationError
from app.repositories.graph_repository import GraphRepository
from app.src.meta_extract.graph_builder import GraphBuilder
from app.src.meta_extract.llm_response import LLMResponse


class MetadataService:
    """메타데이터 추출 서비스 클래스"""

    @cached_property
    def graph_repo(self) -> GraphRepository:
        """Lazy initialization: 최초 접근 시 한 번만 생성"""
        return GraphRepository()

    @cached_property
    def graph_builder(self) -> GraphBuilder:
        """Lazy initialization: 최초 접근 시 한 번만 생성"""
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

    def extract_and_load_to_graph(self, data_text: str) -> Dict[str, Any]:
        """메타데이터 추출 후 Graph DB 적재 - Embedding & 카테고리 포함"""
        try:
            metadata = self._extract_metadata(data_text)
            categories = self._generate_categories(metadata.get("topics", []))
            build_result = self.graph_builder.build_graph(
                metadata=metadata,
                text=data_text,
                categories=categories
            )
            return {
                "metadata": metadata,
                "categories": categories,
                "document_id": build_result["document_id"],
                "similarity_relations": build_result["similarity_relations"]
            }
        except ValueError as e:
            raise ApplicationError(
                code=400,
                message="메타데이터 또는 카테고리 생성에 실패했습니다.",
                result={"error": str(e)}
            )

    def batch_extract_and_load_from_jsonl(self, content: str) -> Dict[str, Any]:
        """JSONL 콘텐츠를 배치 처리하여 메타데이터 추출 및 Graph DB 적재"""
        lines = [line for line in content.strip().split("\n") if line.strip()]
        success_count = 0
        errors = []

        for idx, line in enumerate(lines, start=1):
            try:
                text_data = json.dumps(json.loads(line), ensure_ascii=False)
                self.extract_and_load_to_graph(text_data)
                success_count += 1
            except json.JSONDecodeError as e:
                errors.append({"line": idx, "error": f"JSON 파싱 실패: {str(e)}"})

        return {"success": success_count, "failed": len(errors), "errors": errors}
