from typing import Dict, Any, List

from app.repositories.graph_repository import GraphRepository
from app.schemas.meta import DataInfo
from app.schemas.response import DataInfoResponse
from app.src.meta_extract.llm_response import LLMResponse


class MetadataService:
    """메타데이터 추출 서비스 클래스"""

    def __init__(self, llm_response: LLMResponse = None):
        """Init 매서드 - 의존성 주입"""
        self.llm_response = llm_response or LLMResponse("meta_extract")

    @staticmethod
    def _format_metadata_response(data_infos: List[DataInfo]):
        """메타데이터 응답 포맷팅"""
        result = [DataInfoResponse.from_orm(data_info) for data_info in data_infos]
        return result

    @staticmethod
    def _build_category_query(category: Dict[str, Any]) -> str:
        name = category.get("name", "")
        topics = category.get("topics", [])

        parts = [f"MERGE (c:Category {{name: '{name}'}})"]
        for topic in topics:
            relation = topic.get("relation", "BELONGS_TO")
            parts.extend([
                f"WITH c\nMERGE (t:Topic {{name: '{topic.get('topic', '')}'}})",
                f"MERGE (t)-[:{relation}]->(c)"
            ])
        return "\n".join(parts)

    def extract_and_load_to_graph(self, data_text: str) -> Dict[str, Any]:
        """메타데이터 추출 후 Graph DB 적재 - 카테고리 추출 포함"""
        graph_repo = GraphRepository()

        metadata = LLMResponse("meta_extract").generate_llm_response(variables={"data_text": data_text})
        if not isinstance(metadata, dict):
            raise ValueError("Metadata extraction failed")

        cypher_result = LLMResponse("cypher_generate").generate_llm_response(
            variables={"metadata": metadata, "text": data_text})
        if not isinstance(cypher_result, dict):
            raise ValueError("Cypher generation failed")

        doc_query = cypher_result["cypher_query"].replace("apoc.create.uuid()", "randomUUID()")

        existing_cats = graph_repo.execute_query("MATCH (c:Category) RETURN c.name as category")
        cat_names = [item["category"] for item in existing_cats if "category" in item]

        category_result = LLMResponse("category_generate").generate_llm_response(
            variables={"topics": metadata.get("topics", []), "existing_categories": cat_names}
        )
        if not isinstance(category_result, dict):
            raise ValueError("Category generation failed")

        cat_queries = [self._build_category_query(cat) for cat in category_result.get("categories", [])]

        combined_query = ";\n".join([doc_query] + cat_queries)
        graph_repo.execute_query(combined_query)

        return {
            "metadata": metadata,
            "cypher_query": combined_query,
            "categories": category_result.get("categories", [])
        }
