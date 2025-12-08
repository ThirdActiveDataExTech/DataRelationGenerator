from datetime import datetime

from langchain_core.tools import tool

from app.repositories.meta_repository import MetadataDBRepository
from app.src.meta_extract.llm_response import LLMResponse


@tool
def extract_metadata_llm_tool(text: str, prompt_name: str, prompt_label: str) -> dict:
    """LLM을 통한 메타데이터 추출"""
    llm_response = LLMResponse()
    metadata = llm_response.extract_metadata_by_llm(text, prompt_name, prompt_label)
    return metadata


@tool
def save_metadata_sqlite_tool(text: str, metadata: dict, db) -> int:
    """SQLite에 메타데이터 저장"""
    repository = MetadataDBRepository()
    repository.save_metadata_to_db(
        origin_text=text,
        topics=metadata.get("topics", []),
        tags=metadata.get("tags", []),
        description=metadata.get("description", ""),
        extract_time=datetime.now(),
        db=db
    )
    # TODO: 저장된 레코드 ID 반환 (실제로는 repository에서 반환받아야 함)
    return 1  # <--- tmp

