from datetime import datetime
from typing import Dict, Any

from .llm_response import LLMResponse
from app.repositories.meta_repository import MetadataDBRepository
from app.schemas.response import DataInfoResponse

from app.schemas.meta import DataInfo
from fastapi import HTTPException


class MetadataService:
    """메타데이터 추출 서비스 클래스"""

    def __init__(self, llm_response: LLMResponse = None, repository: MetadataDBRepository = None):
        """Init 매서드 - 의존성 주입"""
        self.llm_response = llm_response or LLMResponse()
        self.repository = repository or MetadataDBRepository()

    def extract_metadata(self, data_text: str, prompt_name: str, prompt_label: str, db) -> Dict[str, Any]:
        """메타데이터 추출 후 DB 저장 처리"""
        meta_info = self.llm_response.extract_metadata_by_llm(data_text, prompt_name, prompt_label)
        self.repository.save_metadata_to_db(
            origin_text=data_text,
            topics=meta_info.get("topics", []),
            tags=meta_info.get("tags", []),
            description=meta_info.get("description", ""),
            extract_time=datetime.now(),
            db=db
        )
        return meta_info

    def update_metadata(self, data_text: str, prompt_name: str, prompt_label: str, data_id: int, db) -> Dict[str, Any]:
        """메타데이터 추출 후 DB 저장 처리"""
        meta_info = self.llm_response.extract_metadata_by_llm(data_text, prompt_name, prompt_label)
        result = self.repository.update_metadata(data_text, db, data_id, meta_info)
        return result

    def get_all_metadata(self, db):
        """모든 메타데이터 조회"""
        data_infos = self.repository.get_all_metadata(db)
        result = [DataInfoResponse.from_orm(data_info) for data_info in data_infos]
        return result

    def delete_metadata_by_id(self, data_id: int, db):
        """ID로 메타데이터 삭제"""
        self.repository.delete_metadata_by_id(data_id, db)

    def search_metadatas(self, query: str, search_field: str, db):
        """메타데이터 검색"""
        if search_field == "id":
            try:
                data_infos = [self.repository.get_data_by_id(int(query), db)]
            except ValueError as e:
                raise HTTPException(status_code=400, detail="ValueError: ID는 정수여야 합니다.", headers=str(e))
        elif search_field in ["origin_text", "description"]:
            search_field_attr = getattr(DataInfo, search_field)
            data_infos = self.repository.execute_search_query(search_field_attr, query, db)
        elif search_field in ["tags", "topics"]:
            search_field_attr = getattr(DataInfo, search_field)
            query = "\"" + query + "\""
            data_infos = self.repository.execute_search_query(search_field_attr, query, db)
        else:
            data_infos = []
        if data_infos:
            return data_infos
        return []

    def create_db_and_tables(self):
        """메타데이터 테이블 초기화"""
        self.repository.create_db_and_tables()
