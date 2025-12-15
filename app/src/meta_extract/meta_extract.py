from datetime import datetime
from typing import Dict, Any, List

from app.exceptions.service import ApplicationError
from app.repositories.meta_repository import MetadataDBRepository
from app.schemas.meta import DataInfo
from app.schemas.response import DataInfoResponse
from app.src.meta_extract.llm_response import LLMResponse


class MetadataService:
    """메타데이터 추출 서비스 클래스"""

    def __init__(self, llm_response: LLMResponse = None, repository: MetadataDBRepository = None):
        """Init 매서드 - 의존성 주입"""
        self.llm_response = llm_response or LLMResponse("meta_extract")
        self.repository = repository or MetadataDBRepository()

    def extract_metadata(self, data_text: str, prompt_name: str, prompt_label: str, db) -> Dict[str, Any]:
        """메타데이터 추출 후 DB 저장 처리"""
        meta_info = self.llm_response.generate_llm_response(prompt_name, prompt_label,
                                                            variables={"data_text": data_text})
        if isinstance(meta_info, str):
            return {"error": "String response received instead of metadata dict"}

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
        meta_info = self.llm_response.generate_llm_response(prompt_name, prompt_label,
                                                            variables={"data_text": data_text})
        if isinstance(meta_info, str):
            return {"error": "String response received instead of metadata dict"}
            
        result = self.repository.update_metadata(data_text, db, data_id, meta_info)
        return result

    def delete_metadata_by_id(self, data_id: int, db):
        """ID로 메타데이터 삭제"""
        self.repository.delete_metadata_by_id(data_id, db)

    def get_metadatas(self, query: str | None, search_option: str, search_field: str | None, db):
        """메타데이터 검색 처리"""
        if search_option == "all":
            data_infos = self._get_all_metadata(db)
        elif search_option == "field":
            if not (search_field and query):
                raise ApplicationError(code=400, message="InvalidItemParams: search_field와 query는 필수 입력값입니다.",
                                       result={"None"})
            data_infos = self._search_metadatas(query, search_field, db)
        else:
            data_infos = []
        return self._format_metadata_response(data_infos)

    def _get_all_metadata(self, db):
        """모든 메타데이터 조회"""
        data_infos = self.repository.get_all_metadata(db)
        return data_infos

    def _search_metadatas(self, query: str, search_field: str, db):
        """메타데이터 검색"""
        if search_field == "id":
            try:
                data_infos = [self.repository.get_data_by_id(int(query), db)]
            except ValueError as e:
                raise ApplicationError(code=400, message="ValueError: ID는 정수여야 합니다.", result=str(e))
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

    @staticmethod
    def _format_metadata_response(data_infos: List[DataInfo]):
        """메타데이터 응답 포맷팅"""
        result = [DataInfoResponse.from_orm(data_info) for data_info in data_infos]
        return result

    def create_db_and_tables(self):
        """메타데이터 테이블 초기화"""
        self.repository.create_db_and_tables()
