from typing import Optional, Any
import logging
import json
from datetime import datetime

from sqlalchemy import DateTime, create_engine
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from app.schemas.meta import DataInfo
from app.exceptions.meta_extract import MetadataNotFoundError, MetadataDBError
from app.schemas.meta import Base
from app.config import settings


class MetadataDBRepository:
    """메타데이터 DB 처리 클래스"""

    def __init__(self, sqlite_file_name: str = settings.SQLITE_FILE_NAME):
        sqlite_url = f"sqlite:///{sqlite_file_name}"
        self.engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})

    @staticmethod
    def get_data_by_id(data_id: int, db: Session) -> DataInfo | None:
        """ID로 메타데이터 조회"""
        try:
            data_info = db.query(DataInfo).filter(DataInfo.id == data_id).first()
        except OperationalError as e:
            logging.error(f"Database operation failed: {e}")
            raise MetadataDBError(str(e))

        if not data_info:
            logging.warning(f"Metadata id ({data_id}) not found.")
            return None
        return data_info

    @staticmethod
    def get_all_metadata(db: Session) -> list[DataInfo]:
        """모든 메타데이터 조회"""
        try:
            data_infos = db.query(DataInfo).all()
            return data_infos
        except OperationalError as e:
            logging.error(f"Database operation failed: {e}")
            raise MetadataDBError(str(e))

    def update_metadata(self, data_text: str, db: Session,
                        data_id: int, meta_info: dict) -> Optional[DataInfo]:
        """메타데이터 생성 후 DB 업데이트 처리"""
        data_info = self.get_data_by_id(data_id, db)
        if not data_info:
            raise MetadataNotFoundError("id", data_id)
        topics_json = self._serialize_list_to_json(meta_info.get("topics", []))
        tags_json = self._serialize_list_to_json(meta_info.get("tags", []))

        setattr(data_info, 'origin_text', data_text)
        setattr(data_info, 'topics', topics_json)
        setattr(data_info, 'tags', tags_json)
        setattr(data_info, 'description', meta_info.get("description", ""))
        setattr(data_info, 'extract_time', datetime.now())

        result = self._update_metadata_in_db(data_info, db)
        return result

    @staticmethod
    def _serialize_list_to_json(data: list, default_value: list = None) -> str:
        """리스트를 JSON 문자열로 변환하는 함수"""
        if default_value is None:
            default_value = []

        try:
            return json.dumps(data if data else default_value)
        except (TypeError, ValueError) as e:
            logging.error(f"JSON serialization failed: {e}")
            return json.dumps(default_value, ensure_ascii=False)

    @staticmethod
    def _update_metadata_in_db(data_info: DataInfo, db: Session) -> DataInfo:
        """메타데이터 DB 업데이트 처리"""
        try:
            db.commit()
            db.refresh(data_info)
            return data_info
        except OperationalError as e:
            db.rollback()
            raise MetadataDBError(str(e))

    def save_metadata_to_db(self, origin_text: str, topics: list[str], tags: list[str], description: str,
                            extract_time: DateTime, db: Session) -> DataInfo:
        """메타데이터 저장 처리"""
        topics_json = self._serialize_list_to_json(topics)
        tags_json = self._serialize_list_to_json(tags)

        db_data_info = DataInfo(origin_text=origin_text, topics=topics_json, tags=tags_json, description=description,
                                extract_time=extract_time)

        try:
            db.add(db_data_info)
            db.commit()
            return db_data_info
        except OperationalError as e:
            db.rollback()
            raise MetadataDBError(str(e))

    def delete_metadata_by_id(self, data_id: int, db: Session):
        """ID로 메타데이터 삭제"""
        data_info = self.get_data_by_id(data_id, db)
        if not data_info:
            logging.error(f"Metadata deletion failed: id: {data_id} not found.")
            raise MetadataNotFoundError("id", data_id)
        try:
            db.delete(data_info)
            db.commit()
        except OperationalError as e:
            db.rollback()
            logging.error(f"Metadata deletion failed: {e}")
            raise MetadataDBError(str(e))

    @staticmethod
    def execute_search_query(search_field: Any, query: str, db: Session) -> list[DataInfo]:
        """필드별 검색 쿼리 실행"""
        try:
            result = db.query(DataInfo).filter(search_field.like(f'%{query}%')).all()
            return result
        except OperationalError as e:
            logging.error(f"Database operation failed: {e}")
            raise MetadataDBError(str(e))

    def create_db_and_tables(self):
        """Create db and tables."""
        try:
            Base.metadata.create_all(self.engine)
        except SQLAlchemyError as e:
            logging.error(f"Database initialization failed while creating tables: {e}")
            raise MetadataDBError(str(e))
