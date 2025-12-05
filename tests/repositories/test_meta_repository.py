from datetime import datetime

import pytest

from app.exceptions.meta_extract import MetadataNotFoundError
from app.repositories.meta_repository import MetadataDBRepository
from app.schemas.meta import DataInfo


def test_get_data_by_id_success(session):
    """ID로 데이터 조회 성공 테스트"""
    MetadataDBRepository("database.db")
    test_data_info = DataInfo(
        id=1,
        origin_text="test text",
        topics="[\"topic1\", \"topic2\"]",
        tags="[\"tag1\", \"tag2\"]",
        description="test description",
        extract_time=datetime.now()
    )
    session.add(test_data_info)
    session.commit()

    result = MetadataDBRepository.get_data_by_id(1, session)

    assert result.id == 1
    assert result.origin_text == "test text"
    assert result.topics == "[\"topic1\", \"topic2\"]"
    assert result.tags == "[\"tag1\", \"tag2\"]"
    assert result.description == "test description"


def test_get_data_by_id_not_found(session):
    """ID로 데이터 조회 실패 테스트"""
    result = MetadataDBRepository.get_data_by_id(123, session)
    assert result is None


def test_get_all_metadata_success(session):
    """전체 메타데이터 조회 성공 테스트"""
    test_data_info = [
        DataInfo(
            id=1,
            origin_text="test text",
            topics="[\"topic1\", \"topic2\"]",
            tags="[\"tag1\", \"tag2\"]",
            description="test description",
            extract_time=datetime.now()
        ),
        DataInfo(
            id=2,
            origin_text="test text2",
            topics="[\"topic3\", \"topic4\"]",
            tags="[\"tag3\", \"tag4\"]",
            description="test description2",
            extract_time=datetime.now()
        )
    ]

    session.add(test_data_info[0])
    session.add(test_data_info[1])
    session.commit()

    results = MetadataDBRepository.get_all_metadata(session)

    assert len(results) == 2
    assert results[0].id == 1


def test_serialize_list_to_json_success():
    """리스트를 JSON으로 직렬화 성공 테스트"""
    data = ["item1", "item2"]
    result = MetadataDBRepository._serialize_list_to_json(data)
    assert result == "[\"item1\", \"item2\"]"


def test_serialize_list_to_json_empty():
    """리스트를 JSON으로 직렬화 성공 테스트 - 빈 리스트"""
    result = MetadataDBRepository._serialize_list_to_json([])
    assert result == "[]"


def test_save_metadata_to_db_success(session):
    """메타데이터 저장 성공 테스트"""
    repository = MetadataDBRepository("database.db")

    result = repository.save_metadata_to_db(
        origin_text="test text",
        topics=["topic1"],
        tags=["tag1"],
        description="test description",
        extract_time=datetime.now(),
        db=session
    )

    assert result.origin_text == "test text"
    assert result.description == "test description"


def test_delete_metadata_by_id_success(session):
    """메타데이터 삭제 성공 테스트"""
    repository = MetadataDBRepository("database.db")
    test_data_info = DataInfo(
        id=1,
        origin_text="test text",
        topics="[\"topic1\", \"topic2\"]",
        tags="[\"tag1\", \"tag2\"]",
        description="test description",
        extract_time=datetime.now()
    )
    session.add(test_data_info)
    session.commit()

    repository.delete_metadata_by_id(1, session)

    result = MetadataDBRepository.get_data_by_id(1, session)
    assert result is None


def test_delete_metadata_by_id(session):
    """ID로 메타데이터 삭제 에러 발생 테스트"""

    repository = MetadataDBRepository("database.db")

    with pytest.raises(MetadataNotFoundError) as exc_info:
        repository.delete_metadata_by_id(123, session)

    assert exc_info.value.code == 100404
    assert "메타데이터가 존재하지 않습니다." in exc_info.value.message
    assert exc_info.value.result == {"exception": "MetadataNotFoundError"}


def test_execute_search_query_success(session):
    """검색 쿼리 실행 성공 테스트"""
    test_data_info = [
        DataInfo(
            id=1,
            origin_text="test text",
            topics="[\"topic1\", \"topic2\"]",
            tags="[\"tag1\", \"tag2\"]",
            description="test description",
            extract_time=datetime.now()
        ),
        DataInfo(
            id=2,
            origin_text="test text2",
            topics="[\"topic3\", \"topic4\"]",
            tags="[\"tag3\", \"tag4\"]",
            description="test description2",
            extract_time=datetime.now()
        )
    ]
    session.add(test_data_info[0])
    session.add(test_data_info[1])
    session.commit()

    results = MetadataDBRepository.execute_search_query(
        DataInfo.origin_text, "test", session
    )

    assert len(results) == 2
    assert results[0].origin_text == "test text"
