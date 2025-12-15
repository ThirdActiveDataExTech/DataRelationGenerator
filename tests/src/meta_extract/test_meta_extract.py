from unittest.mock import Mock, patch

import pytest

from app.exceptions.service import ApplicationError
from app.schemas.meta import DataInfo
from app.src.meta_extract.meta_extract import MetadataService


@pytest.fixture
def mock_llm_response():
    """메타 추출해주는 LLM mocking"""
    mock = Mock()
    mock.generate_llm_response.return_value = {
        "topics": ["topic1", "topic2"],
        "tags": ["tag1", "tag2"],
        "description": "test description"
    }
    return mock


@pytest.fixture
def mock_repository():
    """Metadata DB Repository mocking"""
    mock = Mock()
    mock.get_all_metadata.return_value = [
        DataInfo(id=1, origin_text="test", topics='["topic1"]', tags='["tag1"]', description="test")
    ]
    mock.get_data_by_id.return_value = DataInfo(id=1, origin_text="test", topics='["topic1"]', tags='["tag1"]',
                                                description="test")
    return mock


def test_extract_metadata_success(mock_llm_response, mock_repository):
    """메타데이터 추출 성공 테스트"""
    metadata_service = MetadataService(mock_llm_response, mock_repository)

    result = metadata_service.extract_metadata(
        data_text="test data test",
        db=Mock()
    )

    mock_llm_response.generate_llm_response.assert_called_once_with(variables={"data_text": "test data test"})
    mock_repository.save_metadata_to_db.assert_called_once()
    assert result["topics"] == ["topic1", "topic2"]
    assert result["tags"] == ["tag1", "tag2"]
    assert result["description"] == "test description"


def test_get_metadatas_field_option_success(mock_llm_response, mock_repository):
    """필드별 메타데이터 검색 처리 성공 테스트"""
    metadata_service = MetadataService(mock_llm_response, mock_repository)
    dbMock = Mock()
    with patch.object(metadata_service, '_search_metadatas') as mock_search:
        with patch.object(metadata_service, '_format_metadata_response') as mock_format:
            mock_search.return_value = []
            mock_format.return_value = []

            result = metadata_service.get_metadatas(
                query="test",
                search_option="field",
                search_field="origin_text",
                db=dbMock
            )
            mock_search.assert_called_once_with("test", "origin_text", dbMock)
            mock_format.assert_called_once()


def test_get_metadatas_field_option_missing_params(mock_llm_response, mock_repository):
    """필드별 검색 - 파라미터 누락 시 에러 발생 테스트"""
    metadata_service = MetadataService(mock_llm_response, mock_repository)
    with pytest.raises(ApplicationError) as err_info:
        metadata_service.get_metadatas(
            query=None,
            search_option="field",
            search_field=None,
            db=Mock()
        )

    assert err_info.value.code == 400
    assert "InvalidItemParams: search_field와 query는 필수 입력값입니다." == err_info.value.message


def test_search_metadatas_by_id_invalid(mock_llm_response, mock_repository):
    """잘못된 ID로 검색 테스트"""
    db_mock = Mock()
    metadata_service = MetadataService(mock_llm_response, mock_repository)
    with pytest.raises(ApplicationError) as exc_info:
        metadata_service._search_metadatas("invalid", "id", db_mock)

    assert exc_info.value.code == 400
    assert "정수여야 합니다" in exc_info.value.message


def test_search_metadatas_by_id(mock_llm_response, mock_repository):
    """ID로 메타데이터 검색 테스트"""
    db_mock = Mock()
    metadata_service = MetadataService(mock_llm_response, mock_repository)
    result = metadata_service._search_metadatas("1", "id", db_mock)

    mock_repository.get_data_by_id.assert_called_once_with(1, db_mock)
    assert len(result) == 1


def test_format_metadata_response(mock_llm_response, mock_repository):
    """메타데이터 응답 포맷팅 테스트"""
    metadata_service = MetadataService(mock_llm_response, mock_repository)

    data_infos = [
        DataInfo(id=1, origin_text="test", topics='["topic1"]', tags='["tag1"]', description="test")
    ]

    with patch('app.src.meta_extract.meta_extract.DataInfoResponse') as mock_response:
        mock_response.from_orm.return_value = {"id": 1, "origin_text": "test"}

        result = metadata_service._format_metadata_response(data_infos)

        mock_response.from_orm.assert_called_once()
