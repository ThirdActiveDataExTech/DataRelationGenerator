"""서비스 자체 로직에 관한 커스텀 예외처리 작성"""

from starlette import status

from app.config import settings
from app.exceptions.base import ApplicationError


class TokenValidationError(ApplicationError):
    """유효하지 않은 토큰 설정"""

    def __init__(self, x_token):
        self.code = int(f"{settings.SERVICE_CODE}{status.HTTP_401_UNAUTHORIZED}")
        self.message = "Invalid x-token header"
        self.result = {"current_x_token": x_token}


class CatalogEntryNotFoundError(ApplicationError):
    """① 통합 카탈로그(catalog-service)에서 엔트리를 찾지 못함"""

    def __init__(self, catalog_key):
        self.code = int(f"{settings.SERVICE_CODE}{status.HTTP_404_NOT_FOUND}")
        self.message = "Catalog entry not found"
        self.result = {"catalog_key": catalog_key}


class DatasetNotFoundError(ApplicationError):
    """원장에 등록되지 않은 데이터셋(table_name)"""

    def __init__(self, table_name):
        self.code = int(f"{settings.SERVICE_CODE}{status.HTTP_404_NOT_FOUND}")
        self.message = "Dataset not found"
        self.result = {"table_name": table_name}


class RelationNotFoundError(ApplicationError):
    """원장에 없는 관계(id)"""

    def __init__(self, relation_id):
        self.code = int(f"{settings.SERVICE_CODE}{status.HTTP_404_NOT_FOUND}")
        self.message = "Relation not found"
        self.result = {"relation_id": relation_id}


class DuplicateDatasetContentError(ApplicationError):
    """같은 내용(content_sha256)의 데이터파일이 다른 이름으로 이미 등록됨"""

    def __init__(self, table_name):
        self.code = int(f"{settings.SERVICE_CODE}{status.HTTP_409_CONFLICT}")
        self.message = "Dataset with the same content is already registered"
        self.result = {"table_name": table_name}
