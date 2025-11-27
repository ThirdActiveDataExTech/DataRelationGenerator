from starlette import status

from app.config import settings
from app.exceptions.base import ApplicationError

from typing import Union


class MetadataNotFoundError(ApplicationError):
    """메타데이터 조회 시 입력 값의 메타데이터가 존재하지 않을 때 발생하는 예외"""

    def __init__(self, data_field: str, data_content: Union[int, str]):
        self.code = int(f"{settings.SERVICE_CODE}{status.HTTP_404_NOT_FOUND}")
        self.message = f"해당 Column({data_field}: {data_content})의 메타데이터가 존재하지 않습니다."
        self.result = {"exception": "MetadataNotFoundError"}


class MetadataDBError(ApplicationError):
    """메타데이터 DB 처리 중 에러 발생 시 예외"""

    def __init__(self, exc):
        self.code = int(f"{settings.SERVICE_CODE}{status.HTTP_500_INTERNAL_SERVER_ERROR}")
        self.message = f"메타데이터 DB 처리 중 에러가 발생했습니다."
        self.result = {"exception": exc}
