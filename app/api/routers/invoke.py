from typing import Annotated

from fastapi import APIRouter, Body, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.db import get_db
from app.schemas.meta import MetadataRequest
from app.schemas.response import APIResponseModel
from app.src.meta_extract.meta_extract import MetadataService

router = APIRouter(prefix="/invoke", tags=["invoke"])

service = MetadataService()


@router.post("", tags=["invoke"], response_model=APIResponseModel, response_class=JSONResponse)
async def extract_metadata(
        db: Annotated[Session, Depends(get_db)],
        request: Annotated[MetadataRequest, Body(...)]
):
    """메타데이터 추출 엔드포인트"""
    meta_info = service.extract_metadata(request.text.strip(), request.prompt_name, request.prompt_label, db)
    return APIResponseModel(result=meta_info, description="메타데이터 추출 완료")

