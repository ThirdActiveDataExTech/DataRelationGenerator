from fastapi import APIRouter, Body
from app.schemas.response import APIResponseModel
from app.src.meta_extract.meta_extract import generate_metadata
from fastapi.responses import JSONResponse
from typing import Annotated
from pydantic import BaseModel

router = APIRouter(prefix="/meta_extract", tags=["meta_extract"])


class MetadataRequest(BaseModel):
    text: str


@router.post("/extract", tags=["meta_extract"], response_model=APIResponseModel, response_class=JSONResponse)
async def extract_metadata(
        request: Annotated[MetadataRequest, Body(...)]
):
    """메타데이터 추출 엔드포인트"""
    try:
        data_text = request.text.strip()
        mata_info = generate_metadata(data_text)
        return APIResponseModel(result=mata_info, description="메타데이터 추출 완료")
    except Exception as e:
        return APIResponseModel(code=500, message="메타데이터 추출 실패", result={}, description=str(e))
