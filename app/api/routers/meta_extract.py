from typing import Annotated

from fastapi import APIRouter, Body, UploadFile, File
from fastapi.responses import JSONResponse

from app.schemas.response import APIResponseModel
from app.src.meta_extract.meta_extractor import meta_service

router = APIRouter(prefix="/metadata", tags=["metadata"])


@router.post("/extract_and_load_to_graph", tags=["metadata"], response_model=APIResponseModel,
             response_class=JSONResponse)
async def extract_and_load_to_graph(text: Annotated[str, Body(embed=True)]):
    """메타데이터 추출 후 Graph DB 적재 - 카테고리 추출 포함"""
    result = meta_service.extract_and_load_to_graph(text.strip())
    return APIResponseModel(result=result, description="메타데이터 추출 후 GraphDB 적재 완료")


@router.post("/batch_extract_and_load", tags=["metadata"], response_model=APIResponseModel,
             response_class=JSONResponse)
async def batch_extract_and_load_from_jsonl(file: Annotated[UploadFile, File()]):
    """JSONL 파일 업로드하여 각 줄을 문자열로 변환 후 메타데이터 추출 및 Graph DB 적재"""
    content = await file.read()
    result = meta_service.batch_extract_and_load_from_jsonl(content.decode("utf-8"))
    return APIResponseModel(
        result=result,
        description=f"처리 완료: {result['success']}건 성공, {result['failed']}건 실패"
    )
