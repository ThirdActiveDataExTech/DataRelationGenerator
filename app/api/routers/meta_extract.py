from typing import Annotated

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

from app.schemas.response import APIResponseModel
from app.src.meta_extract.meta_extract import MetadataService

router = APIRouter(prefix="/metadata", tags=["metadata"])

service = MetadataService()


@router.post("/extract_and_load_to_graph", tags=["metadata"], response_model=APIResponseModel,
             response_class=JSONResponse)
async def extract_and_load_to_graph(text: Annotated[str, Body(embed=True)]):
    """메타데이터 추출 후 Graph DB 적재 - 카테고리 추출 포함"""
    result = service.extract_and_load_to_graph(text.strip())
    return APIResponseModel(result=result, description="메타데이터 추출 후 GraphDB 적재 완료")
