from typing import Annotated

from fastapi import APIRouter, Body, Depends, Path, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.db import get_db
from app.schemas.meta import MetadataRequest, SearchField, SearchOption
from app.schemas.response import APIResponseModel, DataInfoResponse
from app.src.meta_extract.meta_extract import MetadataService

router = APIRouter(prefix="/metadata", tags=["metadata"])

service = MetadataService()


@router.post("/init", tags=["metadata"], response_model=APIResponseModel, response_class=JSONResponse)
async def init_db():
    """메타데이터 테이블 초기화 엔드포인트"""
    service.create_db_and_tables()
    return APIResponseModel(result={}, description="메타데이터 테이블 초기화 완료")


@router.put("/{data_id}", tags=["metadata"], response_model=APIResponseModel, response_class=JSONResponse)
async def update_metadata(
        request: Annotated[MetadataRequest, Body(...)],
        db: Annotated[Session, Depends(get_db)],
        data_id: Annotated[int, Path(..., description="수정할 메타데이터 ID")]
):
    """메타데이터 업데이트"""
    result = service.update_metadata(request.text.strip(), data_id, db)
    return APIResponseModel(result=DataInfoResponse.from_orm(result), description="메타데이터 업데이트 완료")


@router.delete("/{data_id}", tags=["metadata"], response_model=APIResponseModel, response_class=JSONResponse)
async def delete_metadata(
        db: Annotated[Session, Depends(get_db)],
        data_id: Annotated[int, Path(..., description="삭제할 메타데이터 ID")],
):
    """메타데이터 삭제"""
    service.delete_metadata_by_id(data_id, db)
    return APIResponseModel(result={"deleted_id": data_id}, description="메타데이터 삭제 완료")


@router.get("", tags=["metadata"], response_model=APIResponseModel, response_class=JSONResponse)
async def get_metadata(
        db: Annotated[Session, Depends(get_db)],
        search_option: SearchOption,
        search_field: SearchField | None = None,
        query: Annotated[str, Query()] | None = None,
):
    """메타데이터 검색 엔드포인트"""
    result = service.get_metadatas(query, search_option, search_field, db)
    return APIResponseModel(result=result, description="메타데이터 검색 완료(검색된 데이터 수: {})".format(len(result)))
