from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query
from app.schemas.response import APIResponseModel, DataInfoResponse
from app.schemas.meta import MetadataRequest, SearchField
from app.db import get_db
from app.src.meta_extract.meta_extract import MetadataService
from fastapi.responses import JSONResponse
from typing import Annotated
from sqlalchemy.orm import Session

router = APIRouter(prefix="/metadata", tags=["metadata"])

service = MetadataService()


@router.post("/extract", tags=["metadata"], response_model=APIResponseModel, response_class=JSONResponse)
async def extract_metadata(
        db: Annotated[Session, Depends(get_db)],
        request: Annotated[MetadataRequest, Body(...)]
):
    """메타데이터 추출 엔드포인트"""
    meta_info = service.extract_metadata(request.text.strip(), request.prompt_name, request.prompt_label, db)
    return APIResponseModel(result=meta_info, description="메타데이터 추출 완료")


@router.post("/database/init", tags=["metadata"], response_model=APIResponseModel, response_class=JSONResponse)
async def init_db():
    """메타데이터 테이블 초기화 엔드포인트"""
    service.create_db_and_tables()
    return APIResponseModel(result={}, description="메타데이터 테이블 초기화 완료")



@router.get("", tags=["metadata"], response_model=APIResponseModel, response_class=JSONResponse)
async def get_all_metadata(db: Annotated[Session, Depends(get_db)]):
    """DB에 저장된 메타데이터 조회"""
    result = service.get_all_metadata(db)
    return APIResponseModel(result=result, description="메타데이터 전체 조회 완료")


@router.put("/{data_id}", tags=["metadata"], response_model=APIResponseModel, response_class=JSONResponse)
async def update_metadata(
        request: Annotated[MetadataRequest, Body(...)],
        db: Annotated[Session, Depends(get_db)],
        data_id: Annotated[int, Path(..., description="수정할 메타데이터 ID")]
):
    """메타데이터 업데이트"""
    result = service.update_metadata(request.text.strip(), request.prompt_name, request.prompt_label, data_id, db)
    return APIResponseModel(result=DataInfoResponse.from_orm(result), description="메타데이터 업데이트 완료")


@router.delete("/{data_id}", tags=["metadata"], response_model=APIResponseModel, response_class=JSONResponse)
async def delete_metadata(
        db: Annotated[Session, Depends(get_db)],
        data_id: Annotated[int, Path(..., description="삭제할 메타데이터 ID")],
):
    """메타데이터 삭제"""
    service.delete_metadata_by_id(data_id, db)
    return APIResponseModel(result={"deleted_id": data_id}, description="메타데이터 삭제 완료")


@router.get("/search/datas", tags=["metadata"], response_model=APIResponseModel, response_class=JSONResponse)
async def search_datas(
        db: Annotated[Session, Depends(get_db)],
        search_field: SearchField,
        query: Annotated[str, Query()] = "판례",
):
    """메타데이터 검색 엔드포인트"""
    data_infos = service.search_metadatas(query, search_field.value, db)
    result = [DataInfoResponse.from_orm(data_info) for data_info in data_infos]
    return APIResponseModel(result=result, description="메타데이터 검색 완료(검색된 데이터 수: {})".format(len(result)))
