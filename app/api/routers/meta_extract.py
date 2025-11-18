from fastapi import APIRouter, Body, Depends
from app.schemas.response import APIResponseModel
from app.src.meta_extract.meta_extract import generate_metadata
from app.db import save_metadata_to_db, create_db_and_tables, get_db
from app.schemas.meta import DataInfo
from fastapi.responses import JSONResponse
from typing import Annotated
from pydantic import BaseModel
from datetime import datetime
from sqlalchemy.orm import Session
import json

router = APIRouter(prefix="/meta_extract", tags=["meta_extract"])


class MetadataRequest(BaseModel):
    text: str = "데이터 값 입력 ex. 대법원 2025.09.18 선고 2023다221885 전원합의체 판결 ★ [손해배상(기)]〈중도상환수수료가 이자제한법 제4조 제1항에 따른 간주이자에 해당하여 이자제한법상 최고이자율 제한에 관한 규정이 적용되는지 문제된 사건〉 [공2025하,1874]"
    prompt_name: str = "generate_metadata"
    prompt_label: str = "production"


@router.post("/extract", tags=["meta_extract"], response_model=APIResponseModel, response_class=JSONResponse)
async def extract_metadata(
        request: Annotated[MetadataRequest, Body(...)]
):
    """메타데이터 추출 엔드포인트 - db 저장 없이 추출만 수행"""
    try:
        data_text = request.text.strip()
        prompt_name = request.prompt_name
        prompt_label = request.prompt_label
        meta_info = generate_metadata(data_text, prompt_name=prompt_name, prompt_label=prompt_label)
        return APIResponseModel(result=meta_info, description="메타데이터 추출 완료")
    except Exception as e:
        return APIResponseModel(code=500, message="메타데이터 추출 실패", result={}, description=str(e))


@router.post("/extract_save_db", tags=["meta_extract"], response_model=APIResponseModel, response_class=JSONResponse)
async def extract_save_db(
        request: Annotated[MetadataRequest, Body(...)]
):
    """메타데이터 추출 엔드포인트"""
    try:
        data_text = request.text.strip()
        prompt_name = request.prompt_name
        prompt_label = request.prompt_label
        meta_info = generate_metadata(data_text, prompt_name=prompt_name, prompt_label=prompt_label)
        save_metadata_to_db(
            origin_text=data_text,
            topics=meta_info.get("topics", []),
            tags=meta_info.get("tags", []),
            description=meta_info.get("description", ""),
            extract_time=datetime.now()
        )
        return APIResponseModel(result=meta_info, description="메타데이터 추출 후 저장 완료")
    except Exception as e:
        return APIResponseModel(code=500, message="메타데이터 추출 실패", result={}, description=str(e))


@router.post("/init_db", tags=["meta_extract"], response_model=APIResponseModel, response_class=JSONResponse)
async def init_db():
    """메타데이터 테이블 초기화 엔드포인트"""
    try:
        create_db_and_tables()
        return APIResponseModel(result={}, description="메타데이터 테이블 초기화 완료")
    except Exception as e:
        return APIResponseModel(code=500, message="메타데이터 테이블 초기화 실패", result={}, description=str(e))


@router.get("/get_all_db", tags=["meta_extract"], response_model=APIResponseModel, response_class=JSONResponse)
async def get_all_metadata(db: Annotated[Session, Depends(get_db)]):
    """DB에 저장된 메타데이터 조회"""
    try:
        result = []
        for data_info in db.query(DataInfo).all():
            result.append({
                "id": data_info.id,
                "origin_text": data_info.origin_text,
                "topics": data_info.topics,
                "tags": data_info.tags,
                "description": data_info.description,
                "extract_time": data_info.extract_time
            })
        return APIResponseModel(result=result, description="메타데이터 조회 완료")
    except Exception as e:
        return APIResponseModel(code=500, message="메타데이터 조회 실패", result=[], description=str(e))
