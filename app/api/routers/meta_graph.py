from typing import Annotated

from fastapi import APIRouter, Body, Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.schemas.meta import MetadataRequest
from app.schemas.response import APIResponseModel
from app.src.graph_workflow.workflow import run_graph_workflow

router = APIRouter()


@router.post("/extract-and-load")
async def extract_and_load_to_graph(
        request: Annotated[MetadataRequest, Body(...)],
        db: Session = Depends(get_db)
):
    """메타데이터 추출 후 Graph DB 적재"""
    result = run_graph_workflow(request.text.strip(), request.prompt_name, request.prompt_label, db)
    return APIResponseModel(result=result, description="메타데이터 추출 후 GraphDB 적재 완료")
