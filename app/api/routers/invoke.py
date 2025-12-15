from typing import Annotated

from fastapi import APIRouter, Body, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.db import get_db
from app.schemas.meta import MetadataRequest
from app.schemas.response import APIResponseModel
from app.src.graph_workflow.agent import MetadataAgent

router = APIRouter(prefix="/invoke", tags=["invoke"])


@router.post("", tags=["invoke"], response_model=APIResponseModel, response_class=JSONResponse)
async def extract_and_load_to_graph(
        db: Annotated[Session, Depends(get_db)],
        request: Annotated[MetadataRequest, Body(...)]
):
    """메타데이터 추출 후 Graph DB 적재"""
    agent = MetadataAgent()
    result = await agent.run_agent_workflow(origin_text = request.text.strip())
    return APIResponseModel(result=result, description="메타데이터 추출 후 GraphDB 적재 완료")
