from typing import Annotated

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

from app.schemas.response import APIResponseModel
from app.src.agent.agent import MetadataAgent

router = APIRouter(prefix="/invoke", tags=["invoke"])


@router.post("", tags=["invoke"], response_model=APIResponseModel, response_class=JSONResponse)
async def extract_and_load_to_graph(text: Annotated[str, Body(embed=True)]):
    """메타데이터 추출 후 Graph DB 적재"""
    agent = MetadataAgent()
    result = await agent.run_agent_workflow(origin_text=text.strip())
    return APIResponseModel(result=result, description="메타데이터 추출 후 GraphDB 적재 완료")
