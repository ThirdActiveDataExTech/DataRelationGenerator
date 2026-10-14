from fastapi import APIRouter, Depends
from starlette.responses import JSONResponse

from app.api.routers import dataset, explore, keyvalue_graph, pilot, workflow
from app.dependencies import get_token_header

api_router = APIRouter(dependencies=[Depends(get_token_header)], default_response_class=JSONResponse)

api_router.include_router(dataset.router)
api_router.include_router(explore.router)
api_router.include_router(pilot.router)
api_router.include_router(keyvalue_graph.router)
api_router.include_router(workflow.router)
