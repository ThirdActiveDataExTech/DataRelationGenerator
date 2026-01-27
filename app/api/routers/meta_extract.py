from typing import Annotated

from fastapi import APIRouter, Body, UploadFile, File
from fastapi.responses import JSONResponse

from app.schemas.response import APIResponseModel
from app.src.meta_extract.meta_extractor import meta_service

router = APIRouter(prefix="/metadata", tags=["metadata"])


@router.post("/extract", tags=["metadata"], response_model=APIResponseModel,
             response_class=JSONResponse)
async def extract_and_load_to_graph(text: Annotated[str, Body(embed=True)]):
    """메타데이터 추출 후 Graph DB 적재 (Step 1)

    Args:
        text: 원본 텍스트 데이터

    Flow:
        Step 1: /metadata/extract - 문서 추가 (text_embedding) ← 현재
        Step 2: /graph/embeddings - 그래프 임베딩 생성
        Step 3: /graph/relations/analyze - 관계 분석

    Returns:
        Document 생성 결과 (metadata, categories, document_id)
    """
    result = await meta_service.extract_and_load_to_graph(text.strip())
    return APIResponseModel(
        result=result,
        description=f"Document 생성 완료 (ID: {result.get('document_id', 'unknown')})"
    )


@router.post("/batch_extract", tags=["metadata"], response_model=APIResponseModel,
             response_class=JSONResponse)
async def batch_extract_and_load_from_jsonl(file: Annotated[UploadFile, File()]):
    """JSONL 파일 배치 처리 - 메타데이터 추출 및 Graph DB 적재

    Args:
        file: JSONL 파일 (각 줄이 하나의 문서)

    Note:
        - 모든 문서 추가 후 /graph/embeddings 호출 필요
        - 대량 문서 처리에 적합
    """
    content = await file.read()
    result = await meta_service.batch_extract_and_load_from_jsonl(content.decode("utf-8"))
    return APIResponseModel(
        result=result,
        description=f"처리 완료: {result['success']}건 성공, {result['failed']}건 실패"
    )
