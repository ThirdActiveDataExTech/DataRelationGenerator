from typing import Annotated

from fastapi import APIRouter, Body, UploadFile, File
from fastapi.responses import JSONResponse

from app.schemas.response import APIResponseModel
from app.src.keyvalue_graph.clustering_normalizer import ClusteringNormalizer
from app.src.keyvalue_graph.llm_based_loader import LLMBasedKeyValueLoader

router = APIRouter(prefix="/keyvalue", tags=["keyvalue-graph"])


@router.post("/extract", tags=["keyvalue-graph"], response_model=APIResponseModel, response_class=JSONResponse)
async def extract_key_value_and_build_graph(
        data: Annotated[str, Body(embed=True)],
        centrality_name: Annotated[str, Body(embed=True)] = "병원현황"
):
    """데이터에서 LLM으로 Key-Value를 추출하고 그래프 생성.

    Args:
        data: 원본 데이터 (JSON 문자열)
        centrality_name: 중심 노드 이름 (예: "병원현황", "판례정보")

    Flow:
        1. LLM으로 Key-Value 추출 (원본 필드명)
        2. 한글 매핑 (FIELD_NAME_MAPPING)
        3. GraphDB에 저장:
           - Centrality 노드 (중심 노드)
           - Document 노드
           - Key 노드 (한글)
           - Value 노드
           - NormalizedValue 노드
           - Centrality -[HAS_KEY]-> Key
           - Key -[HAS_VALUE]-> Value
           - Value -[EXTRACTED_FROM]-> Document
           - Value -[SAME_AS]-> NormalizedValue

    Returns:
        그래프 생성 결과
    """
    loader = LLMBasedKeyValueLoader()
    result = loader.extract_and_build_graph(
        data.strip(),
        centrality_name=centrality_name,
        source_file="api_request",
        record_index=0
    )

    if result["success"]:
        return APIResponseModel(
            result=result,
            description=f"Key-Value 그래프 생성 완료 (Centrality: {centrality_name}, Document ID: {result['document_id']})"
        )
    else:
        return APIResponseModel(
            result=result,
            description=f"Key-Value 그래프 생성 실패: {result.get('error', 'Unknown error')}"
        )


@router.post("/batch_extract", tags=["keyvalue-graph"], response_model=APIResponseModel, response_class=JSONResponse)
async def batch_extract_key_value_from_json(
        file: Annotated[UploadFile, File()],
        centrality_name: str = "병원현황"
):
    """JSON 배열 파일에서 Key-Value를 배치 추출하고 그래프 생성.

    Args:
        file: JSON 배열 파일 [{...}, {...}, ...]
        centrality_name: 중심 노드 이름 (예: "병원현황", "판례정보")

    Returns:
        배치 처리 결과 (성공/실패 통계)
    """
    import json

    content = await file.read()
    data_list = json.loads(content.decode("utf-8"))

    if not isinstance(data_list, list):
        data_list = [data_list]

    loader = LLMBasedKeyValueLoader()

    success_count = 0
    failed_count = 0
    results = []

    for idx, data_item in enumerate(data_list):
        data_text = json.dumps(data_item, ensure_ascii=False)
        result = loader.extract_and_build_graph(
            data_text,
            centrality_name=centrality_name,
            source_file=file.filename or "batch_upload",
            record_index=idx
        )

        if result["success"]:
            success_count += 1
        else:
            failed_count += 1

        results.append({
            "index": idx,
            "document_id": result.get("document_id"),
            "success": result["success"],
            "error": result.get("error")
        })

    return APIResponseModel(
        result={
            "total": len(data_list),
            "success": success_count,
            "failed": failed_count,
            "centrality_name": centrality_name,
            "details": results
        },
        description=f"배치 처리 완료 (Centrality: {centrality_name}): {success_count}건 성공, {failed_count}건 실패"
    )


@router.post("/normalize", tags=["keyvalue-graph"], response_model=APIResponseModel, response_class=JSONResponse)
async def normalize_key_values(
        key_name: Annotated[str | None, Body(embed=True)] = None,
        similarity_threshold: Annotated[float, Body(embed=True)] = 0.85,
        min_cluster_size: Annotated[int, Body(embed=True)] = 2
):
    """클러스터링 기반 Value 자동 정규화.

    Args:
        key_name: 특정 Key명만 정규화 (None이면 전체 Key 정규화)
        similarity_threshold: 유사도 임계값 (0~1, 높을수록 엄격). 기본값 0.85
        min_cluster_size: 최소 클러스터 크기 (작은 클러스터 무시). 기본값 2

    Flow:
        1. Key별로 연결된 모든 Value 조회
        2. Value 간 유사도 행렬 계산 (RapidFuzz token_sort_ratio)
        3. DBSCAN 클러스터링으로 유사한 Value 그룹화
        4. 각 클러스터의 대표값 선정 (빈도수 높은 것, 동률이면 긴 것)
        5. NormalizedValue 노드 생성 및 SAME_AS 관계 생성

    Returns:
        정규화 통계 (Key 수, 클러스터 수, 정규화된 Value 수)
    """
    normalizer = ClusteringNormalizer()

    if key_name:
        result = normalizer.normalize_key_values(
            key_name,
            similarity_threshold=similarity_threshold,
            min_cluster_size=min_cluster_size
        )
        return APIResponseModel(
            result=result,
            description=f"Key '{key_name}' 정규화 완료: {result['clusters_count']}개 클러스터, {result['normalized_count']}개 정규화"
        )
    else:
        result = normalizer.normalize_all_keys(
            similarity_threshold=similarity_threshold,
            min_cluster_size=min_cluster_size
        )
        return APIResponseModel(
            result=result,
            description=(
                    f"전체 정규화 완료: {result['keys_count']}개 Key, "
                    f"{result['clusters_count']}개 클러스터, "
                    f"{result['normalized_count']}개 정규화"
                    + (f", {result['failed_keys']}개 Key 실패" if result.get('failed_keys', 0) > 0 else "")
            )
        )
