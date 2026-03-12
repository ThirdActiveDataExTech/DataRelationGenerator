from typing import Annotated

from fastapi import APIRouter, Body, UploadFile, File
from fastapi.responses import JSONResponse

from app.schemas.response import APIResponseModel
from app.src.keyvalue_graph.clustering_normalizer import ClusteringNormalizer
from app.src.keyvalue_graph.llm_based_loader import LLMBasedKeyValueLoader
from app.src.keyvalue_graph.schema_analyzer import SchemaAnalyzer

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
        centrality_name: str = "병원현황",
        auto_detect_schema: bool = False,
        sample_size: int = 5
):
    """JSON 배열 파일에서 Key-Value를 배치 추출하고 그래프 생성.

    Args:
        file: JSON 배열 파일 [{...}, {...}, ...]
        centrality_name: 중심 노드 이름 (예: "병원현황", "판례정보"). auto_detect_schema=True면 무시됨
        auto_detect_schema: 스키마 자동 감지 여부 (True: 데이터 분석 후 자동 감지, False: 기본 매핑 사용)
        sample_size: 스키마 분석 시 샘플링할 레코드 개수 (auto_detect_schema=True일 때만 사용)

    Returns:
        배치 처리 결과 (성공/실패 통계)
    """
    import json

    content = await file.read()
    data_list = json.loads(content.decode("utf-8"))

    if not isinstance(data_list, list):
        data_list = [data_list]

    # 스키마 자동 감지 모드
    field_mapping = None
    detected_schema = None

    if auto_detect_schema:
        analyzer = SchemaAnalyzer(sample_size=sample_size)
        detected_schema = analyzer.analyze(data_list)

        # SchemaAnalysisResult의 fields를 field_mapping으로 변환
        # fields: Dict[str, FieldAnalysis] → field_mapping: Dict[str, str]
        field_mapping = {
            original_key: field_info.korean_name
            for original_key, field_info in detected_schema.fields.items()
        }

        # 감지된 centrality_name 사용
        centrality_name = detected_schema.centrality_name

    loader = LLMBasedKeyValueLoader(field_mapping=field_mapping)

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

    # 결과 메시지 구성
    if auto_detect_schema and detected_schema:
        description = (
            f"배치 처리 완료 [자동 스키마 감지]\n"
            f"- 도메인: {detected_schema.domain}\n"
            f"- Centrality: {centrality_name}\n"
            f"- 감지된 필드: {len(field_mapping)}개\n"
            f"- 처리 결과: {success_count}건 성공, {failed_count}건 실패"
        )
        result_data = {
            "total": len(data_list),
            "success": success_count,
            "failed": failed_count,
            "centrality_name": centrality_name,
            "auto_detected": True,
            "detected_schema": {
                "domain": detected_schema.domain,
                "centrality_name": detected_schema.centrality_name,
                "fields": {
                    field_name: {
                        "korean_name": field_info.korean_name,
                        "description": field_info.description,
                        "data_type": field_info.data_type
                    }
                    for field_name, field_info in detected_schema.fields.items()
                },
                "confidence": detected_schema.confidence
            },
            "details": results
        }
    else:
        description = f"배치 처리 완료 (Centrality: {centrality_name}): {success_count}건 성공, {failed_count}건 실패"
        result_data = {
            "total": len(data_list),
            "success": success_count,
            "failed": failed_count,
            "centrality_name": centrality_name,
            "auto_detected": False,
            "details": results
        }

    return APIResponseModel(
        result=result_data,
        description=description
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


@router.post("/analyze_schema", tags=["keyvalue-graph"], response_model=APIResponseModel, response_class=JSONResponse)
async def analyze_data_schema(
        file: Annotated[UploadFile, File()],
        sample_size: Annotated[int, Body(embed=True)] = 5
):
    """데이터 스키마 자동 분석

    파일을 업로드하면 샘플링 + 통계 분석을 통해 스키마를 자동으로 추론합니다.

    Args:
        file: JSON 배열 파일 [{...}, {...}, ...]
        sample_size: 샘플링할 레코드 개수 (기본: 5)

    Flow:
        1. 파일에서 레코드 추출
        2. 샘플링 (균등 분포)
        3. 필드별 통계 수집 (unique 개수, null 비율, 샘플 값)
        4. LLM으로 스키마 추론 (도메인, 필드 한글명, 데이터 타입)

    Returns:
        {
            "domain": "법률",
            "centrality_name": "판례정보",
            "fields": {
                "CASE_NO": {
                    "korean_name": "사건번호",
                    "description": "법원 사건 고유 번호",
                    "data_type": "text"
                },
                ...
            },
            "confidence": 0.95
        }
    """
    import json

    # 파일 읽기
    content = await file.read()
    data_list = json.loads(content.decode("utf-8"))

    if not isinstance(data_list, list):
        data_list = [data_list]

    if not data_list:
        return APIResponseModel(
            result={"error": "데이터가 비어있습니다"},
            description="스키마 분석 실패: 빈 데이터"
        )

    # 스키마 분석
    analyzer = SchemaAnalyzer(sample_size=sample_size)
    schema = analyzer.analyze(data_list)

    return APIResponseModel(
        result={
            "domain": schema.domain,
            "centrality_name": schema.centrality_name,
            "fields": {
                field_name: {
                    "korean_name": field_info.korean_name,
                    "description": field_info.description,
                    "data_type": field_info.data_type
                }
                for field_name, field_info in schema.fields.items()
            },
            "confidence": schema.confidence,
            "notes": schema.notes,
            "analyzed_records": len(data_list)
        },
        description=f"스키마 분석 완료: 도메인={schema.domain}, 필드 {len(schema.fields)}개, 신뢰도={schema.confidence:.2f}"
    )
