import json
import logging
from datetime import datetime
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Body, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.schemas.response import APIResponseModel
from app.src.keyvalue_graph.clustering_normalizer import ClusteringNormalizer
from app.src.keyvalue_graph.data_loader import DataRecord
from app.src.keyvalue_graph.neptune_graph_builder import NeptuneGraphBuilder
from app.src.keyvalue_graph.llm_based_loader import LLMBasedKeyValueLoader
from app.src.keyvalue_graph.rdf_graph_builder import RdfGraphBuilder
from app.src.keyvalue_graph.relation_candidate_finder import RelationCandidateFinder
from app.src.keyvalue_graph.schema_analyzer import SchemaAnalyzer

router = APIRouter(prefix="/keyvalue", tags=["keyvalue-graph"])
logger = logging.getLogger(__name__)


# ===== 온디맨드 관계 워크플로우 =====


class RelationProposal(BaseModel):
    """관계 제안 모델"""

    proposal_id: str = Field(description="제안 ID (UUID)")
    source_doc_id: str = Field(description="출발 문서 ID")
    target_doc_id: str = Field(description="도착 문서 ID")
    relation_type: str = Field(description="관계 타입")
    confidence: float = Field(ge=0.0, le=1.0, description="신뢰도 (0~1)")
    reason: str = Field(description="관계 추론 근거")
    approved: bool = Field(default=False, description="승인 여부")


# ===== [TMP] 임시 승인 저장소 (테스트용) =====
# NOTE: 승인 워크플로우는 별도 시스템에서 개발 예정
# 현재 구현은 API 테스트 목적으로만 사용
# TODO: 실제 승인 시스템과 연동 시 이 부분 제거
pending_proposals: dict[str, RelationProposal] = {}


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

        # 원본 필드 매핑
        field_mapping = {
            original_key: field_info.korean_name
            for original_key, field_info in detected_schema.fields.items()
        }

        # extractable_metadata 추가 (합치기!)
        for meta_key, meta_info in detected_schema.extractable_metadata.items():
            field_mapping[meta_key] = meta_info.korean_name

        # 감지된 centrality_name 사용
        centrality_name = detected_schema.centrality_name

        logger.info(
            f"통합 필드 매핑: 원본 {len(detected_schema.fields)}개 + "
            f"메타데이터 {len(detected_schema.extractable_metadata)}개 = "
            f"총 {len(field_mapping)}개"
        )

    loader = LLMBasedKeyValueLoader(field_mapping=field_mapping)

    success_count = 0
    results = []

    for idx, data_item in enumerate(data_list):
        try:
            data_text = json.dumps(data_item, ensure_ascii=False)
            result = loader.extract_and_build_graph(
                data_text,
                centrality_name=centrality_name,
                source_file=file.filename or "batch_upload",
                record_index=idx,
                extractable_metadata=detected_schema.extractable_metadata if detected_schema else None
            )

            success_count += 1
            results.append({
                "index": idx,
                "document_id": result["document_id"],
                "success": True,
                "keys_count": result["keys_count"],
                "values_count": result["values_count"]
            })

        except Exception as e:
            # 에러 발생 시 즉시 중단하고 상세 정보 반환
            error_message = str(e)
            logger.error(f"배치 처리 중단: {idx+1}/{len(data_list)} 레코드에서 에러 발생 - {error_message}")

            error_detail = {
                "index": idx,
                "success": False,
                "error": error_message,
                "error_type": type(e).__name__,
                "centrality_name": centrality_name,
                "processed_count": success_count,
                "failed_at": idx,
                "total_count": len(data_list),
                "previous_results": results
            }

            return APIResponseModel(
                result=error_detail,
                description=f"배치 처리 실패: {idx+1}번째 레코드에서 {type(e).__name__} 에러 발생 (전체 {len(data_list)}개 중 {success_count}개 성공)",
                code=500
            )

    # 모든 레코드 성공 시 결과 반환
    if auto_detect_schema and detected_schema:
        description = (
            f"배치 처리 완료 [자동 스키마 감지]\n"
            f"- 도메인: {detected_schema.domain}\n"
            f"- Centrality: {centrality_name}\n"
            f"- 감지된 필드: {len(field_mapping)}개\n"
            f"- 처리 결과: {success_count}건 성공"
        )
        result_data = {
            "total": len(data_list),
            "success": success_count,
            "failed": 0,
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
        description = f"배치 처리 완료 (Centrality: {centrality_name}): {success_count}건 성공"
        result_data = {
            "total": len(data_list),
            "success": success_count,
            "failed": 0,
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
            "extractable_metadata": {
                meta_name: {
                    "korean_name": meta_info.korean_name,
                    "source_field": meta_info.source_field,
                    "data_type": meta_info.data_type,
                    "description": meta_info.description,
                    "examples": meta_info.examples
                }
                for meta_name, meta_info in schema.extractable_metadata.items()
            },
            "confidence": schema.confidence,
            "notes": schema.notes,
            "analyzed_records": len(data_list)
        },
        description=(
            f"스키마 분석 완료: 도메인={schema.domain}, "
            f"필드 {len(schema.fields)}개, "
            f"추출 가능 메타데이터 {len(schema.extractable_metadata)}개, "
            f"신뢰도={schema.confidence:.2f}"
        )
    )


# ===== RDF (JSON-LD) 엔드포인트 =====


@router.post(
    "/rdf/extract",
    tags=["keyvalue-graph"],
    summary="단건 Key-Value 추출 후 RDF(JSON-LD) 변환",
    response_model=APIResponseModel,
    response_class=JSONResponse,
)
async def extract_key_value_as_rdf(
    data: Annotated[str, Body(embed=True, description="원본 데이터 (JSON 문자열)")],
    centrality_name: Annotated[
        str, Body(embed=True, description="중심 노드 이름 (예: '병원현황', '판례정보')")
    ] = "병원현황",
    namespace: Annotated[str | None, Body(embed=True, description="RDF 네임스페이스 URI (미지정 시 config 기본값)")] = None,
    namespace_prefix: Annotated[
        str | None, Body(embed=True, description="RDF 네임스페이스 접두사 (미지정 시 config 기본값)")
    ] = None,
):
    """데이터에서 Key-Value를 추출하고 RDF(JSON-LD)로 변환."""
    raw_data = json.loads(data.strip())
    record = DataRecord(source_file="api_request", record_index=0, raw_data=raw_data)
    record.extract_key_values()

    rdf_kwargs = {k: v for k, v in {"namespace": namespace, "namespace_prefix": namespace_prefix}.items() if v is not None}
    builder = RdfGraphBuilder(**rdf_kwargs)
    result = builder.build_from_record(record, centrality_name=centrality_name)

    if result.success:
        return APIResponseModel(
            result={"document_id": result.document_id, "json_ld": result.json_ld},
            description=f"RDF(JSON-LD) 변환 완료 (Centrality: {centrality_name}, Document ID: {result.document_id})",
        )
    else:
        return APIResponseModel(
            result={"document_id": result.document_id, "error": result.message},
            description=f"RDF 변환 실패: {result.message}",
        )


@router.post(
    "/rdf/batch_extract",
    tags=["keyvalue-graph"],
    summary="배치 Key-Value 추출 후 RDF(JSON-LD) 변환",
    response_model=APIResponseModel,
    response_class=JSONResponse,
)
async def batch_extract_as_rdf(
    file: Annotated[UploadFile, File(description="JSON 배열 파일 [{...}, {...}, ...]")],
    centrality_name: Annotated[str, Body(description="중심 노드 이름")] = "병원현황",
    namespace: Annotated[str | None, Body(description="RDF 네임스페이스 URI (미지정 시 config 기본값)")] = None,
    namespace_prefix: Annotated[str | None, Body(description="RDF 네임스페이스 접두사 (미지정 시 config 기본값)")] = None,
):
    """JSON 배열 파일에서 Key-Value를 추출하고 RDF(JSON-LD)로 배치 변환."""

    content = await file.read()
    data_list = json.loads(content.decode("utf-8"))

    if not isinstance(data_list, list):
        data_list = [data_list]

    rdf_kwargs = {k: v for k, v in {"namespace": namespace, "namespace_prefix": namespace_prefix}.items() if v is not None}
    builder = RdfGraphBuilder(**rdf_kwargs)
    records = []
    for idx, raw_data in enumerate(data_list):
        record = DataRecord(
            source_file=file.filename or "batch_upload",
            record_index=idx,
            raw_data=raw_data,
        )
        record.extract_key_values()
        records.append(record)

    batch_result = builder.build_batch(records, centrality_name=centrality_name)

    return APIResponseModel(
        result=batch_result,
        description=(
            f"RDF 배치 변환 완료 (Centrality: {centrality_name}): "
            f"{batch_result['success']}건 성공, {batch_result['failed']}건 실패"
        ),
    )


# ===== 온디맨드 관계 워크플로우 엔드포인트 =====


@router.post(
    "/discover_relation_candidates",
    tags=["keyvalue-graph"],
    summary="관계 후보 Document 발굴 (추천)",
    response_model=APIResponseModel,
    response_class=JSONResponse,
)
async def discover_relation_candidates(
    source_doc_id: Annotated[str, Body(embed=True, description="기준 Document ID")],
    top_k: Annotated[int, Body(embed=True, description="반환할 후보 개수")] = 10,
):
    """단일 Document ID로 관계 후보 발굴 (온디맨드 방식)

    프로세스:
    1. Document의 Key-Value 메타데이터 분석
    2. LLM이 핵심 Key 선정 및 검색 전략 수립
    3. 그래프 구조 + 벡터 유사도로 후보 검색
    4. 상위 N개 후보 반환 (사용자 검토용)

    Args:
        source_doc_id: 기준 Document ID
        top_k: 반환할 후보 개수 (기본 10개)

    Returns:
        {
            "candidates": [
                {
                    "doc_id": "xxx_0002",
                    "score": 0.85,
                    "common_keys": ["지역", "병원규모"],
                    "common_values": {"지역": "서울", "병원규모": "대형"}
                },
                ...
            ],
            "search_strategy": "같은 지역의 대형병원 찾기"
        }
    """
    from app.repositories.graph_repository import GraphRepository
    from app.src.meta_extract.llm_response import LLMResponse

    graph_repo = GraphRepository()

    # 1. Document의 Key-Value 메타데이터 조회
    query = """
    MATCH (d:Document {id: $doc_id})-[:EXTRACTED_FROM]-(v:Value)-[:HAS_VALUE]-(k:Key)
    RETURN k.name as key, collect(DISTINCT v.name) as values
    """
    kv_results = graph_repo.execute_query(query, {"doc_id": source_doc_id})

    if not kv_results:
        return APIResponseModel(
            result={"error": "Document를 찾을 수 없습니다"},
            description=f"관계 후보 발굴 실패: Document ID '{source_doc_id}'를 찾을 수 없습니다",
        )

    # Key-Value 형식으로 변환
    key_values = {r["key"]: r["values"] for r in kv_results}

    # 2. LLM으로 핵심 Key 분석
    llm_response = LLMResponse("relation_key_analysis")

    # key_values를 JSON 포맷으로 변환 (LLM이 읽기 쉽게)
    key_values_json = json.dumps(key_values, ensure_ascii=False, indent=2)

    # 줄바꿈 포맷 (Langfuse가 key_values_formatted 사용할 경우 대비)
    key_values_formatted = "\n".join([f"- {k}: {v}" for k, v in key_values.items()])

    analysis_result = llm_response.generate_llm_response(
        variables={
            "document_id": source_doc_id,
            "key_values": key_values,  # 원본 (fallback)
            "key_values_json": key_values_json,  # JSON 포맷
            "key_values_formatted": key_values_formatted,  # 줄바꿈 포맷
        }
    )

    important_keys = [ki["key_name"] for ki in analysis_result["important_keys"]]
    search_strategy = analysis_result["search_strategy"]

    # important_values: LLM이 선정한 중요 Key의 실제 값
    important_values = {k: key_values.get(k, []) for k in important_keys}

    logger.info(f"LLM 분석 결과: 중요 Key={important_keys}, 전략={search_strategy}")

    # 3. 후보 Document 검색
    finder = RelationCandidateFinder(graph_repo=graph_repo)
    candidates = finder.find_candidates(
        source_doc_id=source_doc_id,
        important_keys=important_keys,
        important_values=important_values,
        top_k=top_k,
    )

    if not candidates:
        return APIResponseModel(
            result={
                "source_doc_id": source_doc_id,
                "source_doc_info": {
                    "key_values": key_values,
                    "important_keys": important_keys,
                    "important_values": important_values,
                },
                "candidates": [],
                "search_strategy": search_strategy,
                "important_keys": important_keys,
            },
            description=f"관계 후보를 찾지 못했습니다 (중요 Key: {important_keys})",
        )

    return APIResponseModel(
        result={
            "source_doc_id": source_doc_id,
            "source_doc_info": {
                "key_values": key_values,
                "important_keys": important_keys,
                "important_values": important_values,
            },
            "candidates": [c.to_dict() for c in candidates],
            "search_strategy": search_strategy,
            "important_keys": important_keys,
        },
        description=f"관계 후보 {len(candidates)}개 발견 (전략: {search_strategy})",
    )


@router.post(
    "/propose_relation",
    tags=["keyvalue-graph"],
    summary="[Legacy] 두 데이터 간 관계 제안",
    response_model=APIResponseModel,
    response_class=JSONResponse,
)
async def propose_relation(
    source_doc_id: Annotated[str, Body(embed=True, description="출발 문서 ID")],
    target_doc_id: Annotated[str, Body(embed=True, description="도착 문서 ID")],
):
    """두 Document 간 관계 분석 및 제안

    사용자가 특정 데이터 ID를 입력하면 해당 쌍에 대해서만 관계를 분석하고 제안합니다.

    Args:
        source_doc_id: 출발 문서 ID (예: "data_0001")
        target_doc_id: 도착 문서 ID (예: "data_0002")

    Returns:
        RelationProposal (사용자 검토용)
    """
    from app.repositories.graph_repository import GraphRepository

    # 1. 두 문서 조회
    graph_repo = GraphRepository()
    query = """
    MATCH (d:Document {id: $doc_id})
    RETURN d.id as id, d.raw_text as raw_text
    """

    source_result = graph_repo.execute_query(query, {"doc_id": source_doc_id})
    target_result = graph_repo.execute_query(query, {"doc_id": target_doc_id})

    if not source_result or not target_result:
        return APIResponseModel(
            result={"error": "문서를 찾을 수 없습니다"},
            description=f"관계 제안 실패: 문서 ID '{source_doc_id}' 또는 '{target_doc_id}'를 찾을 수 없습니다",
        )

    source_text = source_result[0]["raw_text"]
    target_text = target_result[0]["raw_text"]

    # 2. LLM으로 관계 분석 (간단한 구현 - 실제로는 더 정교한 프롬프트 필요)
    # TODO: 실제로는 두 문서의 메타데이터, 토픽, 엔티티 등을 비교하여 관계 추론

    # 임시: 간단한 텍스트 유사도 기반 관계 제안
    proposal_id = str(uuid4())

    # 기본 관계 제안 (실제로는 LLM 기반 분석 필요)
    confidence = 0.7
    reason = f"문서 '{source_doc_id}'와 '{target_doc_id}' 간 관계 분석 필요 (LLM 분석 미구현)"

    proposal = RelationProposal(
        proposal_id=proposal_id,
        source_doc_id=source_doc_id,
        target_doc_id=target_doc_id,
        relation_type="RELATED_TO",
        confidence=confidence,
        reason=reason,
        approved=False,
    )

    # 4. 임시 저장소에 저장
    pending_proposals[proposal_id] = proposal

    logger.info(f"관계 제안 생성: {source_doc_id} -> {target_doc_id} (ID: {proposal_id})")

    return APIResponseModel(
        result=proposal.model_dump(),
        description=f"관계 제안 완료 (ID: {proposal_id}): RELATED_TO",
    )


@router.post(
    "/approve_relation",
    tags=["keyvalue-graph"],
    summary="[TMP] 관계 승인 및 저장 (테스트용)",
    response_model=APIResponseModel,
    response_class=JSONResponse,
)
async def approve_relation(
    proposal_id: Annotated[str, Body(embed=True, description="제안 ID")],
    approved: Annotated[bool, Body(embed=True, description="승인 여부")],
):
    """[TMP] 관계 제안 승인 및 그래프DB 저장 (테스트용)

    NOTE: 승인 워크플로우는 별도 시스템에서 개발 예정
    현재 구현은 온디맨드 관계 저장 기능을 테스트하기 위한 임시 엔드포인트

    Args:
        proposal_id: 제안 ID (propose_relation에서 반환)
        approved: 승인 여부 (True: 저장, False: 거부)

    Returns:
        저장 결과

    TODO: 실제 승인 시스템과 연동 시 이 엔드포인트 제거 또는 통합
    """
    # 1. 임시 저장소에서 제안 조회
    if proposal_id not in pending_proposals:
        return APIResponseModel(
            result={"error": "제안을 찾을 수 없습니다"},
            description=f"승인 실패: 제안 ID '{proposal_id}'를 찾을 수 없습니다",
        )

    proposal = pending_proposals[proposal_id]

    # 2. 승인 여부 확인
    if not approved:
        # 거부: 임시 저장소에서 제거
        del pending_proposals[proposal_id]
        logger.info(f"관계 제안 거부: {proposal.source_doc_id} -> {proposal.target_doc_id}")
        return APIResponseModel(
            result={"approved": False, "proposal_id": proposal_id},
            description="관계 제안이 거부되었습니다",
        )

    # 3. 승인: 그래프DB에 저장
    proposal.approved = True
    # TODO: NeptuneGraphBuilder에 create_relation 메서드 추가 필요
    # graph_builder = NeptuneGraphBuilder()
    # result = graph_builder.create_relation(...)

    # 임시: 승인만 처리
    result = {"success": True, "relation_id": f"{proposal.source_doc_id}->{proposal.target_doc_id}"}

    if result["success"]:
        # 임시 저장소에서 제거
        del pending_proposals[proposal_id]
        logger.info(f"관계 저장 완료: {proposal.source_doc_id} -> {proposal.target_doc_id}")
        return APIResponseModel(
            result=result,
            description=f"관계 저장 완료 (ID: {result['relation_id']}): RELATED_TO",
        )
    else:
        return APIResponseModel(
            result=result,
            description=f"관계 저장 실패: {result.get('error', 'Unknown error')}",
        )


@router.get(
    "/pending_proposals",
    tags=["keyvalue-graph"],
    summary="[TMP] 승인 대기 중인 관계 목록 (테스트용)",
    response_model=APIResponseModel,
    response_class=JSONResponse,
)
async def get_pending_proposals():
    """[TMP] 승인 대기 중인 관계 제안 목록 조회 (테스트용)

    NOTE: 승인 워크플로우는 별도 시스템에서 개발 예정
    현재 구현은 테스트 목적으로만 사용

    Returns:
        RelationProposal 리스트

    TODO: 실제 승인 시스템과 연동 시 이 엔드포인트 제거
    """
    proposals_list = [p.model_dump() for p in pending_proposals.values()]

    return APIResponseModel(
        result={"proposals": proposals_list, "count": len(proposals_list)},
        description=f"승인 대기 중인 관계 {len(proposals_list)}건",
    )
