import asyncio
import csv
import io
import json
import uuid
from datetime import datetime, timezone
from typing import Annotated

import logging
from fastapi import APIRouter, Body, UploadFile, File
from fastapi.responses import JSONResponse

from app.schemas.dataset import (
    DatasetIngestRequest,
    DatasetRelation,
    DatasetSearchRequest,
    IngestFromCatalogRequest,
    RelationDecisionRequest,
    RunRelationPipelineRequest,
)
from app.schemas.response import APIResponseModel
from app.src.dataset.dataset_service import dataset_service
from app.src.dataset.relation_pipeline import relation_pipeline

router = APIRouter(prefix="/dataset", tags=["dataset"])
logger = logging.getLogger(__name__)

# job 상태는 SQLite(job_store)에 영속화 — 재시작에도 폴링 API가 상태 유지 (DAG 자동 트리거 전제)
from app.src.dataset.job_store import get_job_store


def _run_kv_build_sync(job_id: str, table_name: str) -> None:
    """등록된 데이터셋의 원본 파일을 재처리하여 Key-Value 그래프를 생성 (백그라운드).

    재업로드 없이, 등록 시 보관한 원본 파일(raw_storage)을 읽어 스키마 분석 → KV 추출 →
    그래프 적재 → 정규화를 수행한다.
    """
    from app.repositories.dataset_repository import get_dataset_repository
    from app.src.dataset.raw_storage import get_raw_storage
    from app.src.keyvalue_graph.clustering_normalizer import ClusteringNormalizer
    from app.src.keyvalue_graph.llm_based_loader import LLMBasedKeyValueLoader
    from app.src.keyvalue_graph.schema_analyzer import SchemaAnalyzer

    try:
        profile = get_dataset_repository().get(table_name)
        if profile is None:
            raise ValueError(f"데이터셋 '{table_name}' 프로필을 찾을 수 없습니다")
        if not profile.raw_storage_key:
            raise ValueError(
                f"데이터셋 '{table_name}'에 보관된 원본 파일이 없습니다. "
                f"파일 업로드(/dataset/ingest)로 등록된 데이터셋만 재처리할 수 있습니다."
            )

        raw = get_raw_storage().load(profile.raw_storage_key)
        if raw is None:
            raise ValueError(f"원본 파일을 읽을 수 없습니다: {profile.raw_storage_key}")

        records = _parse_upload(raw, profile.raw_storage_key)
        if not records:
            raise ValueError("원본 파일에 레코드가 없습니다")

        # 스키마 자동 분석 → 한글 필드 매핑
        analyzer = SchemaAnalyzer(sample_size=5)
        detected = analyzer.analyze(records)
        field_mapping = {
            orig: info.korean_name for orig, info in detected.fields.items()
        }
        for meta_key, meta_info in detected.extractable_metadata.items():
            field_mapping[meta_key] = meta_info.korean_name
        centrality_name = detected.centrality_name

        loader = LLMBasedKeyValueLoader(field_mapping=field_mapping)
        success = 0
        for idx, rec in enumerate(records):
            try:
                loader.extract_and_build_graph(
                    json.dumps(rec, ensure_ascii=False),
                    centrality_name=centrality_name,
                    domain=detected.domain,
                    source_file=profile.raw_storage_key,
                    record_index=idx,
                    extractable_metadata=detected.extractable_metadata,
                    table_name=table_name,
                )
                success += 1
            except Exception as e:
                logger.warning(f"[KVBuild] {table_name} 레코드 {idx} 처리 실패: {e}")

        # Value 정규화
        normalized = ClusteringNormalizer().normalize_all_keys()

        get_job_store().update(job_id,
            status="done",
            finished_at=datetime.now(timezone.utc).isoformat(),
            records=len(records),
            documents_built=success,
            normalized=normalized.get("normalized_count", 0),
        )
    except Exception as e:
        logger.error(f"[KVBuild] job={job_id} 실패: {e}")
        get_job_store().update(job_id,
            status="error",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=str(e),
        )


def _parse_upload(content: bytes, filename: str) -> list[dict]:
    """JSON / JSONL / CSV 파일을 레코드 목록으로 파싱."""
    name = filename.lower()
    if name.endswith(".csv"):
        text = content.decode("utf-8-sig")
        return list(csv.DictReader(io.StringIO(text)))
    if name.endswith(".jsonl"):
        return [
            json.loads(line)
            for line in content.decode("utf-8").splitlines()
            if line.strip()
        ]
    records = json.loads(content.decode("utf-8"))
    return records if isinstance(records, list) else [records]


def _run_pipeline_sync(job_id: str, request: RunRelationPipelineRequest) -> None:
    """스레드풀에서 실행되는 동기 파이프라인."""
    try:
        results = relation_pipeline.run(
            table_names=request.table_names,
            tier2_threshold=request.tier2_threshold,
            tier3_confidence_threshold=request.tier3_confidence_threshold,
            refresh_table_names=request.refresh_table_names,
        )
        get_job_store().update(job_id,
            status="done",
            finished_at=datetime.now(timezone.utc).isoformat(),
            results=[r.model_dump() for r in results],
            total_saved=sum(len(r.saved_relations) for r in results),
            total_pending=sum(len(r.pending_relations) for r in results),
        )
    except Exception as e:
        logger.error(f"[Pipeline] job={job_id} 실패: {e}")
        get_job_store().update(job_id,
            status="error",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=str(e),
        )


@router.post(
    "/ingest",
    response_model=APIResponseModel,
    response_class=JSONResponse,
    summary="테이블 메타데이터 추출 및 Neptune/OpenSearch 적재",
)
async def ingest_dataset(
        file: Annotated[UploadFile, File(description="JSON / JSONL / CSV 파일")],
        table_name: Annotated[str, Body(embed=True, description="테이블명")],
        domain: Annotated[str, Body(embed=True, description="도메인 (medical, broadcast 등). 생략 시 특징 기반 자동 분류")] = "",
        description: Annotated[str, Body(embed=True, description="짧은 설명")] = "",
        table_summary: Annotated[str, Body(embed=True, description="상세 요약 (비어있으면 description 사용)")] = "",
        catalog_identifier: Annotated[str, Body(embed=True, description="통합 카탈로그 식별자 (catalog_entry.identifier)")] = "",
):
    """JSON / JSONL / CSV 파일을 업로드하면 메타데이터를 추출하여 Neptune Dataset 노드와 OpenSearch에 저장."""
    content = await file.read()
    records = _parse_upload(content, file.filename or "upload.json")

    # 원본 파일 보존 (§3.2, 백엔드는 RAW_STORAGE_BACKEND) — 실패해도 적재는 계속.
    # 저장 키는 프로필에 기록해 두어, 재업로드 없이 KV 그래프 처리를 트리거할 수 있게 한다.
    raw_storage_key = ""
    try:
        from app.src.dataset.raw_storage import get_raw_storage
        storage = get_raw_storage()
        if storage.enabled:
            raw_storage_key = storage.store_raw_dataset(table_name, file.filename or "upload.json", content) or ""
            logger.info(f"[RawStorage] 원본 저장: {raw_storage_key}")
    except Exception as e:
        logger.warning(f"[RawStorage] 원본 저장 실패 (적재는 계속): {e}")

    profile = dataset_service.ingest_table(
        records=records,
        table_name=table_name,
        domain=domain,
        description=description,
        table_summary=table_summary,
        raw_storage_key=raw_storage_key,
        catalog_identifier=catalog_identifier,
    )

    return APIResponseModel(
        result=profile.model_dump(exclude={"table_embedding"}),
        description=(
            f"Dataset '{table_name}' 등록 완료: "
            f"{profile.record_count}건, {profile.column_count}개 컬럼"
        ),
    )


@router.post(
    "/ingest_json",
    response_model=APIResponseModel,
    response_class=JSONResponse,
    summary="JSON 바디로 테이블 메타데이터 적재",
)
async def ingest_dataset_json(
        request: DatasetIngestRequest,
        records: Annotated[list[dict], Body(embed=True, description="레코드 목록")],
):
    """JSON 바디로 레코드와 메타데이터를 함께 전달하여 적재."""
    profile = dataset_service.ingest_table(
        records=records,
        table_name=request.table_name,
        domain=request.domain,
        description=request.description,
        table_summary=request.table_summary,
        catalog_identifier=request.catalog_identifier,
    )

    return APIResponseModel(
        result=profile.model_dump(exclude={"table_embedding"}),
        description=f"Dataset '{request.table_name}' 등록 완료",
    )


@router.post(
    "/ingest_from_catalog",
    response_model=APIResponseModel,
    response_class=JSONResponse,
    summary="① 통합 카탈로그 엔트리를 pull 하여 데이터셋 등록",
)
async def ingest_dataset_from_catalog(request: IngestFromCatalogRequest):
    """catalog-service에서 엔트리 메타데이터를 가져와 재업로드 없이 데이터셋으로 등록.

    - `catalog_entry_id`(숫자 PK) 또는 `catalog_identifier`(DCAT identifier) 중 하나 필수
    - 매핑: identifier→catalog_identifier, title→table_name/description,
      description→table_summary, keyword+theme→keywords(시드), theme[0]→domain
    - `records`를 함께 주면 컬럼 프로파일링 포함, 없으면 메타데이터-only 등록
    """
    from app.clients.catalog_client import get_catalog_client, ingest_args_from_entry

    if not request.catalog_entry_id and not request.catalog_identifier:
        return JSONResponse(
            status_code=422,
            content={"detail": "catalog_entry_id 또는 catalog_identifier 중 하나는 필수입니다"},
        )

    client = get_catalog_client()
    entry = (
        client.get_entry(request.catalog_entry_id)
        if request.catalog_entry_id
        else client.find_entry_by_identifier(request.catalog_identifier)
    )

    args = ingest_args_from_entry(
        entry,
        table_name=request.table_name or None,
        domain=request.domain or None,
    )
    profile = dataset_service.ingest_table(records=request.records or [], **args)

    return APIResponseModel(
        result=profile.model_dump(exclude={"table_embedding", "keywords_embedding"}),
        description=(
            f"카탈로그 엔트리(identifier={args['catalog_identifier']}) → "
            f"Dataset '{profile.table_name}' 등록 완료"
            + ("" if request.records else " (메타데이터-only)")
        ),
    )


@router.post(
    "/relation",
    response_model=APIResponseModel,
    response_class=JSONResponse,
    summary="테이블 간 관계 Neptune 저장",
)
async def save_relation(relation: DatasetRelation):
    """두 테이블 간 관계를 Neptune RDF Reification으로 저장.

    Args:
        relation: DatasetRelation (table_a, table_b, score, method, relation_type)

    method 값:
        - rule: FK 또는 동일 컬럼명 자동 확정 (Tier 1)
        - embedding: KNN 유사도 기반 (Tier 2)
        - llm: LLM 검증 통과 (Tier 3)
    """
    dataset_service.save_relation(relation)

    return APIResponseModel(
        result=relation.model_dump(),
        description=(
            f"관계 저장 완료: {relation.table_a} --[{relation.relation_type}]--> {relation.table_b} "
            f"(score={relation.score}, method={relation.method})"
        ),
    )


@router.post(
    "/relation/decision",
    response_model=APIResponseModel,
    response_class=JSONResponse,
    summary="① 승인 워크플로우 결정 콜백 (내부 API)",
)
async def apply_relation_decision(request: RelationDecisionRequest):
    """①의 데이터셋 관계 승인/반려 결정을 그래프에 반영합니다.

    승인 SSOT는 ①(dataset_relation.status)입니다. 이 엔드포인트는 ①의 PATCH
    /relation/datasets/{id} 승인 처리에서 호출되는 내부 콜백으로,
    승인된 관계를 Neptune 지식그래프에 저장합니다 (재환류 없음). 반려는 no-op.
    """
    relation = dataset_service.apply_relation_decision(
        source_identifier=request.source_identifier,
        target_identifier=request.target_identifier,
        relation_type=request.relation_type,
        score=request.score,
        method=request.method,
        evidence=request.evidence,
        approved=request.approved,
    )
    if relation is None:
        return APIResponseModel(
            result={"applied": False},
            description="반려 결정 수신 — 그래프 미반영 (no-op)",
        )
    return APIResponseModel(
        result={"applied": True, "relation": relation.model_dump()},
        description=(
            f"승인 관계 그래프 반영 완료: {relation.table_a} "
            f"--[{relation.relation_type}]--> {relation.table_b}"
        ),
    )


@router.post(
    "/search",
    response_model=APIResponseModel,
    response_class=JSONResponse,
    summary="쿼리로 유사 테이블 검색 (KNN)",
)
async def search_datasets(request: DatasetSearchRequest):
    """쿼리 텍스트 임베딩으로 유사 테이블 KNN 검색.

    Args:
        request: DatasetSearchRequest (query, domain, top_k)

    Returns:
        유사 테이블 목록 (score 내림차순)
    """
    results = dataset_service.search_datasets(
        query=request.query,
        domain=request.domain,
        top_k=request.top_k,
    )

    return APIResponseModel(
        result=[r.model_dump() for r in results],
        description=f"'{request.query}' 검색 결과: {len(results)}개",
    )


@router.post(
    "/nl_search",
    response_model=APIResponseModel,
    response_class=JSONResponse,
    summary="자연어 검색 + 그래프 시각화용 부분 그래프 반환 (§9)",
)
async def nl_search(
        query: Annotated[str, Body(embed=True, description="자연어 질의")],
        top_k: Annotated[int, Body(embed=True, description="상위 데이터셋 수")] = 5,
):
    """자연어 질의를 검색 의도로 변환하여 데이터셋 KNN 검색을 수행하고,

    상위 노드의 1-hop 이웃(Level 1 관계)을 묶은 부분 그래프를 함께 반환한다.
    """
    result = dataset_service.nl_search(query=query, top_k=top_k)
    return APIResponseModel(
        result=result,
        description=(
            f"'{query}' → 의도='{result['intent']}', "
            f"데이터셋 {len(result['datasets'])}개, "
            f"노드 {len(result['graph']['nodes'])}개 / 엣지 {len(result['graph']['edges'])}개"
        ),
    )


@router.post(
    "/{table_name}/build-graph",
    response_model=APIResponseModel,
    response_class=JSONResponse,
    summary="등록된 데이터셋의 KV 그래프 생성 (재업로드 없이)",
)
async def build_dataset_graph(table_name: str):
    """등록 시 보관한 원본 파일로 Key-Value 그래프 생성을 백그라운드 실행하고 job_id 반환.

    상태는 GET /dataset/build-graph/{job_id} 로 폴링.
    """
    job_id = str(uuid.uuid4())
    get_job_store().put(job_id, "kv_build", {
        "status": "running",
        "table_name": table_name,
        "started_at": datetime.now(timezone.utc).isoformat(),
    })
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_kv_build_sync, job_id, table_name)
    return APIResponseModel(
        result={"job_id": job_id, "status": "running", "table_name": table_name},
        description=f"'{table_name}' KV 그래프 생성 시작",
    )


@router.get(
    "/build-graph/{job_id}",
    response_model=APIResponseModel,
    response_class=JSONResponse,
    summary="KV 그래프 생성 작업 상태 조회",
)
async def get_build_graph_status(job_id: str):
    """job_id 의 KV 그래프 생성 상태를 반환. status: running | done | error"""
    job = get_job_store().get(job_id)
    if not job:
        return APIResponseModel(result=None, description="job을 찾을 수 없습니다")
    return APIResponseModel(result=job, description=f"status={job['status']}")


@router.post(
    "/subgraph",
    response_model=APIResponseModel,
    response_class=JSONResponse,
    summary="데이터셋 1-hop 부분 그래프 조회 (노드 확장용, §9)",
)
async def get_subgraph(
        table_names: Annotated[list[str], Body(embed=True, description="중심 데이터셋명 목록")],
):
    """주어진 데이터셋들과 Level 1으로 연결된 1-hop 이웃을 부분 그래프로 반환."""
    graph = dataset_service.get_dataset_subgraph(table_names)
    return APIResponseModel(
        result=graph,
        description=f"노드 {len(graph['nodes'])}개 / 엣지 {len(graph['edges'])}개",
    )


@router.get(
    "/list",
    response_model=APIResponseModel,
    response_class=JSONResponse,
    summary="등록된 Dataset 목록 조회",
)
async def list_datasets():
    """OpenSearch에 등록된 모든 Dataset 목록을 반환.

    각 데이터셋에 KV 그래프 처리 상태(kv_document_count, kv_processed)를 함께 반환하여
    '데이터셋 관리' 화면에서 처리 여부 배지/트리거를 표시할 수 있게 한다.
    """
    from app.repositories.dataset_repository import get_dataset_repository
    from app.repositories.opensearch_repository import get_opensearch_repository
    repo = get_dataset_repository()
    os_repo = get_opensearch_repository()
    profiles = repo.list_all()

    items = []
    for p in profiles:
        doc = p.model_dump(exclude={"table_embedding", "keywords_embedding"})
        kv_count = os_repo.count_documents_by_table(p.table_name)
        doc["kv_document_count"] = kv_count
        doc["kv_processed"] = kv_count > 0
        doc["raw_available"] = bool(p.raw_storage_key)
        items.append(doc)

    return APIResponseModel(
        result=items,
        description=f"Dataset {len(profiles)}개",
    )


@router.post(
    "/relation/pipeline",
    response_model=APIResponseModel,
    response_class=JSONResponse,
    summary="3-Tier 관계 생성 파이프라인 비동기 실행",
)
async def run_relation_pipeline(request: RunRelationPipelineRequest):
    """파이프라인을 백그라운드로 시작하고 job_id를 즉시 반환.

    결과는 GET /dataset/relation/pipeline/{job_id} 로 폴링.
    """
    job_id = str(uuid.uuid4())
    get_job_store().put(job_id, "relation_pipeline", {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "table_names": request.table_names,
        "refresh_table_names": request.refresh_table_names,
    })
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_pipeline_sync, job_id, request)

    return APIResponseModel(
        result={"job_id": job_id, "status": "running"},
        description=(
            f"파이프라인 시작: "
            + (f"{len(request.table_names)}개 테이블" if request.table_names else "전체 테이블")
            + (f" (재평가 {len(request.refresh_table_names)}개)" if request.refresh_table_names else "")
        ),
    )


@router.get(
    "/relation/pipeline/{job_id}",
    response_model=APIResponseModel,
    response_class=JSONResponse,
    summary="파이프라인 실행 상태 조회",
)
async def get_pipeline_status(job_id: str):
    """job_id 의 파이프라인 상태를 반환. status: running | done | error"""
    job = get_job_store().get(job_id)
    if not job:
        return APIResponseModel(result=None, description="job을 찾을 수 없습니다")
    return APIResponseModel(result=job, description=f"status={job['status']}")


@router.post(
    "/sync_from_catalog",
    response_model=APIResponseModel,
    response_class=JSONResponse,
    summary="① 카탈로그 갱신분 동기화 + 관계 파이프라인 증분 실행 (액티브 루프 1회)",
)
async def sync_from_catalog(
        date_from: Annotated[str | None, Body(embed=True, description="조회 시작일(YYYY-MM-DD) 재정의 — 백필용. 생략 시 watermark")] = None,
):
    """카탈로그 변경 감지 → 데이터셋 등록/재평가 표시 → 파이프라인 증분 실행을 백그라운드로 시작.

    상태는 GET /dataset/sync_from_catalog/{job_id} 로 폴링.
    CATALOG_SYNC_ENABLED=True 면 동일 사이클이 주기적으로 자동 실행된다.
    """
    from app.src.dataset.catalog_sync import get_catalog_sync_service

    job_id = str(uuid.uuid4())
    get_job_store().put(job_id, "catalog_sync", {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "date_from": date_from,
    })

    def _run_sync(job_id: str, date_from: str | None) -> None:
        try:
            summary = get_catalog_sync_service().run_cycle(date_from=date_from)
            get_job_store().update(job_id, status="done",
                                   finished_at=datetime.now(timezone.utc).isoformat(), **summary)
        except Exception as e:
            logger.error(f"[CatalogSync] job={job_id} 실패: {e}")
            get_job_store().update(job_id, status="error",
                                   finished_at=datetime.now(timezone.utc).isoformat(), error=str(e))

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_sync, job_id, date_from)
    return APIResponseModel(
        result={"job_id": job_id, "status": "running"},
        description="카탈로그 동기화 사이클 시작",
    )


@router.get(
    "/sync_from_catalog/{job_id}",
    response_model=APIResponseModel,
    response_class=JSONResponse,
    summary="카탈로그 동기화 작업 상태 조회",
)
async def get_sync_from_catalog_status(job_id: str):
    """job_id 의 동기화 사이클 상태를 반환. status: running | done | error"""
    job = get_job_store().get(job_id)
    if not job:
        return APIResponseModel(result=None, description="job을 찾을 수 없습니다")
    return APIResponseModel(result=job, description=f"status={job['status']}")


@router.get(
    "/sync-status",
    response_model=APIResponseModel,
    response_class=JSONResponse,
    summary="OpenSearch ↔ Neptune Dataset 노드 동기화 상태 확인",
)
async def get_sync_status():
    """OpenSearch에 등록된 테이블 목록 vs Neptune Dataset 노드를 비교해 동기화 상태를 반환."""
    from app.repositories.dataset_repository import get_dataset_repository
    from app.clients.neptune import get_neptune_client
    from app.config import settings

    repo = get_dataset_repository()
    profiles = repo.list_all()
    os_names = {p.table_name for p in profiles}

    neptune = get_neptune_client()
    dm = settings.DATASET_RDF_NAMESPACE
    query = f"""
    PREFIX dm: <{dm}>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    SELECT ?tableName WHERE {{
        ?s rdf:type dm:Dataset ;
           dm:tableName ?tableName .
    }}
    """
    try:
        resp = neptune.execute_query(query)
        neptune_names = {
            b["tableName"]["value"]
            for b in resp.get("results", {}).get("bindings", [])
        }
    except Exception:
        neptune_names = set()

    items = [
        {
            "table_name": name,
            "in_opensearch": name in os_names,
            "in_neptune": name in neptune_names,
            "synced": name in os_names and name in neptune_names,
        }
        for name in os_names | neptune_names
    ]
    items.sort(key=lambda x: x["table_name"])

    return APIResponseModel(
        result=items,
        description=f"총 {len(items)}개, 동기화={sum(1 for i in items if i['synced'])}개",
    )


@router.get(
    "/relations",
    response_model=APIResponseModel,
    response_class=JSONResponse,
    summary="Neptune에 저장된 관계 목록 조회",
)
async def list_relations():
    """Neptune에 저장된 모든 Dataset 간 관계를 반환."""
    from app.clients.neptune import get_neptune_client
    from app.config import settings

    neptune = get_neptune_client()
    dm = settings.DATASET_RDF_NAMESPACE
    query = f"""
    PREFIX dm: <{dm}>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    SELECT ?tableA ?tableB ?catalogIdA ?catalogIdB ?score ?method ?relationType ?createdAt WHERE {{
        ?rel rdf:type rdf:Statement ;
             rdf:subject ?s ;
             rdf:predicate dm:relatedTo ;
             rdf:object ?o ;
             dm:relationScore ?score ;
             dm:relationMethod ?method ;
             dm:relationType ?relationType ;
             dm:createdAt ?createdAt .
        ?s dm:tableName ?tableA .
        ?o dm:tableName ?tableB .
        OPTIONAL {{ ?s dm:catalogIdentifier ?catalogIdA }}
        OPTIONAL {{ ?o dm:catalogIdentifier ?catalogIdB }}
    }}
    ORDER BY DESC(?createdAt)
    """
    try:
        resp = neptune.execute_query(query)
        bindings = resp.get("results", {}).get("bindings", [])
        relations = [
            {
                "table_a": b["tableA"]["value"],
                "table_b": b["tableB"]["value"],
                "catalog_identifier_a": b.get("catalogIdA", {}).get("value", ""),
                "catalog_identifier_b": b.get("catalogIdB", {}).get("value", ""),
                "score": float(b["score"]["value"]),
                "method": b["method"]["value"],
                "relation_type": b["relationType"]["value"],
                "created_at": b["createdAt"]["value"],
            }
            for b in bindings
        ]
    except Exception:
        relations = []

    return APIResponseModel(
        result=relations,
        description=f"관계 {len(relations)}개",
    )


@router.get(
    "/stats",
    response_model=APIResponseModel,
    response_class=JSONResponse,
    summary="데이터셋 현황 요약",
)
async def get_stats():
    """데이터셋 수, 관계 수, Tier별 분포 등 현황 요약을 반환."""
    from app.clients.neptune import get_neptune_client
    from app.config import settings
    from app.repositories.dataset_repository import get_dataset_repository

    repo = get_dataset_repository()
    profiles = repo.list_all()
    dataset_count = len(profiles)
    domain_counts: dict[str, int] = {}
    for p in profiles:
        domain_counts[p.domain] = domain_counts.get(p.domain, 0) + 1

    neptune = get_neptune_client()
    dm = settings.DATASET_RDF_NAMESPACE
    query = f"""
    PREFIX dm: <{dm}>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    SELECT ?method (COUNT(?rel) AS ?cnt) WHERE {{
        ?rel rdf:type rdf:Statement ;
             rdf:predicate dm:relatedTo ;
             dm:relationMethod ?method .
    }}
    GROUP BY ?method
    """
    try:
        resp = neptune.execute_query(query)
        method_counts = {
            b["method"]["value"]: int(b["cnt"]["value"])
            for b in resp.get("results", {}).get("bindings", [])
        }
    except Exception:
        method_counts = {}

    total_relations = sum(method_counts.values())

    return APIResponseModel(
        result={
            "dataset_count": dataset_count,
            "relation_count": total_relations,
            "domain_counts": domain_counts,
            "method_counts": method_counts,
        },
        description=f"데이터셋 {dataset_count}개, 관계 {total_relations}개",
    )


@router.delete(
    "/{table_name}",
    response_model=APIResponseModel,
    response_class=JSONResponse,
    summary="Dataset 삭제",
)
async def delete_dataset(table_name: str):
    """OpenSearch + Neptune에서 Dataset 및 연관 관계를 삭제."""
    from app.src.dataset.dataset_service import DatasetService
    ok = DatasetService().delete_table(table_name)
    return APIResponseModel(
        result={"table_name": table_name, "deleted": ok},
        description=f"Dataset '{table_name}' {'삭제 완료' if ok else '삭제 실패'}",
    )
