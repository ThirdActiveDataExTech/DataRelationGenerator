"""Dataset OpenSearch 저장소."""

import logging
from typing import Any

from app.clients.opensearch import OpenSearchClient, get_opensearch_client
from app.config import settings
from app.schemas.dataset import DatasetProfile, DatasetSearchResult

logger = logging.getLogger(__name__)

_DATASETS_INDEX = settings.OPENSEARCH_DATASETS_INDEX


class DatasetRepository:
    """OpenSearch datasets 인덱스 저장소."""

    def __init__(self, client: OpenSearchClient | None = None):
        self.client = client or get_opensearch_client()

    def ensure_index(self, embedding_dim: int) -> None:
        """datasets 인덱스가 없으면 KNN 매핑으로 생성. 있으면 신규 필드 매핑만 보강."""
        if self.client.client.indices.exists(index=_DATASETS_INDEX):
            # 기존 인덱스에 이후 추가된 필드 매핑 보강 (idempotent)
            self.client.client.indices.put_mapping(
                index=_DATASETS_INDEX,
                body={"properties": {
                    "catalog_identifier": {"type": "keyword"},
                    "display_name": {"type": "keyword"},
                    "party": {"type": "keyword"},
                    "provider": {"type": "keyword"},
                    "landing_page": {"type": "keyword"},
                }},
            )
            return

        mapping: dict[str, Any] = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100,
                }
            },
            "mappings": {
                "properties": {
                    "dataset_id": {"type": "keyword"},
                    "table_name": {"type": "keyword"},
                    "display_name": {"type": "keyword"},
                    "catalog_identifier": {"type": "keyword"},
                    "party": {"type": "keyword"},
                    "provider": {"type": "keyword"},
                    "landing_page": {"type": "keyword"},
                    "domain": {"type": "keyword"},
                    "record_count": {"type": "integer"},
                    "column_count": {"type": "integer"},
                    "column_names": {"type": "keyword"},
                    "description": {"type": "text"},
                    "table_summary": {"type": "text"},
                    "table_embedding": {
                        "type": "knn_vector",
                        "dimension": embedding_dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                        },
                    },
                    "keywords": {"type": "keyword"},
                    "keywords_embedding": {
                        "type": "knn_vector",
                        "dimension": embedding_dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                        },
                    },
                    "created_at": {"type": "date"},
                }
            },
        }
        self.client.client.indices.create(index=_DATASETS_INDEX, body=mapping)
        logger.info(f"[OpenSearch] '{_DATASETS_INDEX}' 인덱스 생성 (dim={embedding_dim})")

    def upsert(self, profile: DatasetProfile) -> None:
        """테이블 메타데이터 문서를 upsert (dataset_id를 doc ID로 사용)."""
        self.ensure_index(len(profile.table_embedding))

        body = {
            "dataset_id": profile.dataset_id,
            "table_name": profile.table_name,
            "display_name": profile.display_name,
            "catalog_identifier": profile.catalog_identifier,
            "party": profile.party,
            "provider": profile.provider,
            "landing_page": profile.landing_page,
            "domain_confidence": profile.domain_confidence,
            "domain": profile.domain,
            "record_count": profile.record_count,
            "column_count": profile.column_count,
            "column_names": profile.column_names,
            "sample_records": profile.sample_records,
            "description": profile.description,
            "table_summary": profile.table_summary,
            "table_embedding": profile.table_embedding,
            "keywords": profile.keywords,
            "keywords_embedding": profile.keywords_embedding,
            "created_at": profile.created_at,
            "raw_storage_key": profile.raw_storage_key,
        }
        self.client.index_document(
            index=_DATASETS_INDEX,
            body=body,
            doc_id=profile.dataset_id,
            add_id_to_body=False,
        )
        logger.debug(f"[OpenSearch] Dataset upsert: {profile.table_name} (id={profile.dataset_id})")

    def search_knn(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        domain: str | None = None,
        party: str = "",
        provider: str = "",
        exclude_table_name: str = "",
    ) -> list[DatasetSearchResult]:
        """table_embedding KNN 검색 (도메인/파티/제공기관 조건 필터 지원)."""
        knn_query: dict[str, Any] = {
            "knn": {
                "table_embedding": {
                    "vector": query_embedding,
                    "k": top_k,
                }
            }
        }

        filters = [
            {"term": {field: value}}
            for field, value in (("domain", domain), ("party", party), ("provider", provider))
            if value
        ]
        bool_query: dict[str, Any] = {"must": knn_query}
        if filters:
            bool_query["filter"] = filters
        if exclude_table_name:
            bool_query["must_not"] = [{"term": {"table_name": exclude_table_name}}]

        query_body: dict[str, Any] = {
            "size": top_k,
            "query": knn_query if len(bool_query) == 1 else {"bool": bool_query},
        }

        response = self.client.client.search(index=_DATASETS_INDEX, body=query_body)

        return [
            DatasetSearchResult(
                table_name=hit["_source"]["table_name"],
                display_name=hit["_source"].get("display_name", ""),
                domain=hit["_source"]["domain"],
                description=hit["_source"].get("description", ""),
                table_summary=hit["_source"].get("table_summary", ""),
                score=hit["_score"],
                party=hit["_source"].get("party", ""),
                provider=hit["_source"].get("provider", ""),
                landing_page=hit["_source"].get("landing_page", ""),
            )
            for hit in response["hits"]["hits"]
        ]

    def get(self, table_name: str) -> DatasetProfile | None:
        """table_name으로 단건 조회 (가장 최근 등록 기준)."""
        try:
            resp = self.client.client.search(
                index=_DATASETS_INDEX,
                body={
                    "size": 1,
                    "query": {"term": {"table_name": table_name}},
                    "sort": [{"created_at": {"order": "desc"}}],
                },
            )
            hits = resp["hits"]["hits"]
            if not hits:
                return None
            return DatasetProfile(**hits[0]["_source"])
        except Exception:
            return None

    def list_domain_embeddings(self) -> list[tuple[str, str, list[float]]]:
        """도메인 분류 centroid 계산용 (table_name, domain, table_embedding) 목록."""
        if not self.client.client.indices.exists(index=_DATASETS_INDEX):
            return []
        try:
            resp = self.client.client.search(
                index=_DATASETS_INDEX,
                body={
                    "size": 1000,
                    "query": {"match_all": {}},
                    "_source": {"includes": ["table_name", "domain", "table_embedding"]},
                },
            )
            return [
                (
                    h["_source"].get("table_name", ""),
                    h["_source"].get("domain", ""),
                    h["_source"].get("table_embedding") or [],
                )
                for h in resp["hits"]["hits"]
            ]
        except Exception:
            return []

    def get_by_identifier(self, catalog_identifier: str) -> DatasetProfile | None:
        """통합 카탈로그 식별자(catalog_entry.identifier)로 단건 조회."""
        if not catalog_identifier:
            return None
        try:
            resp = self.client.client.search(
                index=_DATASETS_INDEX,
                body={
                    "size": 1,
                    "query": {"term": {"catalog_identifier": catalog_identifier}},
                    "sort": [{"created_at": {"order": "desc"}}],
                },
            )
            hits = resp["hits"]["hits"]
            if not hits:
                return None
            return DatasetProfile(**hits[0]["_source"])
        except Exception:
            return None

    def list_all(self) -> list[DatasetProfile]:
        """등록된 모든 Dataset 목록 조회 (임베딩 제외)."""
        if not self.client.client.indices.exists(index=_DATASETS_INDEX):
            return []
        try:
            response = self.client.client.search(
                index=_DATASETS_INDEX,
                body={
                    "size": 1000,
                    "query": {"match_all": {}},
                    "_source": {"excludes": ["table_embedding"]},
                    "sort": [{"created_at": {"order": "desc"}}],
                },
            )
            return [
                DatasetProfile(**hit["_source"])
                for hit in response["hits"]["hits"]
            ]
        except Exception:
            return []

    def delete(self, table_name: str) -> bool:
        """table_name과 일치하는 문서를 모두 삭제."""
        try:
            self.client.client.delete_by_query(
                index=_DATASETS_INDEX,
                body={"query": {"term": {"table_name": table_name}}},
            )
            return True
        except Exception:
            return False


def get_dataset_repository() -> DatasetRepository:
    return DatasetRepository()
