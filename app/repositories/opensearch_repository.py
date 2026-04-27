"""OpenSearch Repository 모듈."""

import logging
from typing import List, Optional, Dict, Any

from app.clients.opensearch import OpenSearchClient, get_opensearch_client
from app.config import settings

logger = logging.getLogger(__name__)


class OpenSearchRepository:
    """OpenSearch 저장소."""

    def __init__(self, client: Optional[OpenSearchClient] = None):
        """OpenSearch 저장소 초기화.

        Args:
            client: OpenSearchClient 인스턴스 (기본값: None, 자동 생성)
        """
        self.client = client or get_opensearch_client()
        self.documents_index = settings.OPENSEARCH_DOCUMENTS_INDEX
        self.keys_index = settings.OPENSEARCH_KEYS_INDEX
        self.values_index = settings.OPENSEARCH_VALUES_INDEX

    def index_document(
        self,
        doc_id: str,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Document 노드 인덱싱.

        Args:
            doc_id: 문서 ID
            content: 문서 내용
            embedding: 문서 임베딩 벡터 (768차원)
            metadata: 메타데이터

        Returns:
            인덱싱 결과
        """
        body = {
            "doc_id": doc_id,
            "content": content,
            "embedding": embedding,
            "metadata": metadata or {}
        }
        return self.client.index_document(self.documents_index, doc_id, body)

    def index_key(
        self,
        key_id: str,
        key_name: str,
        embedding: List[float]
    ) -> Dict[str, Any]:
        """Key 노드 인덱싱.

        Args:
            key_id: Key ID
            key_name: Key 이름
            embedding: Key 임베딩 벡터 (768차원)

        Returns:
            인덱싱 결과
        """
        body = {
            "key_id": key_id,
            "key_name": key_name,
            "embedding": embedding
        }
        return self.client.index_document(self.keys_index, key_id, body)

    def index_value(
        self,
        value_id: str,
        value_content: str,
        parent_key_id: str,
        embedding: List[float],
        value_type: str = "text"
    ) -> Dict[str, Any]:
        """Value 노드 인덱싱.

        Args:
            value_id: Value ID
            value_content: Value 내용
            parent_key_id: 부모 Key ID
            embedding: Value 임베딩 벡터 (768차원)
            value_type: Value 타입 (text, number 등)

        Returns:
            인덱싱 결과
        """
        body = {
            "value_id": value_id,
            "value_content": value_content,
            "parent_key_id": parent_key_id,
            "embedding": embedding,
            "value_type": value_type
        }
        return self.client.index_document(self.values_index, value_id, body)

    def bulk_index_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Document 노드 대량 인덱싱.

        Args:
            documents: 문서 리스트
                각 문서는 {_id, doc_id, content, embedding, metadata} 포함

        Returns:
            벌크 인덱싱 결과
        """
        return self.client.bulk_index(self.documents_index, documents)

    def bulk_index_keys(self, keys: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Key 노드 대량 인덱싱.

        Args:
            keys: Key 리스트
                각 Key는 {_id, key_id, key_name, embedding} 포함

        Returns:
            벌크 인덱싱 결과
        """
        return self.client.bulk_index(self.keys_index, keys)

    def bulk_index_values(self, values: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Value 노드 대량 인덱싱.

        Args:
            values: Value 리스트
                각 Value는 {_id, value_id, value_content, parent_key_id, embedding, value_type} 포함

        Returns:
            벌크 인덱싱 결과
        """
        return self.client.bulk_index(self.values_index, values)

    def search_documents(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_doc_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Document 벡터 검색.

        Args:
            query_vector: 쿼리 임베딩 벡터
            top_k: 반환할 최대 결과 수
            filter_doc_ids: 검색 범위를 제한할 Document ID 리스트

        Returns:
            검색 결과 리스트 (각 결과는 doc_id, content, score 포함)
        """
        query_body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": {
                        "knn": {
                            "embedding": {
                                "vector": query_vector,
                                "k": top_k
                            }
                        }
                    }
                }
            }
        }

        if filter_doc_ids:
            query_body["query"]["bool"]["filter"] = [
                {"terms": {"doc_id": filter_doc_ids}}
            ]

        response = self.client.search(self.documents_index, query_body)

        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "doc_id": hit["_source"]["doc_id"],
                "content": hit["_source"]["content"],
                "score": hit["_score"],
                "metadata": hit["_source"].get("metadata", {})
            })

        return results

    def search_keys(
        self,
        query_vector: List[float],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Key 벡터 검색.

        Args:
            query_vector: 쿼리 임베딩 벡터
            top_k: 반환할 최대 결과 수

        Returns:
            검색 결과 리스트 (각 결과는 key_id, key_name, score 포함)
        """
        query_body = {
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_vector,
                        "k": top_k
                    }
                }
            }
        }

        response = self.client.search(self.keys_index, query_body)

        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "key_id": hit["_source"]["key_id"],
                "key_name": hit["_source"]["key_name"],
                "score": hit["_score"]
            })

        return results

    def search_values(
        self,
        query_vector: List[float],
        parent_key_ids: List[str],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Value 벡터 검색 (특정 Key들에 속한 Value만).

        Args:
            query_vector: 쿼리 임베딩 벡터
            parent_key_ids: 검색 범위를 제한할 Key ID 리스트
            top_k: 반환할 최대 결과 수

        Returns:
            검색 결과 리스트 (각 결과는 value_id, value_content, parent_key_id, score 포함)
        """
        query_body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": {
                        "knn": {
                            "embedding": {
                                "vector": query_vector,
                                "k": top_k
                            }
                        }
                    },
                    "filter": [
                        {"terms": {"parent_key_id": parent_key_ids}}
                    ]
                }
            }
        }

        response = self.client.search(self.values_index, query_body)

        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "value_id": hit["_source"]["value_id"],
                "value_content": hit["_source"]["value_content"],
                "parent_key_id": hit["_source"]["parent_key_id"],
                "value_type": hit["_source"]["value_type"],
                "score": hit["_score"]
            })

        return results

    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """Document 삭제.

        Args:
            doc_id: 문서 ID

        Returns:
            삭제 결과
        """
        return self.client.delete_document(self.documents_index, doc_id)

    def delete_key(self, key_id: str) -> Dict[str, Any]:
        """Key 삭제.

        Args:
            key_id: Key ID

        Returns:
            삭제 결과
        """
        return self.client.delete_document(self.keys_index, key_id)

    def delete_value(self, value_id: str) -> Dict[str, Any]:
        """Value 삭제.

        Args:
            value_id: Value ID

        Returns:
            삭제 결과
        """
        return self.client.delete_document(self.values_index, value_id)


def get_opensearch_repository() -> OpenSearchRepository:
    """OpenSearch 저장소 팩토리 함수.

    Returns:
        OpenSearchRepository 인스턴스
    """
    return OpenSearchRepository()
