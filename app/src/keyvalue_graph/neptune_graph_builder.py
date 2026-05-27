"""Neptune + OpenSearch 통합 그래프 빌더."""

import json
import logging
from typing import Dict, Any, List
from urllib.parse import quote

from rdflib import RDF, XSD, Graph, Literal, Namespace

from app.clients.neptune import NeptuneClient, get_neptune_client
from app.repositories.opensearch_repository import OpenSearchRepository, get_opensearch_repository
from app.src.embedding.embedding_service import embedding_service
from app.config import settings
from app.src.keyvalue_graph.data_loader import DataRecord

logger = logging.getLogger(__name__)


class NeptuneGraphBuilder:
    """Neptune + OpenSearch 통합 Key-Value 그래프 빌더."""

    def __init__(
        self,
        neptune_client: NeptuneClient = None,
        opensearch_repo: OpenSearchRepository = None,
        namespace: str = settings.RDF_NAMESPACE,
        namespace_prefix: str = settings.RDF_NAMESPACE_PREFIX
    ):
        """초기화."""
        self.neptune_client = neptune_client or get_neptune_client()
        self.opensearch_repo = opensearch_repo or get_opensearch_repository()
        self.namespace = Namespace(namespace)
        self.namespace_prefix = namespace_prefix
        self._processed_keys = set()  # 이미 처리한 Key 추적 (배치 내 중복 방지)

    def build_from_record(
        self,
        record: DataRecord,
        centrality_name: str = "데이터현황",
        create_document: bool = True
    ) -> Dict[str, Any]:
        """단일 레코드로부터 Neptune + OpenSearch에 저장.

        Args:
            record: 데이터 레코드
            centrality_name: 중심 노드 이름
            create_document: Document 노드 생성 여부

        Returns:
            저장 결과
        """
        doc_id = record.get_document_id()

        try:
            # 1. Neptune에 RDF 트리플 저장
            sparql_insert = self._generate_insert_query(record, centrality_name, create_document)
            logger.info(f"생성된 SPARQL 쿼리:\n{sparql_insert}")
            self.neptune_client.execute_update(sparql_insert)

            # 2. OpenSearch에 임베딩 저장
            self._index_to_opensearch(record, centrality_name)

            return {
                "success": True,
                "document_id": doc_id,
                "keys_count": len(record.key_values),
                "values_count": sum(len(values) for values in record.key_values.values()),
                "message": "Neptune + OpenSearch 저장 완료"
            }

        except Exception as e:
            logger.error(f"그래프 생성 실패 {doc_id}: {e}")
            return {
                "success": False,
                "document_id": doc_id,
                "error": str(e)
            }

    def _escape_sparql_literal(self, s: str) -> str:
        """SPARQL 리터럴 문자열 이스케이핑."""
        return (s
            .replace('\\', '\\\\')  # \ -> \\
            .replace('"', '\\"')    # " -> \"
            .replace('\n', '\\n')   # newline
            .replace('\r', '\\r')   # carriage return
            .replace('\t', '\\t'))  # tab

    def _generate_insert_query(
        self,
        record: DataRecord,
        centrality_name: str,
        create_document: bool
    ) -> str:
        """SPARQL INSERT 쿼리 생성."""
        ns = str(self.namespace)
        doc_id = record.get_document_id()

        # URI 생성
        centrality_uri = f"<{ns}centrality/{quote(centrality_name, safe='')}>"
        doc_uri = f"<{ns}document/{quote(doc_id, safe='')}>"

        triples = []

        # Centrality 노드
        triples.append(f"{centrality_uri} <{RDF.type}> <{ns}Centrality> .")
        triples.append(f"{centrality_uri} <{ns}name> \"{self._escape_sparql_literal(centrality_name)}\" .")

        # Document 노드
        if create_document:
            triples.append(f"{doc_uri} <{RDF.type}> <{ns}Document> .")
            triples.append(f"{doc_uri} <{ns}documentId> \"{self._escape_sparql_literal(doc_id)}\" .")
            triples.append(f"{doc_uri} <{ns}sourceFile> \"{self._escape_sparql_literal(record.source_file)}\" .")
            triples.append(f"{doc_uri} <{ns}recordIndex> {record.record_index} .")
            # rawText는 OpenSearch에 저장되므로 Neptune에서는 생략

        # Key-Value 노드
        for key_name, values in record.key_values.items():
            key_uri = f"<{ns}key/{quote(key_name, safe='')}>"

            triples.append(f"{key_uri} <{RDF.type}> <{ns}Key> .")
            triples.append(f"{key_uri} <{ns}keyName> \"{self._escape_sparql_literal(key_name)}\" .")
            triples.append(f"{centrality_uri} <{ns}hasKey> {key_uri} .")

            for value in values:
                value_id = f"{doc_id}_{key_name}_{value}"
                value_uri = f"<{ns}value/{quote(value_id, safe='')}>"
                value_type = record.infer_value_type(key_name, value)

                triples.append(f"{value_uri} <{RDF.type}> <{ns}Value> .")
                triples.append(f"{value_uri} <{ns}valueId> \"{self._escape_sparql_literal(value_id)}\" .")
                triples.append(f"{value_uri} <{ns}valueContent> \"{self._escape_sparql_literal(str(value))}\" .")
                triples.append(f"{value_uri} <{ns}valueType> \"{value_type.value}\" .")
                triples.append(f"{key_uri} <{ns}hasValue> {value_uri} .")
                triples.append(f"{value_uri} <{ns}extractedFrom> {doc_uri} .")

        insert_data = "\n    ".join(triples)
        return f"""
PREFIX kv: <{ns}>

INSERT DATA {{
    {insert_data}
}}
"""

    def _index_to_opensearch(self, record: DataRecord, centrality_name: str):
        """OpenSearch에 임베딩 인덱싱."""
        doc_id = record.get_document_id()

        # 1. Document 인덱싱
        doc_content = json.dumps(record.raw_data, ensure_ascii=False)
        doc_embedding = embedding_service.generate_embedding(doc_content)

        self.opensearch_repo.index_document(
            doc_id=doc_id,
            content=doc_content,
            embedding=doc_embedding,
            metadata={
                "source_file": record.source_file,
                "record_index": record.record_index,
                "centrality": centrality_name
            }
        )

        # 2. Key 인덱싱 (중복 방지 + field_type_classifier)
        for key_name, values in record.key_values.items():
            # 이미 처리한 Key면 스킵
            if key_name in self._processed_keys:
                continue

            # field_type 분류
            from app.src.keyvalue_graph.field_type_classifier import field_type_classifier
            type_result = field_type_classifier.classify(
                key_name=key_name,
                sample_values=[str(v) for v in values[:5]]
            )
            key_type = type_result["field_type"]
            key_type_confidence = type_result["confidence"]

            # 임베딩 생성 (semantic/proper_noun만)
            key_embedding = None
            if key_type in ["semantic", "proper_noun"]:
                key_embedding = embedding_service.generate_embedding(key_name)

            self.opensearch_repo.index_key(
                key_id=key_name,
                key_name=key_name,
                key_type=key_type,
                key_type_confidence=key_type_confidence,
                embedding=key_embedding
            )

            self._processed_keys.add(key_name)

        # 3. Value 인덱싱 (key_type 기반 처리)
        for key_name, values in record.key_values.items():
            # Key 타입 조회 (_processed_keys에 있거나 OpenSearch에서 조회)
            key_type = self._get_key_type(key_name)

            for value in values:
                value_id = f"{doc_id}_{key_name}_{value}"
                value_content = str(value)

                # 타입별 처리
                value_embedding = None
                numeric_value = None

                if key_type in ["semantic", "proper_noun"]:
                    value_embedding = embedding_service.generate_embedding(value_content)
                elif key_type == "numeric_date":
                    try:
                        numeric_value = float(value_content)
                    except ValueError:
                        logger.warning(f"numeric_date 타입이지만 숫자 변환 실패: {value_content}")

                self.opensearch_repo.index_value(
                    value_id=value_id,
                    value_content=value_content,
                    parent_key_id=key_name,
                    parent_key_type=key_type,
                    embedding=value_embedding,
                    value_type=record.infer_value_type(key_name, value).value,
                    numeric_value=numeric_value
                )

    def _get_key_type(self, key_name: str) -> str:
        """Key 타입 조회 헬퍼 메서드 (OpenSearch에서 조회)

        Args:
            key_name: Key명

        Returns:
            key_type (semantic/numeric_date/proper_noun)
        """
        try:
            # OpenSearch에서 Key 정보 조회
            result = self.opensearch_repo.client.client.get(
                index=self.opensearch_repo.keys_index,
                id=key_name
            )
            return result.get("_source", {}).get("key_type", "semantic")
        except Exception:
            # 조회 실패 시 기본값 반환
            return "semantic"

    def build_batch(
        self,
        records: List[DataRecord],
        centrality_name: str = "데이터현황"
    ) -> Dict[str, Any]:
        """배치로 여러 레코드 저장.

        Args:
            records: 데이터 레코드 리스트
            centrality_name: 중심 노드 이름

        Returns:
            통계 정보
        """
        total_count = len(records)
        success_count = 0
        failed_count = 0
        total_keys = 0
        total_values = 0

        for record in records:
            result = self.build_from_record(record, centrality_name)

            if result["success"]:
                success_count += 1
                total_keys += result["keys_count"]
                total_values += result["values_count"]
            else:
                failed_count += 1

        return {
            "total_records": total_count,
            "success": success_count,
            "failed": failed_count,
            "total_keys": total_keys,
            "total_values": total_values
        }
