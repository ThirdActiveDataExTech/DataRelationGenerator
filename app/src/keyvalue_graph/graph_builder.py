from typing import Any, Dict, List

import logging

from app.repositories.graph_repository import GraphRepository
from app.schemas.keyvalue_graph import GraphBuildResult
from app.src.keyvalue_graph.data_loader import DataRecord

logger = logging.getLogger(__name__)


class KeyValueGraphBuilder:
    """Key-Value 기반 그래프 빌더."""

    def __init__(self, graph_repo: GraphRepository = None):
        """빌더 초기화.

        Args:
            graph_repo: Neo4j 리포지토리
        """
        self.graph_repo = graph_repo or GraphRepository()

    def build_from_record(
            self, record: DataRecord, centrality_name: str = "데이터현황", create_document: bool = True
    ) -> GraphBuildResult:
        """단일 레코드로부터 그래프 생성.

        Args:
            record: 데이터 레코드
            centrality_name: 중심 노드 이름 (예: "병원현황", "판례정보")
            create_document: Document 노드 생성 여부

        Returns:
            GraphBuildResult
        """
        doc_id = record.get_document_id()

        try:
            # 1. Centrality 노드 생성
            self._create_centrality_node(centrality_name)

            # 2. Document 노드 생성
            if create_document:
                self._create_document_node(record)

            # 3. Key-Value 노드 및 관계 생성
            keys_count = len(record.key_values)
            values_count = 0

            for key_name, values in record.key_values.items():
                # Key 노드 생성 (MERGE)
                self._create_key_node(key_name, record.raw_data)

                # Centrality -HAS_KEY-> Key
                self._create_centrality_to_key_relation(centrality_name, key_name)

                for value in values:
                    # Value 노드 생성
                    value_id = self._create_value_node(
                        key_name, value, record.infer_value_type(key_name, value)
                    )
                    values_count += 1

                    # Key -HAS_VALUE-> Value
                    self._create_has_value_relation(key_name, value_id)

                    # Value -EXTRACTED_FROM-> Document
                    self._create_extracted_from_relation(value_id, doc_id)

            return GraphBuildResult(
                document_id=doc_id,
                keys_count=keys_count,
                values_count=values_count,
                success=True,
                message=None,
            )

        except Exception as e:
            logger.error(f"Failed to build graph for {doc_id}: {e}")
            return GraphBuildResult(
                document_id=doc_id,
                keys_count=0,
                values_count=0,
                normalized_count=0,
                success=False,
                message=str(e),
            )

    def _create_centrality_node(self, centrality_name: str) -> None:
        """Centrality 노드 생성 (중심 노드).

        Args:
            centrality_name: 중심 노드 이름 (예: "병원현황")
        """
        query = """
        MERGE (c:Centrality {name: $centrality_name})
        ON CREATE SET
            c.created_at = datetime()
        """
        self.graph_repo.execute_query(
            query,
            {"centrality_name": centrality_name},
        )
        logger.info(f"Centrality 노드 생성: {centrality_name}")

    def _create_document_node(self, record: DataRecord) -> None:
        """Document 노드 생성.

        Args:
            record: 데이터 레코드
        """
        import json

        query = """
        CREATE (d:Document {
            id: $doc_id,
            source_file: $source_file,
            record_index: $record_index,
            raw_text: $raw_text
        })
        """
        self.graph_repo.execute_query(
            query,
            {
                "doc_id": record.get_document_id(),
                "source_file": record.source_file,
                "record_index": record.record_index,
                "raw_text": json.dumps(record.raw_data, ensure_ascii=False),
            },
        )

    def _create_key_node(self, key_name: str, raw_data: Dict[str, Any]) -> None:
        """Key 노드 생성 (MERGE).

        Args:
            key_name: Key명 (한글)
            raw_data: 원본 데이터 (필드명 추출용)
        """
        # 원본 필드명 찾기
        from app.src.keyvalue_graph.field_mappings import FIELD_NAME_MAPPING

        original_field = None
        for orig, korean in FIELD_NAME_MAPPING.items():
            if korean == key_name:
                original_field = orig
                break

        query = """
        MERGE (k:Key {name: $key_name})
        ON CREATE SET
            k.original_field = $original_field,
            k.created_at = datetime()
        """
        self.graph_repo.execute_query(
            query,
            {
                "key_name": key_name,
                "original_field": original_field or key_name,
            },
        )

    def _create_value_node(self, key_name: str, value: str, value_type: str) -> str:
        """Value 노드 생성.

        Args:
            key_name: Key명
            value: 값
            value_type: 값 타입

        Returns:
            Value ID (key_name:value 형식)
        """
        value_id = f"{key_name}:{value}"

        query = """
        MERGE (v:Value {id: $value_id})
        ON CREATE SET
            v.name = $value,
            v.value_type = $value_type,
            v.key_name = $key_name,
            v.created_at = datetime()
        """
        self.graph_repo.execute_query(
            query,
            {
                "value_id": value_id,
                "value": value,
                "value_type": value_type,
                "key_name": key_name,
            },
        )

        return value_id

    def _create_normalized_value_node(self, key_name: str, normalized_value: str) -> None:
        """NormalizedValue 노드 생성.

        Args:
            key_name: Key명
            normalized_value: 정규화된 값
        """
        query = """
        MERGE (nv:NormalizedValue {id: $norm_id})
        ON CREATE SET
            nv.name = $normalized_value,
            nv.key_name = $key_name,
            nv.count = 0,
            nv.created_at = datetime()
        ON MATCH SET
            nv.count = nv.count + 1
        """
        norm_id = f"{key_name}:normalized:{normalized_value}"
        self.graph_repo.execute_query(
            query,
            {
                "norm_id": norm_id,
                "normalized_value": normalized_value,
                "key_name": key_name,
            },
        )

    def _create_centrality_to_key_relation(self, centrality_name: str, key_name: str) -> None:
        """Centrality -[HAS_KEY]-> Key 관계 생성.

        Args:
            centrality_name: Centrality 노드 이름
            key_name: Key명
        """
        query = """
        MATCH (c:Centrality {name: $centrality_name})
        MATCH (k:Key {name: $key_name})
        MERGE (c)-[r:HAS_KEY]->(k)
        """
        self.graph_repo.execute_query(
            query,
            {
                "centrality_name": centrality_name,
                "key_name": key_name,
            },
        )

    def _create_has_value_relation(self, key_name: str, value_id: str) -> None:
        """HAS_VALUE 관계 생성.

        Args:
            key_name: Key명
            value_id: Value ID
        """
        query = """
        MATCH (k:Key {name: $key_name})
        MATCH (v:Value {id: $value_id})
        MERGE (k)-[r:HAS_VALUE]->(v)
        """
        self.graph_repo.execute_query(
            query,
            {
                "key_name": key_name,
                "value_id": value_id,
            },
        )

    def _create_extracted_from_relation(self, value_id: str, doc_id: str) -> None:
        """EXTRACTED_FROM 관계 생성.

        Args:
            value_id: Value ID
            doc_id: Document ID
        """
        query = """
        MATCH (v:Value {id: $value_id})
        MATCH (d:Document {id: $doc_id})
        MERGE (v)-[r:EXTRACTED_FROM]->(d)
        """
        self.graph_repo.execute_query(
            query,
            {
                "value_id": value_id,
                "doc_id": doc_id,
            },
        )

    def _create_same_as_relation(
            self, value_id: str, key_name: str, normalized_value: str
    ) -> None:
        """SAME_AS 관계 생성.

        Args:
            value_id: Value ID
            key_name: Key명
            normalized_value: 정규화된 값
        """
        norm_id = f"{key_name}:normalized:{normalized_value}"

        query = """
        MATCH (v:Value {id: $value_id})
        MATCH (nv:NormalizedValue {id: $norm_id})
        MERGE (v)-[r:SAME_AS]->(nv)
        """
        self.graph_repo.execute_query(
            query,
            {
                "value_id": value_id,
                "norm_id": norm_id,
            },
        )

    def build_batch(self, records: List[DataRecord], centrality_name: str = "데이터현황") -> Dict[str, Any]:
        """배치로 여러 레코드 처리.

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
        total_normalized = 0

        for record in records:
            result = self.build_from_record(record, centrality_name=centrality_name)
            if result.success:
                success_count += 1
                total_keys += result.keys_count
                total_values += result.values_count
                total_normalized += result.normalized_count
            else:
                failed_count += 1

        return {
            "total_records": total_count,
            "success": success_count,
            "failed": failed_count,
            "total_keys": total_keys,
            "total_values": total_values,
            "total_normalized": total_normalized,
            "normalization_rate": (
                total_normalized / total_values if total_values > 0 else 0.0
            ),
        }
