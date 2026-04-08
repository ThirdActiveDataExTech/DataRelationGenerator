"""RDF(JSON-LD) 기반 그래프 빌더.

Key-Value 데이터를 rdflib를 사용하여 RDF 트리플로 변환하고
JSON-LD 형태의 딕셔너리로 반환한다.
"""

import json
import logging
from typing import Any, Dict, List
from urllib.parse import quote

from rdflib import RDF, XSD, Graph, Literal, Namespace

from app.schemas.keyvalue_graph import RdfGraphBuildResult
from app.src.keyvalue_graph.data_loader import DataRecord

logger = logging.getLogger(__name__)

# 커스텀 네임스페이스
CUSTOM_NAMESPACE = Namespace("http://example.org/keyvalue#")
CUSTOM_NAMESPACE_PREFIX: str = "kv"


# JSON-LD @context 정의
JSONLD_CONTEXT = {
    CUSTOM_NAMESPACE_PREFIX: str(CUSTOM_NAMESPACE),
    "rdf": str(RDF),
    "xsd": str(XSD),
    "Centrality": f"{CUSTOM_NAMESPACE_PREFIX}:Centrality",
    "Document": f"{CUSTOM_NAMESPACE_PREFIX}:Document",
    "Key": f"{CUSTOM_NAMESPACE_PREFIX}:Key",
    "Value": f"{CUSTOM_NAMESPACE_PREFIX}:Value",
    "name": f"{CUSTOM_NAMESPACE_PREFIX}:name",
    "hasKey": {"@id": f"{CUSTOM_NAMESPACE_PREFIX}:hasKey", "@type": "@id"},
    "hasValue": {"@id": f"{CUSTOM_NAMESPACE_PREFIX}:hasValue", "@type": "@id"},
    "extractedFrom": {"@id": f"{CUSTOM_NAMESPACE_PREFIX}:extractedFrom", "@type": "@id"},
    "sourceFile": f"{CUSTOM_NAMESPACE_PREFIX}:sourceFile",
    "recordIndex": {"@id": f"{CUSTOM_NAMESPACE_PREFIX}:recordIndex", "@type": "xsd:integer"},
    "rawText": f"{CUSTOM_NAMESPACE_PREFIX}:rawText",
    "originalField": f"{CUSTOM_NAMESPACE_PREFIX}:originalField",
    "valueType": f"{CUSTOM_NAMESPACE_PREFIX}:valueType",
    "keyName": f"{CUSTOM_NAMESPACE_PREFIX}:keyName",
}


class RdfGraphBuilder:
    """RDF(JSON-LD) 기반 Key-Value 그래프 빌더.

    Neo4j 없이 rdflib를 사용하여 RDF 트리플을 생성하고
    JSON-LD 딕셔너리로 반환한다.
    """

    def build_from_record(
        self, record: DataRecord, centrality_name: str = "데이터현황", create_document: bool = True
    ) -> RdfGraphBuildResult:
        """단일 레코드로부터 RDF 그래프 생성 (JSON-LD dict 반환).

        Args:
            record: 데이터 레코드
            centrality_name: 중심 노드 이름 (예: "병원현황", "판례정보")
            create_document: Document 노드 생성 여부

        Returns:
            RdfGraphBuildResult (json_ld 딕셔너리 포함)
        """
        doc_id = record.get_document_id()

        try:
            g = Graph()
            g.bind(CUSTOM_NAMESPACE_PREFIX, CUSTOM_NAMESPACE)

            # 1. Centrality 노드
            centrality_uri = CUSTOM_NAMESPACE[f"centrality/{quote(centrality_name, safe='')}"]
            g.add((centrality_uri, RDF.type, CUSTOM_NAMESPACE.Centrality))
            g.add((centrality_uri, CUSTOM_NAMESPACE.name, Literal(centrality_name)))

            # 2. Document 노드
            doc_uri = CUSTOM_NAMESPACE[f"document/{quote(doc_id, safe='')}"]
            if create_document:
                raw_text = json.dumps(record.raw_data, ensure_ascii=False)
                g.add((doc_uri, RDF.type, CUSTOM_NAMESPACE.Document))
                g.add((doc_uri, CUSTOM_NAMESPACE.name, Literal(doc_id)))
                g.add((doc_uri, CUSTOM_NAMESPACE.sourceFile, Literal(record.source_file)))
                g.add((doc_uri, CUSTOM_NAMESPACE.recordIndex, Literal(record.record_index, datatype=XSD.integer)))
                g.add((doc_uri, CUSTOM_NAMESPACE.rawText, Literal(raw_text)))

            # 3. Key-Value 노드 및 관계
            keys_count = len(record.key_values)
            values_count = 0

            for key_name, values in record.key_values.items():
                key_uri = CUSTOM_NAMESPACE[f"key/{quote(key_name, safe='')}"]

                # Key 노드 속성
                g.add((key_uri, RDF.type, CUSTOM_NAMESPACE.Key))
                g.add((key_uri, CUSTOM_NAMESPACE.name, Literal(key_name)))

                # 원본 필드명 매핑
                original_field = self._find_original_field(key_name)
                if original_field:
                    g.add((key_uri, CUSTOM_NAMESPACE.originalField, Literal(original_field)))

                # Centrality -hasKey-> Key
                g.add((centrality_uri, CUSTOM_NAMESPACE.hasKey, key_uri))

                for value in values:
                    value_id = f"{key_name}:{value}"
                    value_uri = CUSTOM_NAMESPACE[f"value/{quote(value_id, safe='')}"]
                    value_type = record.infer_value_type(key_name, value)

                    # Value 노드 속성
                    g.add((value_uri, RDF.type, CUSTOM_NAMESPACE.Value))
                    g.add((value_uri, CUSTOM_NAMESPACE.name, Literal(value)))
                    g.add((value_uri, CUSTOM_NAMESPACE.valueType, Literal(value_type.value)))
                    g.add((value_uri, CUSTOM_NAMESPACE.keyName, Literal(key_name)))

                    # Key -hasValue-> Value
                    g.add((key_uri, CUSTOM_NAMESPACE.hasValue, value_uri))

                    # Value -extractedFrom-> Document
                    g.add((value_uri, CUSTOM_NAMESPACE.extractedFrom, doc_uri))

                    values_count += 1

            # JSON-LD 직렬화 (compact form with @context)
            context = {CUSTOM_NAMESPACE_PREFIX: str(CUSTOM_NAMESPACE)}
            jsonld_str = g.serialize(format="json-ld", context=context)
            jsonld_data = json.loads(jsonld_str)

            return RdfGraphBuildResult(
                document_id=doc_id,
                keys_count=keys_count,
                values_count=values_count,
                json_ld=jsonld_data,
                success=True,
                message=None,
            )

        except Exception as e:
            logger.error(f"Failed to build RDF graph for {doc_id}: {e}")
            return RdfGraphBuildResult(
                document_id=doc_id,
                keys_count=0,
                values_count=0,
                json_ld=None,
                success=False,
                message=str(e),
            )

    def build_batch(self, records: List[DataRecord], centrality_name: str = "데이터현황") -> Dict[str, Any]:
        """배치로 여러 레코드를 RDF로 변환.

        Args:
            records: 데이터 레코드 리스트
            centrality_name: 중심 노드 이름

        Returns:
            통계 정보 + JSON-LD 리스트
        """
        total_count = len(records)
        success_count = 0
        failed_count = 0
        total_keys = 0
        total_values = 0
        jsonld_results = []

        for record in records:
            result = self.build_from_record(record, centrality_name=centrality_name)
            if result.success:
                success_count += 1
                total_keys += result.keys_count
                total_values += result.values_count
                jsonld_results.append(result.json_ld)
            else:
                failed_count += 1

        return {
            "total_records": total_count,
            "success": success_count,
            "failed": failed_count,
            "total_keys": total_keys,
            "total_values": total_values,
            "json_ld_graphs": jsonld_results,
        }

    @staticmethod
    def _find_original_field(key_name: str) -> str | None:
        """한글 Key명에 대응하는 원본 필드명을 찾는다."""
        from app.src.keyvalue_graph.field_mappings import FIELD_NAME_MAPPING

        for orig, korean in FIELD_NAME_MAPPING.items():
            if korean == key_name:
                return orig
        return None
