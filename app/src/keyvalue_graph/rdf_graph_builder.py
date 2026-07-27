"""RDF(JSON-LD) 기반 그래프 빌더.

Key-Value 데이터를 rdflib를 사용하여 RDF 트리플로 변환하고
JSON-LD 형태의 딕셔너리로 반환한다.
"""

import json
from typing import Any
from urllib.parse import quote

import logging
from rdflib import RDF, XSD, Graph, Literal, Namespace

from app.config import settings
from app.schemas.keyvalue_graph import RdfGraphBuildResult
from app.src.keyvalue_graph.data_loader import DataRecord
from app.src.keyvalue_graph.field_mappings import FIELD_NAME_MAPPING

logger = logging.getLogger(__name__)


class RdfGraphBuilder:
    """RDF(JSON-LD) 기반 Key-Value 그래프 빌더.

    rdflib를 사용하여 RDF 트리플을 생성하고
    JSON-LD 딕셔너리로 반환한다.
    """

    def __init__(
            self,
            namespace: str = settings.RDF_NAMESPACE,
            namespace_prefix: str = settings.RDF_NAMESPACE_PREFIX,
    ):
        """RDF 그래프 빌더를 초기화한다."""
        self.namespace = Namespace(namespace)
        self.namespace_prefix = namespace_prefix
        self.jsonld_context = self._build_jsonld_context()

    def _build_jsonld_context(self) -> dict:
        """인스턴스 namespace/prefix 기반으로 JSON-LD @context를 생성한다."""
        p = self.namespace_prefix
        return {
            p: str(self.namespace),
            "rdf": str(RDF),
            "xsd": str(XSD),
            "Centrality": f"{p}:Centrality",
            "Document": f"{p}:Document",
            "Key": f"{p}:Key",
            "Value": f"{p}:Value",
            "name": f"{p}:name",
            "hasKey": {"@id": f"{p}:hasKey", "@type": "@id"},
            "hasValue": {"@id": f"{p}:hasValue", "@type": "@id"},
            "extractedFrom": {"@id": f"{p}:extractedFrom", "@type": "@id"},
            "sourceFile": f"{p}:sourceFile",
            "recordIndex": {"@id": f"{p}:recordIndex", "@type": "xsd:integer"},
            "rawText": f"{p}:rawText",
            "originalField": f"{p}:originalField",
            "valueType": f"{p}:valueType",
            "keyName": f"{p}:keyName",
        }

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
            g.bind(self.namespace_prefix, self.namespace)

            keys_count, values_count = self._add_record_to_graph(g, record, centrality_name, create_document)

            jsonld_str = g.serialize(format="json-ld", context=self.jsonld_context)
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

    def _add_record_to_graph(
            self, g: Graph, record: DataRecord, centrality_name: str, create_document: bool = True
    ) -> tuple[int, int]:
        """단일 레코드의 트리플을 기존 Graph에 추가한다.

        Args:
            g: 트리플을 추가할 rdflib Graph
            record: 데이터 레코드
            centrality_name: 중심 노드 이름
            create_document: Document 노드 생성 여부

        Returns:
            (keys_count, values_count)
        """
        ns = self.namespace
        doc_id = record.get_document_id()

        # Centrality 노드
        centrality_uri = ns[f"centrality/{quote(centrality_name, safe='')}"]
        g.add((centrality_uri, RDF.type, ns.Centrality))
        g.add((centrality_uri, ns.name, Literal(centrality_name)))

        # Document 노드
        doc_uri = ns[f"document/{quote(doc_id, safe='')}"]
        if create_document:
            raw_text = json.dumps(record.raw_data, ensure_ascii=False)
            g.add((doc_uri, RDF.type, ns.Document))
            g.add((doc_uri, ns.name, Literal(doc_id)))
            g.add((doc_uri, ns.sourceFile, Literal(record.source_file)))
            g.add((doc_uri, ns.recordIndex, Literal(record.record_index, datatype=XSD.integer)))
            g.add((doc_uri, ns.rawText, Literal(raw_text)))

        # Key-Value 노드 및 관계
        keys_count = len(record.key_values)
        values_count = 0

        for key_name, values in record.key_values.items():
            key_uri = ns[f"key/{quote(key_name, safe='')}"]
            g.add((key_uri, RDF.type, ns.Key))
            g.add((key_uri, ns.name, Literal(key_name)))

            original_field = self._find_original_field(key_name)
            if original_field:
                g.add((key_uri, ns.originalField, Literal(original_field)))

            g.add((centrality_uri, ns.hasKey, key_uri))

            for value in values:
                value_id = f"{key_name}:{value}"
                value_uri = ns[f"value/{quote(value_id, safe='')}"]
                value_type = record.infer_value_type(key_name, value)

                g.add((value_uri, RDF.type, ns.Value))
                g.add((value_uri, ns.name, Literal(value)))
                g.add((value_uri, ns.valueType, Literal(value_type.value)))
                g.add((value_uri, ns.keyName, Literal(key_name)))
                g.add((key_uri, ns.hasValue, value_uri))
                g.add((value_uri, ns.extractedFrom, doc_uri))

                values_count += 1

        return keys_count, values_count

    def build_batch(self, records: list[DataRecord], centrality_name: str = "데이터현황") -> dict[str, Any]:
        """배치로 여러 레코드를 하나의 통합 RDF 그래프로 변환.

        모든 레코드를 단일 rdflib.Graph에 축적한다.
        동일 URI의 트리플은 rdflib의 set 특성에 의해 자동 중복제거된다.

        Args:
            records: 데이터 레코드 리스트
            centrality_name: 중심 노드 이름

        Returns:
            통계 정보 + 통합 JSON-LD dict
        """
        ns = self.namespace
        g = Graph()
        g.bind(self.namespace_prefix, ns)

        total_count = len(records)
        success_count = 0
        failed_count = 0
        total_keys = 0
        total_values = 0

        for record in records:
            try:
                keys_count, values_count = self._add_record_to_graph(g, record, centrality_name)
                success_count += 1
                total_keys += keys_count
                total_values += values_count
            except Exception as e:
                logger.error(f"Failed to add record {record.get_document_id()} to RDF graph: {e}")
                failed_count += 1

        # 통합 그래프를 단일 JSON-LD로 직렬화
        jsonld_str = g.serialize(format="json-ld", context=self.jsonld_context)
        jsonld_data = json.loads(jsonld_str)

        return {
            "total_records": total_count,
            "success": success_count,
            "failed": failed_count,
            "total_keys": total_keys,
            "total_values": total_values,
            "total_triples": len(g),
            "json_ld": jsonld_data,
        }

    @staticmethod
    def _find_original_field(key_name: str) -> str | None:
        """한글 Key명에 대응하는 원본 필드명을 찾는다."""
        for orig, korean in FIELD_NAME_MAPPING.items():
            if korean == key_name:
                return orig
        return None
