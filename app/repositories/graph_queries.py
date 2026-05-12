"""Neptune SPARQL 쿼리 상수

Knowledge Graph 작업에 사용되는 SPARQL 쿼리를 관리합니다.
"""

from app.config import settings

# RDF Namespace
NS = settings.RDF_NAMESPACE
PREFIX = settings.RDF_NAMESPACE_PREFIX

# ========== SPARQL Prefixes ==========
SPARQL_PREFIXES = f"""
PREFIX {PREFIX}: <{NS}>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""

# ========== Document 조회 ==========

GET_DOCUMENTS_BY_IDS = SPARQL_PREFIXES + """
SELECT ?doc_id ?text ?description
WHERE {
    ?doc rdf:type kv:Document .
    ?doc kv:documentId ?doc_id .
    OPTIONAL { ?doc kv:rawText ?text }
    OPTIONAL { ?doc kv:sourceFile ?description }
    FILTER(?doc_id IN ($doc_ids))
}
"""

GET_DOCUMENT_INFO = SPARQL_PREFIXES + """
SELECT ?text ?description
WHERE {
    ?doc rdf:type kv:Document .
    ?doc kv:documentId "$doc_id" .
    OPTIONAL { ?doc kv:rawText ?text }
    OPTIONAL { ?doc kv:sourceFile ?description }
}
"""

GET_DOCUMENT_NODES = SPARQL_PREFIXES + """
SELECT ?id ?description
WHERE {
    ?doc rdf:type kv:Document .
    ?doc kv:documentId ?id .
    OPTIONAL { ?doc kv:sourceFile ?description }
    FILTER(?id IN ($doc_ids))
}
"""

# ========== 검색 관련 ==========

# Note: Neptune ML을 사용한 vector 검색은 별도 구현 필요
SEARCH_BY_GRAPH_EMBEDDING = """
# Neptune ML을 사용하여 vector similarity search 수행
# 현재는 placeholder - Neptune ML 설정 후 구현 필요
"""

# ========== 그래프 구조 관련 (Key-Value) ==========

# SPARQL에서는 그래프 구조를 한 번에 가져오기 어려우므로
# 여러 쿼리로 분할하거나 CONSTRUCT 사용

EXTRACT_GRAPH_1HOP = SPARQL_PREFIXES + """
# 1-hop 그래프: Document -> Value -> Key
# Neptune에서는 여러 쿼리로 분할하여 조회해야 함

# 쿼리 1: Document 노드 조회
SELECT ?doc ?doc_id ?raw_text ?source_file
WHERE {
    ?doc rdf:type kv:Document .
    ?doc kv:documentId ?doc_id .
    OPTIONAL { ?doc kv:rawText ?raw_text }
    OPTIONAL { ?doc kv:sourceFile ?source_file }
    FILTER(?doc_id IN ($doc_ids))
}

# 쿼리 2: Value 노드 및 EXTRACTED_FROM 관계 조회
# SELECT ?value ?value_id ?value_content ?doc_id
# WHERE {
#     ?doc rdf:type kv:Document .
#     ?doc kv:documentId ?doc_id .
#     ?value kv:extractedFrom ?doc .
#     ?value kv:valueContent ?value_content .
#     OPTIONAL { ?value kv:valueId ?value_id }
#     FILTER(?doc_id IN ($doc_ids))
# }

# 쿼리 3: Key 노드 및 HAS_VALUE 관계 조회
# SELECT ?key ?key_name ?value_id
# WHERE {
#     ?value kv:extractedFrom ?doc .
#     ?doc kv:documentId ?doc_id .
#     ?key kv:hasValue ?value .
#     ?key kv:keyName ?key_name .
#     OPTIONAL { ?value kv:valueId ?value_id }
#     FILTER(?doc_id IN ($doc_ids))
# }
"""

EXTRACT_GRAPH_2HOP = SPARQL_PREFIXES + """
# 2-hop 그래프
# Neptune에서는 복잡한 path 쿼리가 제한적이므로
# 필요한 관계를 명시적으로 조회해야 함
# 현재는 placeholder
"""

EXTRACT_GRAPH_FROM_VALUES = SPARQL_PREFIXES + """
# Value 중심 그래프
SELECT ?value ?value_id ?doc ?doc_id ?key ?key_name
WHERE {
    ?value rdf:type kv:Value .
    ?value kv:valueId ?value_id .

    OPTIONAL {
        ?value kv:extractedFrom ?doc .
        ?doc kv:documentId ?doc_id .
    }

    OPTIONAL {
        ?key kv:hasValue ?value .
        ?key kv:keyName ?key_name .
    }

    FILTER(?value_id IN ($value_ids))
}
"""

GET_NODE_BY_ID = SPARQL_PREFIXES + """
SELECT ?node ?type ?prop ?value
WHERE {
    {
        # Document 타입 조회
        ?node rdf:type kv:Document .
        ?node kv:documentId "$node_id" .
        ?node rdf:type ?type .
        OPTIONAL { ?node ?prop ?value }
    }
    UNION
    {
        # Key 타입 조회
        ?node rdf:type kv:Key .
        ?node kv:keyName "$node_id" .
        ?node rdf:type ?type .
        OPTIONAL { ?node ?prop ?value }
    }
    UNION
    {
        # Value 타입 조회
        ?node rdf:type kv:Value .
        ?node kv:valueId "$node_id" .
        ?node rdf:type ?type .
        OPTIONAL { ?node ?prop ?value }
    }
}
LIMIT 100
"""

# ========== 보조 함수 ==========

def build_doc_ids_filter(doc_ids: list[str]) -> str:
    """Document ID 리스트를 SPARQL FILTER IN 구문으로 변환.

    Args:
        doc_ids: Document ID 리스트

    Returns:
        SPARQL FILTER IN 문자열
    """
    quoted_ids = [f'"{doc_id}"' for doc_id in doc_ids]
    return "(" + ", ".join(quoted_ids) + ")"


def build_value_ids_filter(value_ids: list[str]) -> str:
    """Value ID 리스트를 SPARQL FILTER IN 구문으로 변환.

    Args:
        value_ids: Value ID 리스트

    Returns:
        SPARQL FILTER IN 문자열
    """
    quoted_ids = [f'"{value_id}"' for value_id in value_ids]
    return "(" + ", ".join(quoted_ids) + ")"
