"""Neo4j Cypher 쿼리 상수

Knowledge Graph 작업에 사용되는 Cypher 쿼리를 관리합니다.
"""

# ========== Document 조회 ==========

GET_DOCUMENTS_BY_IDS = """
MATCH (d:Document)
WHERE d.id IN $doc_ids
RETURN d.id as doc_id, d.raw_text as text, d.source_file as description
"""

GET_DOCUMENT_INFO = """
MATCH (d:Document {id: $doc_id})
RETURN d.raw_text as text, d.source_file as description
"""

GET_DOCUMENT_NODES = """
MATCH (d:Document)
WHERE d.id IN $doc_ids
RETURN d.id as id, d.source_file as description
"""

# ========== 검색 관련 ==========

SEARCH_BY_GRAPH_EMBEDDING = """
MATCH (source:Document {id: $doc_id})
WHERE source.graph_embedding IS NOT NULL
CALL db.index.vector.queryNodes('graph_embedding_index', $top_k, source.graph_embedding)
YIELD node, score
WHERE node.id <> $doc_id
RETURN node.id as doc_id, score
"""

# ========== 그래프 구조 관련 (Key-Value) ==========

EXTRACT_GRAPH_1HOP = """
MATCH (d:Document)
WHERE d.id IN $doc_ids
OPTIONAL MATCH (v:Value)-[r1:EXTRACTED_FROM]->(d)
OPTIONAL MATCH (k:Key)-[r2:HAS_VALUE]->(v)
WITH collect(DISTINCT d) + collect(DISTINCT v) + collect(DISTINCT k) as all_nodes,
     collect(DISTINCT r1) + collect(DISTINCT r2) as all_rels
RETURN [node IN all_nodes WHERE node IS NOT NULL | {
    id: COALESCE(node.id, node.name, elementId(node)),
    labels: labels(node),
    properties: properties(node)
}] as nodes,
[rel IN all_rels WHERE rel IS NOT NULL | {
    type: type(rel),
    start: COALESCE(startNode(rel).id, startNode(rel).name, elementId(startNode(rel))),
    end: COALESCE(endNode(rel).id, endNode(rel).name, elementId(endNode(rel))),
    properties: properties(rel)
}] as relationships
"""

EXTRACT_GRAPH_2HOP = """
MATCH (d:Document)
WHERE d.id IN $doc_ids
OPTIONAL MATCH path = (d)-[r1]-(n1:Key|Value|Centrality|Document)-[r2]-(n2:Key|Value|Centrality|Document)
WITH collect(DISTINCT d) + collect(DISTINCT n1) + collect(DISTINCT n2) as nodes,
     collect(DISTINCT r1) + collect(DISTINCT r2) as rels
RETURN [node IN nodes | {
    id: COALESCE(node.id, node.name, elementId(node)),
    labels: labels(node),
    properties: properties(node)
}] as nodes,
[rel IN rels WHERE rel IS NOT NULL | {
    type: type(rel),
    start: COALESCE(startNode(rel).id, startNode(rel).name, elementId(startNode(rel))),
    end: COALESCE(endNode(rel).id, endNode(rel).name, elementId(endNode(rel))),
    properties: properties(rel)
}] as relationships
"""

GET_NODE_BY_ID = """
MATCH (n)
WHERE n.id = $node_id OR elementId(n) = $node_id
RETURN elementId(n) as id, labels(n) as labels, properties(n) as properties
LIMIT 1
"""
