"""Neo4j Cypher 쿼리 상수

Knowledge Graph 작업에 사용되는 Cypher 쿼리를 관리합니다.
"""

# ========== Document 조회 ==========

GET_DOCUMENTS_BY_IDS = """
MATCH (d:Document)
WHERE d.id IN $doc_ids
RETURN d.id as doc_id, d.text as text, d.description as description
"""

GET_DOCUMENT_INFO = """
MATCH (d:Document {id: $doc_id})
RETURN d.text as text, d.description as description
"""

GET_DOCUMENT_NODES = """
MATCH (d:Document)
WHERE d.id IN $doc_ids
RETURN d.id as id, d.description as description
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

GET_RELATED_DOCUMENTS = """
MATCH (source:Document {id: $doc_id})
MATCH (source)-[r:SIMILAR_TO|EXTENDS|CONTRADICTS|PREREQUISITE_OF|IMPLEMENTS|CO_OCCURS]->(target:Document)
RETURN target.id as target_doc_id,
       target.description as target_description,
       type(r) as relation_type,
       r.reason as reason,
       r.confidence as confidence
ORDER BY r.confidence DESC
LIMIT $limit
"""

# ========== 그래프 구조 관련 ==========

GET_GRAPH_CONNECTIONS = """
MATCH (source:Document {id: $source_id})
MATCH (target:Document {id: $target_id})

// 공유 Entity
OPTIONAL MATCH (source)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(target)
WITH source, target, collect(DISTINCT e.name) as shared_entities

// 공유 Topic
OPTIONAL MATCH (source)-[:HAS_TOPIC]->(t:Topic)<-[:HAS_TOPIC]-(target)
WITH source, target, shared_entities, collect(DISTINCT t.name) as shared_topics

// 공유 Category (2-hop)
OPTIONAL MATCH (source)-[:HAS_TOPIC]->(:Topic)-[:BELONGS_TO]->(c:Category)
              <-[:BELONGS_TO]-(:Topic)<-[:HAS_TOPIC]-(target)
WITH shared_entities, shared_topics, count(DISTINCT c) as shared_categories

// Graph Score 계산 (Entity: 2점, Topic: 3점, Category: 2점)
RETURN shared_entities,
       shared_topics,
       size(shared_entities) * 2 + size(shared_topics) * 3 + shared_categories * 2 as graph_score
"""

# 1-hop: Document와 직접 연결된 노드/관계
EXTRACT_GRAPH_1HOP = """
MATCH (d:Document)
WHERE d.id IN $doc_ids
OPTIONAL MATCH path = (d)-[r]-(n:Entity|Topic|Category|Document)
WITH collect(DISTINCT d) + collect(DISTINCT n) as nodes,
     collect(DISTINCT r) as rels
RETURN [node IN nodes | {
    id: COALESCE(node.id, elementId(node)),
    labels: labels(node),
    properties: properties(node)
}] as nodes,
[rel IN rels WHERE rel IS NOT NULL | {
    type: type(rel),
    start: COALESCE(startNode(rel).id, elementId(startNode(rel))),
    end: COALESCE(endNode(rel).id, elementId(endNode(rel))),
    properties: properties(rel)
}] as relationships
"""

# 2-hop: 2단계 연결까지 포함
EXTRACT_GRAPH_2HOP = """
MATCH (d:Document)
WHERE d.id IN $doc_ids
OPTIONAL MATCH path = (d)-[r1]-(n1:Entity|Topic|Category|Document)-[r2]-(n2:Entity|Topic|Category|Document)
WITH collect(DISTINCT d) + collect(DISTINCT n1) + collect(DISTINCT n2) as nodes,
     collect(DISTINCT r1) + collect(DISTINCT r2) as rels
RETURN [node IN nodes | {
    id: COALESCE(node.id, elementId(node)),
    labels: labels(node),
    properties: properties(node)
}] as nodes,
[rel IN rels WHERE rel IS NOT NULL | {
    type: type(rel),
    start: COALESCE(startNode(rel).id, elementId(startNode(rel))),
    end: COALESCE(endNode(rel).id, elementId(endNode(rel))),
    properties: properties(rel)
}] as relationships
"""

# ========== 메타데이터 관련 ==========

CALCULATE_COMMON_METADATA = """
MATCH (d:Document)
WHERE d.id IN $doc_ids
OPTIONAL MATCH (d)-[:HAS_TOPIC]->(t:Topic)-[:BELONGS_TO]->(c:Category)
WITH d, collect(DISTINCT c.name) as categories
OPTIONAL MATCH (d)-[:MENTIONS]->(e:Entity)
WITH d, categories, collect(DISTINCT e.name) as entities
RETURN d.id as doc_id, categories, entities
"""

GET_CATEGORY_RELATIONSHIPS = """
MATCH (d:Document)-[:HAS_TOPIC]->(t:Topic)-[:BELONGS_TO]->(c:Category)
WHERE d.id IN $doc_ids AND c.name IN $category_names
RETURN DISTINCT d.id as doc_id, c.name as category_name
"""

GET_ENTITY_RELATIONSHIPS = """
MATCH (d:Document)-[:MENTIONS]->(e:Entity)
WHERE d.id IN $doc_ids AND e.name IN $entity_names
RETURN DISTINCT d.id as doc_id, e.name as entity_name
"""

# ========== 노드 탐색 관련 ==========

GET_NODE_BY_ENTITY = """
MATCH (n:Entity {name: $name})
RETURN elementId(n) as id, labels(n) as labels, properties(n) as properties
LIMIT 1
"""

GET_NODE_BY_CATEGORY = """
MATCH (n:Category {name: $name})
RETURN elementId(n) as id, labels(n) as labels, properties(n) as properties
LIMIT 1
"""

GET_NODE_BY_ID = """
MATCH (n)
WHERE n.id = $node_id OR elementId(n) = $node_id
RETURN elementId(n) as id, labels(n) as labels, properties(n) as properties
LIMIT 1
"""

EXPLORE_NODE_TEMPLATE = """
{match_clause}
OPTIONAL MATCH (center)-[r]-(connected:Document|Entity|Topic|Category)
WITH center, collect(DISTINCT connected) as connected_nodes, collect(DISTINCT r) as relationships
WITH [center] + connected_nodes as all_nodes, relationships
RETURN [node IN all_nodes WHERE node IS NOT NULL | {{
    id: COALESCE(node.id, node.name, elementId(node)),
    labels: labels(node),
    properties: properties(node)
}}] as nodes,
[rel IN relationships WHERE rel IS NOT NULL | {{
    type: type(rel),
    start: COALESCE(startNode(rel).id, startNode(rel).name, elementId(startNode(rel))),
    end: COALESCE(endNode(rel).id, endNode(rel).name, elementId(endNode(rel))),
    properties: properties(rel)
}}] as relationships
"""
