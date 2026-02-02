"""neo4j-graphrag-python Knowledge Graph 스키마 정의"""
from typing import Dict, Any, List, Tuple

from neo4j_graphrag.experimental.components.schema import NodeType, RelationshipType

# Node 타입 정의
KG_NODE_TYPES: List[NodeType] = [
    NodeType(
        label="Document",
        description="A document with extracted metadata",
        properties=[
            {"name": "id", "type": "STRING", "required": True},
            {"name": "text", "type": "STRING"},
            {"name": "description", "type": "STRING"}
        ]
    ),
    NodeType(
        label="Topic",
        description="A topic or subject area",
        properties=[
            {"name": "name", "type": "STRING", "required": True}
        ]
    ),
    NodeType(
        label="Tag",
        description="A keyword or tag",
        properties=[
            {"name": "name", "type": "STRING", "required": True}
        ]
    )
]

# Relationship 타입 정의
KG_RELATIONSHIP_TYPES: List[RelationshipType] = [
    # ========== Document 간 의미적 관계 ==========
    RelationshipType(
        label="SIMILAR_TO",
        description="Documents discuss similar concepts, topics, or share comparable content",
        properties=[
            {"name": "reason", "type": "STRING"},
            {"name": "confidence", "type": "FLOAT"}
        ]
    ),
    RelationshipType(
        label="CONTRADICTS",
        description="Documents present conflicting or opposing views on the same subject",
        properties=[
            {"name": "reason", "type": "STRING"},
            {"name": "confidence", "type": "FLOAT"}
        ]
    ),
    RelationshipType(
        label="EXTENDS",
        description="Document extends, elaborates, or develops concepts from another document",
        properties=[
            {"name": "reason", "type": "STRING"},
            {"name": "confidence", "type": "FLOAT"}
        ]
    ),
    RelationshipType(
        label="PREREQUISITE_OF",
        description="Document contains prerequisite knowledge required to understand another document",
        properties=[
            {"name": "reason", "type": "STRING"},
            {"name": "confidence", "type": "FLOAT"}
        ]
    ),
    RelationshipType(
        label="IMPLEMENTS",
        description="Document implements or applies theories, concepts, or plans from another document",
        properties=[
            {"name": "reason", "type": "STRING"},
            {"name": "confidence", "type": "FLOAT"}
        ]
    ),
    RelationshipType(
        label="CO_OCCURS",
        description="Documents frequently appear together in graph structure (shared topics, tags, connections)",
        properties=[
            {"name": "reason", "type": "STRING"},
            {"name": "confidence", "type": "FLOAT"},
            {"name": "graph_score", "type": "FLOAT"}
        ]
    ),

    # ========== Document와 메타데이터 간 관계 ==========
    RelationshipType(
        label="HAS_TOPIC",
        description="Document belongs to a topic"
    ),
    RelationshipType(
        label="HAS_TAG",
        description="Document has a tag"
    )
]

# 허용 패턴 정의
KG_PATTERNS: List[Tuple[str, str, str]] = [
    # Document 간 관계
    ("Document", "SIMILAR_TO", "Document"),
    ("Document", "CONTRADICTS", "Document"),
    ("Document", "EXTENDS", "Document"),
    ("Document", "PREREQUISITE_OF", "Document"),
    ("Document", "IMPLEMENTS", "Document"),
    ("Document", "CO_OCCURS", "Document"),
    # 메타데이터 관계
    ("Document", "HAS_TOPIC", "Topic"),
    ("Document", "HAS_TAG", "Tag")
]

# SimpleKGPipeline에서 사용할 전체 스키마
KG_SCHEMA: Dict[str, Any] = {
    "node_types": KG_NODE_TYPES,
    "relationship_types": KG_RELATIONSHIP_TYPES,
    "patterns": KG_PATTERNS
}
