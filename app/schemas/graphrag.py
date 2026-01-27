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
    # Document 간 의미적 관계 (LLM이 판단)
    RelationshipType(
        label="EXTENDS",
        description="Document extends or builds upon concepts from another document",
        properties=[
            {"name": "reason", "type": "STRING"},
            {"name": "confidence", "type": "FLOAT"}
        ]
    ),
    RelationshipType(
        label="USES",
        description="Document uses or references another document",
        properties=[
            {"name": "reason", "type": "STRING"},
            {"name": "confidence", "type": "FLOAT"}
        ]
    ),
    RelationshipType(
        label="PREREQUISITE_OF",
        description="Document is a prerequisite for understanding another document",
        properties=[
            {"name": "reason", "type": "STRING"},
            {"name": "confidence", "type": "FLOAT"}
        ]
    ),
    RelationshipType(
        label="COMPARES_WITH",
        description="Document compares or contrasts with another document",
        properties=[
            {"name": "reason", "type": "STRING"},
            {"name": "confidence", "type": "FLOAT"}
        ]
    ),
    RelationshipType(
        label="APPLIES_TO",
        description="Concepts in document apply to situations in another document",
        properties=[
            {"name": "reason", "type": "STRING"},
            {"name": "confidence", "type": "FLOAT"}
        ]
    ),
    RelationshipType(
        label="SAME_DOMAIN",
        description="Documents discuss the same domain or field",
        properties=[
            {"name": "reason", "type": "STRING"},
            {"name": "confidence", "type": "FLOAT"}
        ]
    ),
    # Document와 메타데이터 간 관계
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
    ("Document", "EXTENDS", "Document"),
    ("Document", "USES", "Document"),
    ("Document", "PREREQUISITE_OF", "Document"),
    ("Document", "COMPARES_WITH", "Document"),
    ("Document", "APPLIES_TO", "Document"),
    ("Document", "SAME_DOMAIN", "Document"),
    ("Document", "HAS_TOPIC", "Topic"),
    ("Document", "HAS_TAG", "Tag")
]


# SimpleKGPipeline에서 사용할 전체 스키마
KG_SCHEMA: Dict[str, Any] = {
    "node_types": KG_NODE_TYPES,
    "relationship_types": KG_RELATIONSHIP_TYPES,
    "patterns": KG_PATTERNS
}
