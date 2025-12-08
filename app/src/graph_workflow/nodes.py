from app.schemas.graph_state import GraphWorkflowState
from app.src.graph_workflow.tools import extract_metadata_llm_tool, save_metadata_sqlite_tool


# TODO: state 상세하게 나누기.

def extract_metadata_node(state: GraphWorkflowState) -> dict:
    """메타데이터 추출 노드"""

    metadata = extract_metadata_llm_tool.invoke({
        "text": state.input_text,
        "prompt_name": state.prompt_name,
        "prompt_label": state.prompt_label
    })
    return {
        "metadata": metadata,
        "status": "metadata_extracted"
    }


def save_to_sqlite_node(state: GraphWorkflowState) -> dict:
    """SQLite 저장 노드"""

    # TODO: db 객체를 state에서 받아오거나 의존성 주입 방식 개선 필요
    record_id = save_metadata_sqlite_tool.invoke({
        "text": state.input_text,
        "metadata": state.metadata,
        "db": None  # TODO: 임시로 None, 실제로는 db 객체 전달 필요 어디서 받아오게 하지
    })
    return {
        "sqlite_record_id": record_id,
        "status": "saved_to_sqlite"
    }


def create_document_node(state: GraphWorkflowState) -> dict:
    """Document 노드 생성 쿼리 준비"""


def create_topic_nodes(state: GraphWorkflowState) -> dict:
    """Topic 노드들 생성 쿼리 준비"""


def create_tag_nodes(state: GraphWorkflowState) -> dict:
    """Tag 노드들 생성 쿼리 준비"""


def load_to_neo4j_node(state: GraphWorkflowState) -> dict:
    """Neo4j에 모든 쿼리 실행"""
