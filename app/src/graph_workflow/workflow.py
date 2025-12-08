from langgraph.graph import StateGraph, START, END

from app.schemas.graph_state import GraphWorkflowState
from app.src.graph_workflow.nodes import (
    extract_metadata_node,
    save_to_sqlite_node
)


def create_graph_workflow() -> StateGraph:
    """Graph DB 적재 워크플로우 생성"""

    # StateGraph 초기화
    workflow = StateGraph(GraphWorkflowState)

    # 노드 추가
    workflow.add_node("extract_metadata", extract_metadata_node)
    workflow.add_node("save_to_sqlite", save_to_sqlite_node)

    # 엣지 연결
    workflow.add_edge(START, "extract_metadata")
    workflow.add_edge("extract_metadata", "save_to_sqlite")

    workflow.add_edge("save_to_sqlite", END)

    return workflow.compile()


def run_graph_workflow(input_text: str, prompt_name: str, prompt_label: str, db) -> dict:
    """워크플로우 실행"""

    # 초기 상태 설정
    initial_state = {
        "input_text": input_text,
        "prompt_name": prompt_name,
        "prompt_label": prompt_label,
        "metadata": {},
        "sqlite_record_id": 0,
        "status": "started"
    }

    # 워크플로우 생성 및 실행
    graph = create_graph_workflow()
    result = graph.invoke(initial_state)

    return result
