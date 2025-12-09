from langgraph.graph import StateGraph, START, END
from functools import partial

from app.schemas.graph_state import GraphWorkflowState
from app.src.graph_workflow.nodes import (
    llm_meta_node,
    llm_neo4j_node,
    tool_cypher_node,
    tool_hardcoded_node,
    tool_sqlite_node,
    raise_error_node,
    check_llm_meta_result,
    check_cypher_execution
)


def create_graph_workflow(db) -> StateGraph:
    """Graph DB 적재 워크플로우 생성"""

    # StateGraph 초기화
    workflow = StateGraph(GraphWorkflowState)

    # db 의존성을 주입하기 위해 partial 사용
    llm_meta_with_db = partial(llm_meta_node, db=db)
    tool_sqlite_with_db = partial(tool_sqlite_node, db=db)

    # 노드 추가
    workflow.add_node("llm_meta", llm_meta_with_db)
    workflow.add_node("llm_neo4j", llm_neo4j_node)
    workflow.add_node("tool_cypher", tool_cypher_node)
    workflow.add_node("tool_hardcoded", tool_hardcoded_node)
    workflow.add_node("tool_sqlite", tool_sqlite_with_db)
    workflow.add_node("raise_error", raise_error_node)

    # START에서 llm_meta로
    workflow.add_edge(START, "llm_meta")

    # llm_meta에서 조건부 라우팅
    workflow.add_conditional_edges(
        "llm_meta",
        check_llm_meta_result,
        {
            "llm_neo4j": "llm_neo4j",
            "tool_sqlite": "tool_sqlite",
            "raise_error": "raise_error",
            "retry_meta": "llm_meta"
        }
    )

    # tool_sqlite에서 END로 (SQLite만 저장하는 경로)
    workflow.add_edge("tool_sqlite", END)

    # tool_sqlite_neo4j에서 tool_cypher로 (SQLite 저장 후 Neo4j 처리)
    workflow.add_edge("tool_sqlite_neo4j", "tool_cypher")

    # tool_cypher에서 조건부 라우팅
    workflow.add_conditional_edges(
        "tool_cypher",
        check_cypher_execution,
        {
            "end_success": END,
            "retry_neo4j": "llm_neo4j",
            "tool_hardcoded": "tool_hardcoded",
            "raise_error": "raise_error"
        }
    )

    # tool_hardcoded에서 다시 tool_cypher로
    workflow.add_edge("tool_hardcoded", "tool_cypher")

    # raise_error에서 END로
    workflow.add_edge("raise_error", END)

    return workflow.compile()


def run_graph_workflow(text: str, prompt_name: str, prompt_label: str, db) -> GraphWorkflowState:
    """워크플로우 실행"""

    # 초기 상태 설정
    initial_state = GraphWorkflowState(
        text=text,
        prompt_name=prompt_name,
        prompt_label=prompt_label,
        metadata={},
        sqlite_status="pending",
        sqlite_error="",
        cypher_queries=[],
        execution_status="pending",
        cypher_error="",
        llm_error="",
        retry_count=0,
        fallback_retry=False,
        messages=[]
    )

    # 워크플로우 생성 및 실행
    graph = create_graph_workflow(db)
    result = graph.invoke(initial_state)

    return result
