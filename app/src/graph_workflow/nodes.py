from typing import Literal
from app.exceptions.service import ApplicationError

from app.schemas.graph_state import GraphWorkflowState
from app.src.graph_workflow.tools import (
    llm_meta_extract_tool,
    llm_cypher_generate_tool,
    cypher_execution_tool,
    hardcoded_fallback_tool,
    sqlite_insert_tool
)


def llm_meta_node(state: GraphWorkflowState) -> dict:
    """LLMNode: 텍스트에서 메타데이터 추출"""
    result = llm_meta_extract_tool.invoke({
        "text": state.text,
        "prompt_name": state.prompt_name,
        "prompt_label": state.prompt_label
    })

    # 업데이트할 필드만 반환
    return result["update_state"]


def llm_neo4j_node(state: GraphWorkflowState) -> dict:
    """LLMNode: LLM을 사용해 Neo4j Cypher 쿼리 생성"""
    result = llm_cypher_generate_tool.invoke({
        "metadata": state.metadata,
        "text": state.text
    })

    # 업데이트할 필드만 반환
    return result["update_state"]


def tool_cypher_node(state: GraphWorkflowState) -> dict:
    """ToolNode: Cypher 쿼리 실행"""
    result = cypher_execution_tool.invoke({
        "queries": state.cypher_queries
    })

    # 업데이트할 필드만 반환
    return result["update_state"]


def tool_hardcoded_node(state: GraphWorkflowState) -> dict:
    """ToolNode: Neo4j 클라이언트의 하드코딩된 메서드 사용"""
    result = hardcoded_fallback_tool.invoke({
        "metadata": state.metadata,
        "text": state.text
    })

    # 업데이트할 필드만 반환
    return result["update_state"]


def tool_sqlite_node(state: GraphWorkflowState, db) -> dict:
    """ToolNode: SQLite에 메타데이터 저장"""
    result = sqlite_insert_tool.invoke({
        "text": state.text,
        "metadata": state.metadata or {},
        "db": db
    })

    # 업데이트할 필드만 반환
    return result["update_state"]


def raise_error_node(state: GraphWorkflowState) -> GraphWorkflowState:
    """에러 발생 노드"""
    error_msg = f"Workflow failed: LLM Error: {state.llm_error}, Cypher Error: {state.cypher_error}, SQLite Error: {state.sqlite_error}"
    raise ApplicationError(code=500, message=error_msg, result={"exception": "LangGraph - Flow Fail"})


# 조건부 엣지 함수들
def check_llm_meta_result(state: GraphWorkflowState) -> list:
    """LLM Meta 결과에 따른 라우팅 - 메타데이터가 있으면 두 갈래로"""
    if state.llm_error and state.retry_count < 3:
        state.retry_count = state.retry_count + 1
        return ["llm_meta"]
    elif state.metadata:
        return ["llm_neo4j", "tool_sqlite"]  # 메타데이터가 있으면 두 갈래 모두
    else:
        return ["raise_error"]


def check_cypher_execution(state: GraphWorkflowState) -> Literal[
    "end_success", "tool_hardcoded", "raise_error"]:
    """Cypher 실행 결과에 따른 라우팅"""
    if state.execution_status == "success":
        return "end_success"
    elif state.execution_status == "fail":
        return "tool_hardcoded"  # 실패시 폴백으로
    else:
        return "raise_error"
