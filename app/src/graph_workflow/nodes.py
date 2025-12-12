from typing import Literal, Any

from app.exceptions.service import ApplicationError
from app.schemas.graph_state import GraphWorkflowState

# Constants
MAX_RETRY_COUNT = 3
from app.src.graph_workflow.tools import (
    llm_meta_extract_function,
    llm_cypher_generate_function,
    cypher_execution_tool,
    hardcoded_fallback_tool,
    sqlite_insert_tool
)


def llm_meta_node(state: GraphWorkflowState) -> dict:
    """LLM Node: 텍스트에서 메타데이터 추출"""
    result = llm_meta_extract_function(
        text=state.text,
        prompt_name=state.prompt_name,
        prompt_label=state.prompt_label
    )

    # 업데이트할 필드 준비
    update_state = result["update_state"].copy()

    # 재시도 중인 경우 카운트 증가
    if state.llm_error and state.retry_count < MAX_RETRY_COUNT:
        update_state["retry_count"] = state.retry_count + 1

    return update_state


def llm_neo4j_node(state: GraphWorkflowState) -> dict:
    """LLM Node: LLM을 사용해 Neo4j Cypher 쿼리 생성"""
    result = llm_cypher_generate_function(
        metadata=state.metadata,
        text=state.text
    )

    # 업데이트할 필드만 반환
    return result["update_state"]


def tool_cypher_node(state: GraphWorkflowState) -> dict:
    """ToolNode: Cypher 쿼리 실행"""
    result = cypher_execution_tool.invoke({
        "queries": state.cypher_queries
    })

    # 업데이트할 필드 준비
    update_state = result["update_state"].copy()

    # Cypher 실행이 실패하고 재시도 카운트가 MAX_RETRY_COUNT 미만이면 카운트 증가
    if result["status"] == "fail" and state.retry_count < MAX_RETRY_COUNT:
        update_state["retry_count"] = state.retry_count + 1

    return update_state


def tool_hardcoded_node(state: GraphWorkflowState) -> dict:
    """ToolNode: Neo4j 클라이언트의 하드코딩된 메서드 사용"""
    result = hardcoded_fallback_tool.invoke({
        "metadata": state.metadata,
        "text": state.text
    })

    # 업데이트할 필드만 반환
    return result["update_state"]


def tool_sqlite_node(state: GraphWorkflowState, db: Any) -> dict:
    """ToolNode: SQLite에 메타데이터 저장"""
    result = sqlite_insert_tool(
        text=state.text,
        metadata=state.metadata,
        db=db
    )

    # 업데이트할 필드만 반환
    return result["update_state"]


def raise_error_node(state: GraphWorkflowState) -> GraphWorkflowState:
    """에러 발생 노드"""
    errors = []
    if state.llm_error:
        errors.append(f"LLM Error: {state.llm_error}")
    if state.cypher_error:
        errors.append(f"Cypher Error: {state.cypher_error}")
    if state.sqlite_error:
        errors.append(f"SQLite Error: {state.sqlite_error}")
    error_msg = f"Workflow failed: {', '.join(errors)}"
    raise ApplicationError(code=500, message=error_msg, result={"exception": "LangGraph - Flow Fail"})


# 조건부 엣지 함수들
def check_llm_meta_result(state: GraphWorkflowState) -> list:
    """LLM Meta 결과에 따른 라우팅 - 메타데이터가 있으면 두 갈래로"""
    if state.llm_error and state.retry_count < MAX_RETRY_COUNT:
        return ["llm_meta"]  # 재시도는 llm_meta_node에서 카운트 증가 처리
    elif state.metadata:
        return ["llm_neo4j", "tool_sqlite"]  # 메타데이터가 있으면 두 갈래 모두
    else:
        return ["raise_error"]


def check_cypher_execution(state: GraphWorkflowState) -> Literal[
    "end_success", "llm_neo4j", "tool_hardcoded", "raise_error"]:
    """Cypher 실행 결과에 따른 라우팅"""
    if state.execution_status == "success":
        return "end_success"
    elif state.execution_status == "fail":
        # fallback_retry가 True면 이미 하드코딩된 쿼리로 재시도했으므로 에러
        if state.fallback_retry:
            return "raise_error"
        # retry_count가 MAX_RETRY_COUNT 미만이면 llm_neo4j로 재시도
        elif state.retry_count < MAX_RETRY_COUNT:
            return "llm_neo4j"
        # MAX_RETRY_COUNT 재시도 후에는 하드코딩된 쿼리로 폴백
        else:
            return "tool_hardcoded"
    else:
        return "raise_error"
