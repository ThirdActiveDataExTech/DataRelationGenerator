import json
from typing import Optional

import logging
from langchain_core.messages import AIMessage
from langchain_core.messages import SystemMessage
from sqlalchemy.orm import Session

from app.clients.agent_client import llm_client
from app.constants import METADATA_AGENT_SYSTEM_PROMPT
from app.schemas.graph_state import AgentState
from app.src.agent.tools import (
    metadata_extract_tool,
    cypher_generate_tool,
    hardcoded_fallback_tool
)


class GraphComponents:
    """메타데이터 추출 및 Graph DB 적재 - LangGraph Components (Nodes & Routers)"""

    def __init__(self, mcp_tools: Optional[list] = None):
        """Init 매서드"""
        self.llm = llm_client.primary_model
        #
        # Agent가 동적으로 선택할 수 있는 tool 정의
        self.tools = [
            metadata_extract_tool,
            cypher_generate_tool,
            hardcoded_fallback_tool
        ]
        if mcp_tools:
            self.tools.extend(mcp_tools)

        # LLM에 도구 바인딩
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        logging.debug(f"MetadataAgent initialized with {len(self.tools)} tools")

    @staticmethod
    def _inject_context_to_tool_calls(state: AgentState) -> dict:
        """[Node] Tool call의 arguments에 state context 주입"""
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None

        if not last_message or not getattr(last_message, "tool_calls", None):
            logging.debug("[inject_context] No tool calls found, passing through")
            return {}

        tool_calls = getattr(last_message, "tool_calls", [])
        logging.debug(f"[inject_context] Processing {len(tool_calls)} tool call(s)")

        origin_text = state.get("origin_text", "")
        extracted_metadata = state.get("extracted_metadata")
        generated_queries = state.get("generated_queries")

        for tool_call in tool_calls:
            tool_name = tool_call["name"]

            if tool_name == "metadata_extract_tool":
                tool_call["args"] = {
                    "text": origin_text
                }
            elif tool_name in ("cypher_generate_tool", "hardcoded_fallback_tool"):
                tool_call["args"] = {
                    "text": origin_text,
                    "metadata": extracted_metadata or {}
                }
            elif tool_name == "cypher_execution_tool":
                tool_call["args"] = {
                    "queries": generated_queries or []
                }
            elif tool_name == "write_neo4j_cypher":
                queries_list = generated_queries or []
                combined_query = ";\n".join(queries_list) if queries_list else ""
                tool_call["args"] = {
                    "query": combined_query
                }
        return {}

    @staticmethod
    def _parse_tool_result(msg_content: str | dict) -> dict:
        """Tool 결과를 dict로 파싱"""
        return json.loads(msg_content) if isinstance(msg_content, str) else msg_content

    @staticmethod
    def _handle_success_result(result: dict, state_updates: dict) -> None:
        """성공한 Tool 결과 처리"""
        if "metadata" in result:
            state_updates["extracted_metadata"] = result["metadata"]
            state_updates["metadata_extract_retry"] = 0

        if "queries" in result:
            state_updates["generated_queries"] = result["queries"]

        if "message" in result and "Neo4j" in result["message"]:
            state_updates["neo4j_result"] = result

    @staticmethod
    def _handle_error_result(result: dict, state: AgentState, state_updates: dict) -> None:
        """실패한 Tool 결과 처리"""
        error_message = result.get("message", "")

        if "메타데이터 추출 실패" in error_message:
            state_updates["last_failed_step"] = "metadata_extract"
            state_updates["metadata_extract_retry"] = state.get("metadata_extract_retry", 0) + 1

        elif "Cypher 쿼리 생성 실패" in error_message or "Neo4j 쿼리 실행 실패" in error_message:
            state_updates["last_failed_step"] = "cypher"
            state_updates["cypher_retry"] = state.get("cypher_retry", 0) + 1

    @staticmethod
    def _extract_state_from_tool_results(state: AgentState) -> dict:
        """[Node] Tool 실행 결과에서 state 업데이트 및 실패 감지"""
        messages = state.get("messages", [])
        state_updates = {}

        for msg in reversed(messages):
            if not (hasattr(msg, "type") and msg.type == "tool"):
                continue

            try:
                result = GraphComponents._parse_tool_result(msg.content)

                if not isinstance(result, dict):
                    continue

                if result.get("status") == "success":
                    GraphComponents._handle_success_result(result, state_updates)
                elif result.get("status") == "error":
                    GraphComponents._handle_error_result(result, state, state_updates)

                break

            except Exception as e:
                logging.warning(f"[extract_state] Failed to parse tool result: {e}")

        return state_updates

    @staticmethod
    def _route_after_extract(state: AgentState) -> str:
        """extract_state 후 라우팅 결정"""
        last_failed = state.get("last_failed_step")
        extracted_metadata = state.get("extracted_metadata")
        neo4j_result = state.get("neo4j_result")

        # metadata_extract 실패 시 retry 로직
        if last_failed == "metadata_extract":
            retry_count = state.get("metadata_extract_retry", 0)
            logging.info(f"[route_after_extract] metadata_extract failed, retry_count={retry_count}")
            return "reasoning_node" if retry_count < 2 else "end"

        # cypher 실패 시 retry 로직
        if last_failed == "cypher":
            retry_count = state.get("cypher_retry", 0)
            fallback_used = state.get("fallback_used", False)

            if retry_count == 1:
                logging.info("[route_after_extract] Routing to reasoning_node. cypher 생성 retry")
                return "reasoning_node"
            elif retry_count > 1 and not fallback_used:
                logging.warning("[route_after_extract] Cypher generation/execution failed, routing to fallback")
                return "fallback"
            elif fallback_used:
                logging.error("[route_after_extract] Fallback already used but still failing, ending workflow")
                return "end"

        # 모든 작업 완료 확인
        if extracted_metadata and neo4j_result:
            logging.info("[route_after_extract] All tasks completed, ending workflow")
            return "end"

        logging.info("[route_after_extract] Continuing to next step")
        return "reasoning_node"

    async def _reasoning_node(self, state: AgentState) -> dict:
        """[Node] reasoning node - Agent의 판단 / 추론 / 의사결정을 하며 다음 행동을 결정하는 단계"""
        logging.info("[reasoning_node] Starting agent decision making")

        messages = state["messages"]

        if not messages or messages[0].type != "system":
            sys_msg = SystemMessage(content=METADATA_AGENT_SYSTEM_PROMPT)
            messages = [sys_msg, *messages]

        result = await self.llm_with_tools.ainvoke(messages)

        return {"messages": [result]}

    @staticmethod
    async def _fallback_node(state: AgentState) -> dict:
        """[Node] Hardcoded fallback을 강제로 호출하는 노드"""
        logging.info("[fallback_node] Forcing hardcoded_fallback_tool call")

        fallback_call = AIMessage(
            content="",
            tool_calls=[{
                "name": "hardcoded_fallback_tool",
                "args": {},
                "id": "fallback_call"
            }]
        )

        return {"messages": [fallback_call], "fallback_used": True}
