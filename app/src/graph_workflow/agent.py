import json
from typing import Optional

import logging
from langchain_core.messages import AIMessage
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition, ToolNode
from sqlalchemy.orm import Session

from app.clients.agent_client import llm_client
from app.config import langfuse_handler
from app.constants import METADATA_AGENT_SYSTEM_PROMPT, METADATA_AGENT_INIT_MESSAGE
from app.schemas.graph_state import AgentState
from app.src.graph_workflow.tools import (
    metadata_extract_tool,
    cypher_generate_tool,
    cypher_execution_tool,
    hardcoded_fallback_tool,
    sqlite_insert_tool
)


class MetadataAgent:
    """메타데이터 추출 및 Graph DB 적재를 위한 Agent"""

    def __init__(self, db: Optional[Session] = None):
        """Init 매서드"""
        self.db = db
        self.llm = llm_client.primary_model

        # Agent가 동적으로 선택할 수 있는 tool 정의
        self.tools = [
            metadata_extract_tool,
            cypher_generate_tool,
            cypher_execution_tool,
            hardcoded_fallback_tool,
            sqlite_insert_tool
        ]

        # LLM에 도구 바인딩
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        logging.debug(f"MetadataAgent initialized with {len(self.tools)} tools")

    @staticmethod
    def _inject_context_to_tool_calls(state: AgentState) -> AgentState:
        """[Node] Tool call의 arguments에 state context 주입"""
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None

        if not last_message or not hasattr(last_message, "tool_calls"):
            logging.debug("[inject_context] No tool calls found, passing through")
            return state

        logging.debug(f"[inject_context] Processing {len(last_message.tool_calls)} tool call(s)")

        origin_text = state.get("origin_text", "")
        prompt_name = state.get("prompt_name", "generate_metadata")
        prompt_label = state.get("prompt_label", "production")
        extracted_metadata = state.get("extracted_metadata")
        generated_queries = state.get("generated_queries")

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})

            if tool_name == "metadata_extract_tool":
                tool_call["args"] = {
                    "text": origin_text,
                    "prompt_name": prompt_name,
                    "prompt_label": prompt_label
                }
            elif tool_name == "sqlite_insert_tool":
                tool_call["args"] = {
                    "text": origin_text,
                    "metadata": extracted_metadata or {}
                }
            elif tool_name == "cypher_generate_tool":
                tool_call["args"] = {
                    "text": origin_text,
                    "metadata": extracted_metadata or {}
                }
            elif tool_name == "cypher_execution_tool":
                tool_call["args"] = {
                    "queries": generated_queries or []
                }
            elif tool_name == "hardcoded_fallback_tool":
                tool_call["args"] = {
                    "text": origin_text,
                    "metadata": extracted_metadata or {}
                }
        return state

    @staticmethod
    def _parse_tool_result(msg_content: str | dict) -> dict:
        """Tool 결과를 dict로 파싱"""
        if isinstance(msg_content, str):
            return json.loads(msg_content)
        return msg_content

    @staticmethod
    def _handle_success_result(result: dict, state_updates: dict) -> None:
        """성공한 Tool 결과 처리"""
        if "metadata" in result:
            state_updates["extracted_metadata"] = result["metadata"]
            state_updates["metadata_extract_retry"] = 0

        if "queries" in result:
            state_updates["generated_queries"] = result["queries"]

        if "message" in result and "SQLite" in result["message"]:
            state_updates["sqlite_result"] = result

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
    def _extract_state_from_tool_results(state: AgentState) -> AgentState:
        """[Node] Tool 실행 결과에서 state 업데이트 및 실패 감지"""
        messages = state.get("messages", [])
        state_updates = {}

        for msg in reversed(messages):
            if not (hasattr(msg, "type") and msg.type == "tool"):
                continue

            try:
                result = MetadataAgent._parse_tool_result(msg.content)

                if not isinstance(result, dict):
                    continue

                if result.get("status") == "success":
                    MetadataAgent._handle_success_result(result, state_updates)
                elif result.get("status") == "error":
                    MetadataAgent._handle_error_result(result, state, state_updates)

            except Exception as e:
                logging.warning(f"[extract_state] Failed to parse tool result: {e}")

        return state_updates

    @staticmethod
    def _route_after_extract(state: AgentState) -> str:
        """extract_state 후 라우팅 결정"""
        last_failed = state.get("last_failed_step")

        # metadata_extract 실패 시 retry 로직
        if last_failed == "metadata_extract":
            retry_count = state.get("metadata_extract_retry", 0)
            if retry_count < 2:
                return "reasoning_node"  # LLM에게 다시 시도하도록
            else:
                return "end"

        # cypher 실패 시 retry 로직
        if last_failed == "cypher":
            retry_count = state.get("cypher_retry", 0)
            if retry_count == 1:
                logging.info(f"[route_after_extract] Routing to reasoning_node. cypher 생성 retry")
                return "reasoning_node"  # LLM에게 다시 generate 요청
            elif retry_count > 1:
                logging.warning(f"[route_after_extract] Cypher generation/execution failed, routing to fallback")
                return "fallback"  # hardcoded fallback 사용

        # 정상 진행
        return "reasoning_node"

    async def _reasoning_node(self, state: AgentState) -> AgentState:
        """[Node] reasoning node - Agent의 판단 / 추론 / 의사결정을 하며 다음 행동을 결정하는 단계"""
        logging.info("[reasoning_node] Starting agent decision making")

        messages = state["messages"]

        if not messages or messages[0].type != "system":
            sys_msg = SystemMessage(content=METADATA_AGENT_SYSTEM_PROMPT)
            messages = [sys_msg, *messages]

        result = await self.llm_with_tools.ainvoke(messages)

        return {"messages": [result]}

    async def _fallback_node(self, state: AgentState) -> AgentState:
        """[Node] Hardcoded fallback을 강제로 호출하는 노드"""
        logging.info("[fallback_node] Forcing hardcoded_fallback_tool call")

        # AIMessage with tool_call for hardcoded_fallback_tool
        fallback_call = AIMessage(
            content="",
            tool_calls=[{
                "name": "hardcoded_fallback_tool",
                "args": {},
                "id": "fallback_call"
            }]
        )

        return {"messages": [fallback_call]}

    def build_agent_graph(self) -> StateGraph:
        """Agent 그래프 빌드"""
        builder = StateGraph(AgentState)

        # 노드 추가
        builder.add_node("reasoning_node", self._reasoning_node)
        builder.add_node("fallback", self._fallback_node)
        builder.add_node("inject_context", self._inject_context_to_tool_calls)
        builder.add_node("tool_node", ToolNode(self.tools))
        builder.add_node("extract_state", self._extract_state_from_tool_results)

        # 엣지 구성
        builder.add_edge(START, "reasoning_node")

        # LLM 결정 후 라우팅
        builder.add_conditional_edges(
            "reasoning_node",
            tools_condition,
            {
                "tools": "inject_context",
                END: END
            }
        )

        # Fallback도 inject_context로
        builder.add_edge("fallback", "inject_context")
        builder.add_edge("inject_context", "tool_node")
        builder.add_edge("tool_node", "extract_state")

        # extract_state 후 retry 로직에 따라 라우팅
        builder.add_conditional_edges(
            "extract_state",
            self._route_after_extract,
            {
                "reasoning_node": "reasoning_node",
                "fallback": "fallback",
                "end": END
            }
        )

        return builder

    async def run_agent_workflow(self, origin_text: str) -> AgentState:
        """Agent 워크플로우 실행"""
        # 초기 메시지 구성 - 원본 텍스트는 state로 전달하고 LLM에게는 작업 지시만
        initial_message = {
            "role": "user",
            "content": METADATA_AGENT_INIT_MESSAGE
        }

        # Agent 그래프 빌드 및 실행
        graph = self.build_agent_graph().compile().with_config({"callbacks": [langfuse_handler]})

        # origin_text와 초기 state를 포함하여 전달
        result = await graph.ainvoke({
            "messages": [initial_message],
            "origin_text": origin_text,
            "prompt_name": "generate_metadata",
            "prompt_label": "production",
            "extracted_metadata": None,
            "generated_queries": None,
            "sqlite_result": None,
            "metadata_extract_retry": 0,
            "cypher_retry": 0,
            "last_failed_step": None
        })

        logging.info("Agent workflow completed")

        # AgentState 결과 반환
        return result
