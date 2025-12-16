import logging
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition, ToolNode

from app.config import langfuse_handler
from app.constants import METADATA_AGENT_INIT_MESSAGE
from app.schemas.graph_state import AgentState
from app.src.agent.graph_components import GraphComponents
from app.src.agent.tools import (
    metadata_extract_tool,
    cypher_generate_tool,
    cypher_execution_tool,
    hardcoded_fallback_tool
)


class MetadataAgent:
    """메타데이터 추출 및 Graph DB 적재를 위한 Agent"""

    def __init__(self):
        """Init 매서드"""
        self.graph_components = GraphComponents()
        self.tools = [
            metadata_extract_tool,
            cypher_generate_tool,
            cypher_execution_tool,
            hardcoded_fallback_tool
        ]

    def build_agent_graph(self) -> StateGraph:
        """Agent 그래프 빌드"""
        builder = StateGraph(AgentState)

        # 노드 추가
        builder.add_node("reasoning_node", self.graph_components._reasoning_node)
        builder.add_node("fallback", self.graph_components._fallback_node)
        builder.add_node("inject_context", self.graph_components._inject_context_to_tool_calls)
        builder.add_node("tool_node", ToolNode(self.tools))
        builder.add_node("extract_state", self.graph_components._extract_state_from_tool_results)

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
            self.graph_components._route_after_extract,
            {
                "reasoning_node": "reasoning_node",
                "fallback": "fallback",
                "end": END
            }
        )

        return builder

    async def run_agent_workflow(self, origin_text: str) -> dict:
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
            "extracted_metadata": None,
            "generated_queries": None,
            "neo4j_result": None,
            "metadata_extract_retry": 0,
            "cypher_retry": 0,
            "last_failed_step": None
        })

        logging.info("Agent workflow completed")
        
        return {
            "origin_text": result.get("origin_text"),
            "extracted_metadata": result.get("extracted_metadata"),
            "generated_queries": result.get("generated_queries"),
            "neo4j_result": result.get("neo4j_result"),
            "metadata_extract_retry": result.get("metadata_extract_retry"),
            "cypher_retry": result.get("cypher_retry"),
            "last_failed_step": result.get("last_failed_step")
        }
