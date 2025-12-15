import asyncio
from typing import Optional

import logging
from sqlalchemy.orm import Session

from app.src.graph_workflow.agent import MetadataAgent, AgentState


async def create_agent_workflow(db: Optional[Session] = None) -> MetadataAgent:
    """Agent 기반 워크플로우 생성"""
    logging.info("Creating Agent-based workflow")

    # Agent 인스턴스 생성
    agent = MetadataAgent(db=db)

    return agent


async def run_agent_workflow(text: str, prompt_name: str, prompt_label: str,
                             db: Optional[Session] = None) -> AgentState:
    """Agent 기반 워크플로우 실행"""
    logging.info(f"Starting Agent workflow for text: {text[:50]}...")

    # Agent 생성
    agent = await create_agent_workflow(db)

    # Agent 워크플로우 실행
    result = await agent.run_agent_workflow(text)

    logging.info("Agent workflow completed successfully")
    return {
        "origin_text": result.get("origin_text", ""),
        "extracted_metadata": result.get("extracted_metadata"),
        "generated_queries": result.get("generated_queries"),
        "sqlite_result": result.get("sqlite_result"),
        "metadata_extract_retry": result.get("metadata_extract_retry", 0),
        "cypher_retry": result.get("cypher_retry", 0),
        "last_failed_step": result.get("last_failed_step")
    }


def run_graph_workflow(text: str, prompt_name: str, prompt_label: str, db) -> AgentState:
    """기존 동기 인터페이스 호환성을 위한 래퍼 함수"""
    logging.info("Running workflow via sync wrapper")

    # 비동기 함수를 동기적으로 실행
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(
        run_agent_workflow(text, prompt_name, prompt_label, db)
    )
