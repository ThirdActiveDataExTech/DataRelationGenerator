from typing import Dict, Any, Union

import logging
from langfuse.model import PromptClient
from openai import InternalServerError

from app.config import langfuse_handler
from app.dependencies import chatopenai_client, get_langfuse_prompt
from app.exceptions.base import ApplicationError
from app.schemas.meta import metadata_tool, cypher_tool


class LLMResponse:
    """LLM 응답 처리 클래스 - 메타데이터 추출 및 Neo4j 쿼리 생성용"""

    def __init__(self, task: str):
        """Init 매서드"""
        self.task = task
        self.fallback_prompt = None
        if task == "meta_extract":
            self.fallback_prompt = self.META_EXTRACT_FALLBACK_PROMPT
        else:
            self.fallback_prompt = self.CYPHER_GENERATION_FALLBACK_PROMPT

    META_EXTRACT_FALLBACK_PROMPT: str = """
    **반드시 tool call을 사용하여 응답할것.**
      당신은 데이터 분석 전문가입니다.
      다음 데이터를 분석하여 메타데이터를 JSON 형식으로 추출해주세요.
      가이드라인:
    
      - tags: 데이터를 설명할 수 있는 핵심 키워드/태그들 (3-10개) 구체적이고 검색 가능한 키워드로 작성
      - topics는 데이터가 속하는 주요 분야/카테고리로 작성. 데이터의 주요 주제/분야들 (2-5개)  
      - description은  데이터에 대한 간략하고 명확한 설명
      - 빈 문자열이나 중복값은 제외
      - 반드시 function tool 호출로 응답
      ---
    데이터를 분석하여 위 형식에 맞는 JSON만 응답해주세요.
    데이터: {data_text}
    """

    CYPHER_GENERATION_FALLBACK_PROMPT: str = """
    **반드시 generate_cypher_query tool을 사용하여 응답할것.**
    당신은 Neo4j Cypher 전문가입니다.
    다음 메타데이터와 텍스트를 기반으로 단일 Cypher 쿼리를 생성해주세요.
    
    요구사항:
    1. 하나의 완전한 쿼리로 Document 노드와 모든 관계를 생성
    2. UNWIND를 사용하여 topics와 tags를 효율적으로 처리  
    3. 문자열은 작은따옴표로 감싸고 특수문자는 이스케이프
    4. apoc.create.uuid()를 사용하여 고유 ID 생성
    5. 반드시 generate_cypher_query function tool을 호출하여 응답
    
    메타데이터: {metadata}
    텍스트: {text}
    
    예시:
    CREATE (d:Document) 
    SET d.id = 'doc_' + apoc.create.uuid(),
        d.text = '예시 텍스트 내용',
        d.description = '예시 설명',
        d.created_at = datetime()
    WITH d
    UNWIND ['주제1', '주제2'] AS topic_name
    MERGE (t:Topic) ON CREATE SET t.name = topic_name
    CREATE (d)-[:HAS_TOPIC]->(t)
    WITH d  
    UNWIND ['태그1', '태그2'] AS tag_name
    MERGE (tag:Tag) ON CREATE SET tag.name = tag_name
    CREATE (d)-[:HAS_TAG]->(tag)
    
    위 형식을 따라 실제 메타데이터 값으로 완전한 단일 쿼리를 생성하세요.
    다른 설명이나 여러 쿼리는 반환하지 마세요.
    """

    def generate_llm_response(self, prompt_name: str, prompt_label: str, variables: Any) -> Dict[str, Any] | str:
        """통합 LLM 응답 처리"""
        try:
            prompt = self._get_prompt(prompt_name, prompt_label)
            compiled_prompt = self._compile_prompt(prompt, **variables)
            response = self._invoke_llm_unified(compiled_prompt)
            return self._process_llm_response(response)
        except InternalServerError:
            raise

    @staticmethod
    def _extract_message_from_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """tool_call에서 args 추출하기"""
        try:
            response_message = tool_call["args"]
            logging.debug(f"Extracted metadata: {response_message}")
            raise IndexError
            return response_message
        except IndexError as e:
            raise e

    def _process_llm_response(self, response: Any) -> Dict[str, Any]:
        """LLM 응답 처리하여 메타데이터 추출"""
        try:
            tool_call = response.tool_calls[0]
            return self._extract_message_from_tool_call(tool_call)
        except (IndexError, AttributeError) as e:
            if self.task == "cypher_generate":
                raise e
            raise ApplicationError(code=400, message=f"No tool calls in llm_response: {response.content}",
                                   result={"exception": str(e)})

    @staticmethod
    def _compile_prompt(prompt, **variables):
        """프롬프트 컴파일 처리 - 랭퓨즈와 폴백 프롬프트 처리 대응"""
        if hasattr(prompt, 'compile'):
            return prompt.compile(**variables)
        return prompt.format(**variables)

    def _get_prompt(self, prompt_name: str, prompt_label: str) -> Union[PromptClient, str]:
        """통합 프롬프트 가져오기 - Langfuse 실패 시 폴백 프롬프트 반환"""
        prompt = get_langfuse_prompt(prompt_name=prompt_name, prompt_label=prompt_label)
        if prompt is None:
            logging.warning(f"LLM Response: {self.task} 프롬프트 폴백")
            prompt = self.fallback_prompt
        return prompt

    def _invoke_llm_unified(self, compiled_prompt: str) -> Any:
        """통합 LLM 호출"""
        llm = chatopenai_client.bind_tools([metadata_tool, cypher_tool])
        return llm.invoke(
            compiled_prompt,
            config={
                "callbacks": [langfuse_handler],
                "metadata": {"langfuse_tags": [self.task]},
            },
        )
