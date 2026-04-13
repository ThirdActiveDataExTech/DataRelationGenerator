import time
from typing import Dict, Any, Union

import logging
from langfuse.model import PromptClient
from openai import APIError

from app.config import langfuse_handler
from app.constants import (
    DOCUMENT_RELATION_INFERENCE_PROMPT_NAME,
    DOCUMENT_RELATION_INFERENCE_PROMPT_LABEL
)
from app.dependencies import chatopenai_client, get_langfuse_prompt
from app.exceptions.base import ApplicationError
from app.schemas.meta import (
    document_relation_tool,
    document_relevance_tool,
    key_value_extraction_tool,
    cluster_normalization_tool,
    query_key_inference_tool
)
from app.schemas.schema_analysis import schema_analysis_tool
from app.src.meta_extract.fallback_prompts import FallbackPrompts


class LLMResponse:
    """LLM 응답 처리 클래스 - 메타데이터 추출 및 Neo4j 쿼리 생성용"""

    PROMPT_CONFIG = {
        "document_relation_inference": (
            DOCUMENT_RELATION_INFERENCE_PROMPT_NAME, DOCUMENT_RELATION_INFERENCE_PROMPT_LABEL),
        "document_relevance_verification": ("document_relevance_verification", "production"),
        "key_value_extract": ("key_value_extract", "production"),
        "cluster_normalize": ("cluster_normalize", "production"),
        "schema_analysis": ("schema_analysis", "production"),
        "query_key_inference": ("query_key_inference", "production"),
    }

    def __init__(self, task: str):
        """Init 매서드"""
        self.task = task
        config = self.PROMPT_CONFIG.get(task)
        if config is None:
            raise ValueError(f"Unknown task: {task}")
        self.prompt_name, self.prompt_label = config

    def generate_llm_response(self, variables: Any) -> Dict[str, Any] | str:
        """통합 LLM 응답 처리"""
        prompt = self._get_prompt()
        compiled_prompt = self._compile_prompt(prompt, **variables)
        response = self._invoke_llm_unified(compiled_prompt)
        return self._process_llm_response(response)

    @staticmethod
    def _extract_message_from_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """tool_call에서 args 추출하기"""
        try:
            response_message = tool_call["args"]
            logging.debug(f"Extracted metadata: {response_message}")
            return response_message
        except KeyError:
            raise KeyError("tool_call에 'args' 키가 없습니다")

    def _process_llm_response(self, response: Any) -> Dict[str, Any]:
        """LLM 응답 처리하여 메타데이터 추출"""
        try:
            tool_call = response.tool_calls[0]
            return self._extract_message_from_tool_call(tool_call)
        except (IndexError, AttributeError, KeyError) as e:
            raise ApplicationError(code=400, message=f"No tool calls in llm_response: {str(e)}",
                                   result={"exception": f"llm response: {response.content}"})

    @staticmethod
    def _compile_prompt(prompt, **variables):
        """프롬프트 컴파일 처리 - 랭퓨즈와 폴백 프롬프트 처리 대응"""
        if hasattr(prompt, 'compile'):
            return prompt.compile(**variables)
        return prompt.format(**variables)

    def _get_prompt(self) -> Union[PromptClient, str]:
        """통합 프롬프트 가져오기 - Langfuse 실패 시 폴백 프롬프트 반환"""
        prompt = get_langfuse_prompt(prompt_name=self.prompt_name, prompt_label=self.prompt_label)
        if prompt is None:
            logging.warning(f"LLM Response: {self.task} 프롬프트 폴백")
            prompt = FallbackPrompts.get_prompt(self.task)
        return prompt

    def _invoke_llm_unified(self, compiled_prompt: str) -> Any:
        """통합 LLM 호출"""
        if self.task == "document_relation_inference":
            tools = [document_relation_tool]
        elif self.task == "document_relevance_verification":
            tools = [document_relevance_tool]
        elif self.task == "key_value_extract":
            tools = [key_value_extraction_tool]
        elif self.task == "cluster_normalize":
            tools = [cluster_normalization_tool]
        elif self.task == "schema_analysis":
            tools = [schema_analysis_tool]
        elif self.task == "query_key_inference":
            tools = [query_key_inference_tool]
        else:
            tools = []

        llm = chatopenai_client.bind_tools(tools)
        for attempt in range(2):
            try:
                return llm.invoke(
                    compiled_prompt,
                    config={
                        "callbacks": [langfuse_handler],
                        "metadata": {"langfuse_tags": [self.task]},
                    },
                )
            except APIError as e:
                if ("503" in str(e) or "Too many connections" in str(e)) and attempt == 0:
                    logging.warning(f"LLM 503 에러, 2초 후 재시도")
                    time.sleep(2)
                    continue
                raise
