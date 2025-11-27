from typing import Dict, Any, Union
import logging
from langfuse.model import PromptClient
from openai import pydantic_function_tool

from app.config import langfuse_handler
from app.dependencies import chatopenai_client, get_langfuse_prompt
from app.schemas.meta import MetadataInfo
from app.exceptions.base import ApplicationError


# Pydantic 모델 기반 OpenAI Tool 정의
tool_definition = pydantic_function_tool(
    MetadataInfo,
    name="extract_metadata_info",
    description="주어진 데이터에서 메타데이터 정보를 추출합니다."
)


class LLMResponse:
    """LLM 응답 처리 클래스 - 메타데이터 추출용"""

    def __init__(self):
        """Init 매서드"""
        pass

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

    def extract_metadata_by_llm(self, data_text: str, prompt_name: str, prompt_label: str) -> Dict[str, Any]:
        """메타데이터 추출 모델 응답 받아오기"""
        # 프롬프트 가져오기(캐싱)
        prompt = self._get_prompt(prompt_name, prompt_label)
        prompt_variables = {"data_text": data_text}
        compiled_prompt = self._compile_prompt(prompt, **prompt_variables)
        response = self._invoke_llm_with_prompt(compiled_prompt)
        # 응답 처리
        return self._process_llm_response(response)

    def _get_prompt(self, prompt_name: str, prompt_label: str) -> Union[PromptClient, str]:
        """메타데이터 추출용 프롬프트를 가져오기. - Langfuse 실패 시 폴백 프롬프트를 반환."""
        prompt = get_langfuse_prompt(prompt_name=prompt_name, prompt_label=prompt_label)
        if prompt is None:
            prompt = self.META_EXTRACT_FALLBACK_PROMPT
        return prompt

    @staticmethod
    def _extract_message_from_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """tool_call에서 args 추출하기"""
        try:
            response_message = tool_call["args"]
            logging.debug(f"Extracted metadata: {response_message}")
            return response_message
        except IndexError as e:
            raise ApplicationError(code=400, message="Tool call has no arguments: 'args'", result={"exception": str(e)})

    def _process_llm_response(self, response: Any) -> Dict[str, Any]:
        """LLM 응답 처리하여 메타데이터 추출"""
        try:
            tool_call = response.tool_calls[0]
            return self._extract_message_from_tool_call(tool_call)
        except (IndexError, AttributeError) as e:
            raise ApplicationError(code=400, message=f"No tool calls in llm_response: {response.content}",
                                   result={"exception": str(e)})

    @staticmethod
    def _compile_prompt(prompt, **variables):
        """프롬프트 컴파일 처리 - 랭퓨즈와 폴백 프롬프트 처리 대응"""
        if hasattr(prompt, 'compile'):
            return prompt.compile(**variables)
        return prompt.format(**variables)

    @staticmethod
    def _invoke_llm_with_prompt(compiled_prompt: str) -> Any:
        """LLM에 컴파일된 프롬프트로 요청 보내기"""
        llm = chatopenai_client.bind_tools([tool_definition])
        return llm.invoke(
            compiled_prompt,
            config={
                "callbacks": [langfuse_handler],
                "metadata": {"langfuse_tags": ["generate_metadata"]},
            },
        )
