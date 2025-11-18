from app.dependencies import chatopenai_client, get_langfuse_prompt
import logging
from typing import Dict, Any, Union
from app.config import langfuse_handler
from app.schemas.meta import MetadataInfo
from openai import pydantic_function_tool
from langfuse.model import PromptClient
from .prompts import META_EXTRACT_FALLBACK_PROMPT

META_EXTRACT_PROMPT = None

# Pydantic 모델 기반 OpenAI Tool 정의
tool_definition = pydantic_function_tool(
    MetadataInfo,
    name="extract_metadata_info",
    description="주어진 데이터에서 메타데이터 정보를 추출합니다."
)


# langfuse handler설정을 위해 langchain_openai 사용 변경
def compile_prompt(prompt, **variables):
    """프롬프트 컴파일 처리 - 랭퓨즈와 폴백 프롬프트 처리 대응"""
    if hasattr(prompt, 'compile'):
        return prompt.compile(**variables)
    return prompt.format(**variables)

def generate_metadata(data_text: str, prompt_name: str, prompt_label: str) -> Dict[str, Any]:
    """메타데이터 추출 모델 응답 받아오기"""
    try:
        prompt = get_meta_extract_prompt(prompt_name=prompt_name, prompt_label=prompt_label)
        prompt_variables = {"data_text": data_text}
        compiled_prompt = compile_prompt(prompt, **prompt_variables)
        llm = chatopenai_client.bind_tools([tool_definition])
        response = llm.invoke(
            compiled_prompt,
            config={
                "callbacks": [langfuse_handler],
                "metadata": {"langfuse_tags": ["generate_metadata"]},
            },
        )
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_call = response.tool_calls[0]
            # 구조: { ...  "args": {...}, "id": "...", "type": "tool_call" }
            if "args" in tool_call:
                response_message = tool_call["args"]
                logging.debug(f"Extracted metadata: {response_message}")
            else:
                raise ValueError("No function in tool_call")
        else:
            raise ValueError("No tool calls in llm_response")
        return response_message
    except Exception as e:
        logging.error(f"Metadata extraction failed: {e}")
        raise e


def get_meta_extract_prompt(prompt_name: str, prompt_label: str) -> Union[PromptClient, str, None]:
    """메타데이터 추출 프롬프트 캐싱 및 반환"""
    global META_EXTRACT_PROMPT
    if META_EXTRACT_PROMPT is None:
        try:
            META_EXTRACT_PROMPT = get_langfuse_prompt(prompt_name=prompt_name, prompt_label=prompt_label)
        except Exception as e:
            # 폴백 프롬프트 반환
            META_EXTRACT_PROMPT = META_EXTRACT_FALLBACK_PROMPT
            logging.warning(f"langfuse prompt fail - return fallback prompt: {e}")
    return META_EXTRACT_PROMPT
