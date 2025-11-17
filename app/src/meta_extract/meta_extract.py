from app.dependencies import langfuse_client, get_chatopenai_client
import logging
from typing import Dict, Any
from app.config import langfuse_handler


# langfuse handler설정을 위해 langchain_openai 사용 변경
def generate_metadata(data_text: str) -> Dict[str, Any]:
    """메타데이터 추출 모델 응답 받아오기"""
    try:
        if langfuse_client is None:
            raise ValueError("Langfuse client is not initialized")
        prompt = langfuse_client.get_prompt("generate_metadata", label="production")
        prompt_variables = {"data_text": data_text}
        compiled_prompt = prompt.compile(**prompt_variables)
        # 후에 get client로 빼야함
        llm = get_chatopenai_client()
        response = llm.invoke(
            compiled_prompt,
            config={
                "callbacks": [langfuse_handler],
                "metadata": {"langfuse_tags": ["generate_llm_response_v2"]},
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
