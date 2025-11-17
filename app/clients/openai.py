from langfuse.openai import OpenAI  # pyright: ignore
from openai import pydantic_function_tool
from app.config import settings
from app.schemas.meta import MetadataInfo
from langchain_openai import ChatOpenAI

tool_definition = pydantic_function_tool(
    MetadataInfo,
    name="extract_metadata_info",
    description="주어진 데이터에서 메타데이터 정보를 추출합니다."
)


def get_client(api_key: str = settings.CLAUDE_API_KEY, base_url=settings.LITE_LLM_BASE_URL) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


def get_chat_client(
    model: str = settings.CLAUDE_MODEL_NAME,
    base_url: str = settings.LITE_LLM_BASE_URL,
    temperature: float = 0.0,
    api_key: str = settings.CLAUDE_API_KEY,
):
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
    ).bind_tools([tool_definition])
