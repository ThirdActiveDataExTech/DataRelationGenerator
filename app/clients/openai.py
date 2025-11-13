from langfuse.openai import OpenAI  # pyright: ignore

from app.config import settings


def get_client(api_key: str = settings.CLAUDE_API_KEY, base_url=settings.LITE_LLM_BASE_URL) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)
