"""LLM function tool 생성 유틸.

openai.pydantic_function_tool 은 Field(ge=/le=) 제약을 JSON Schema 의
minimum/maximum 으로 내보내는데, Bedrock Claude 의 tool 스키마 검증이
number 타입의 minimum/maximum 을 지원하지 않아 400 에러를 반환한다
(litellm 경유 시 "tools.0.custom: For 'number' type, properties maximum,
minimum are not supported").
"""

from typing import Any, Optional, Type

from openai import pydantic_function_tool as _openai_pydantic_function_tool
from pydantic import BaseModel

_UNSUPPORTED_NUMBER_KEYS = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}


def _strip_number_bounds(node: Any) -> None:
    """JSON Schema 트리에서 Bedrock 미지원 number 제약 키를 제거 (in-place)."""
    if isinstance(node, dict):
        for key in _UNSUPPORTED_NUMBER_KEYS & node.keys():
            del node[key]
        for value in node.values():
            _strip_number_bounds(value)
    elif isinstance(node, list):
        for item in node:
            _strip_number_bounds(item)


def pydantic_function_tool(
    model: Type[BaseModel],
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Any:
    """openai.pydantic_function_tool + Bedrock 호환 스키마 정리."""
    tool = _openai_pydantic_function_tool(model, name=name, description=description)
    _strip_number_bounds(tool["function"].get("parameters", {}))
    return tool
