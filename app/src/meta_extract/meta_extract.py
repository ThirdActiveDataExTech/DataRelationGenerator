from app.schemas.meta import MetadataInfo
from openai import pydantic_function_tool
from app.config import settings
from app.dependencies import openai_client
import logging, json

tool_definition = pydantic_function_tool(
    MetadataInfo,
    name="extract_metadata_info",
    description="주어진 데이터에서 메타데이터 정보를 추출합니다."
)

# 데이터 관련 보다는 데이터를 설명할 이라고 적으면 좋을듯..
# langfuse 추가하기
system_message = """
  당신은 데이터 분석 전문가입니다.
  다음 데이터를 분석하여 메타데이터를 JSON 형식으로 추출해주세요.
  가이드라인:

  - tags: 데이터와 관련된 핵심 키워드/태그들 (3-10개) 구체적이고 검색 가능한 키워드로 작성
  - topics는 데이터가 속하는 주요 분야/카테고리로 작성. 데이터의 주요 주제/분야들 (2-5개)  
  - description은  데이터에 대한 간략하고 명확한 설명
  - 빈 문자열이나 중복값은 제외
  - 반드시 fuction tool 호출로 응답
"""


def generate_metadata(data_text: str, model: str = settings.CLAUDE_MODEL_NAME):
    """메타데이터 추출 모델 응답 받아오기"""
    try:

        llm_response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_message

                },
                {
                    "role": "user",
                    "content": f"데이터를 분석하여 위 형식에 맞는 JSON만 응답해주세요.\n데이터: {data_text}"
                }
            ],
            temperature=0.0,
            tools=[tool_definition]
        )

        response_message = json.loads(llm_response.choices[0].message.tool_calls[0].function.arguments)
        logging.info(f"Metadata extraction successful. \n\n {response_message}")
        return response_message
    except Exception as e:
        logging.error(f"Metadata extraction failed: {e}")
        raise e
