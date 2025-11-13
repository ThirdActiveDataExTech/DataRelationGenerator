from app.schemas.meta import MetadataInfo
from openai import pydantic_function_tool
from app.config import settings
from app.dependencies import openai_client

too_definition = pydantic_function_tool(
    MetadataInfo,
    name="extract_metadata_info",
    description="주어진 데이터에서 메타데이터 정보를 추출합니다."
)

# 데이터 관련 보다는 데이터를 설명할 이라고 적으면 좋을듯..
# langfuse 추가하기
system_message = """
  # 데이터 메타정보 추출 프롬프트

  다음 데이터를 분석하여 메타데이터를 JSON 형식으로 추출해주세요.

  ## 데이터:
  {input_data}

  ## 추출 요구사항:
  1. **tags**: 데이터와 관련된 핵심 키워드/태그들 (3-10개)
  2. **topics**: 데이터의 주요 주제/분야들 (2-5개)  
  3. **description**: 데이터에 대한 간략하고 명확한 설명 (1-2문장)

  ## 출력 형식:
  ```json
  {
    "tags": ["태그1", "태그2", "태그3"],
    "topics": ["주제1", "주제2"],
    "description": "데이터에 대한 간략한 설명"
  }

  가이드라인:

  - tags는 구체적이고 검색 가능한 키워드로 작성
  - topics는 데이터가 속하는 주요 분야/카테고리로 작성
  - description은 데이터의 핵심 내용을 요약
  - 모든 텍스트는 한국어로 작성
  - 빈 문자열이나 중복값은 제외
  - JSON 형식을 정확히 준수

  데이터를 분석하여 위 형식에 맞는 JSON만 응답해주세요.
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
                    "content": f"데이터: {data_text}"
                }
            ],
            temperature=0.0,
            functions=[too_definition],
        )
        # print(f"Metadata Extraction LLM Response: {llm_response}")
        return llm_response
    except Exception as e:
        raise e
