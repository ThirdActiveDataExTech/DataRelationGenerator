# 적절한 위치 다시 생각해보기
from langfuse.model import PromptClient

META_EXTRACT_FALLBACK_PROMPT: str = """
**반드시 tool call을 사용하여 응답할것.**
  당신은 데이터 분석 전문가입니다.
  다음 데이터를 분석하여 메타데이터를 JSON 형식으로 추출해주세요.
  가이드라인:

  - tags: 데이터를 설명할 수 있는 핵심 키워드/태그들 (3-10개) 구체적이고 검색 가능한 키워드로 작성
  - topics는 데이터가 속하는 주요 분야/카테고리로 작성. 데이터의 주요 주제/분야들 (2-5개)  
  - description은  데이터에 대한 간략하고 명확한 설명
  - 빈 문자열이나 중복값은 제외
  - 반드시 fuction tool 호출로 응답
  ---
데이터를 분석하여 위 형식에 맞는 JSON만 응답해주세요.
데이터: {data_text}
"""