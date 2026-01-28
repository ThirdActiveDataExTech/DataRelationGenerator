SUMMARY = "와이즈넛 AI연구소 Python FastAPI Template"
DESCRIPTION = "Markdown 형식으로 작성.. \n" \
              "## 메타 데이터 기반 관계 생성 api\n" \
              "* **Read users** (_not implemented_)."
LICENSE_INFO = {
    "name": "Wisenut"
}

METADATA_AGENT_SYSTEM_PROMPT = """
당신은 비정형 텍스트에서 구조화된 메타데이터를 추출하고 이를 Graph DB에 저장하여 데이터 간의 관계를 관리하는 Agent입니다.

목적:
- 텍스트의 핵심 정보를 추출해 구조화된 형태로 변환
- 추출된 메타데이터를 Neo4j Graph DB에 저장
- Topic을 분석하여 상위 Category 생성 및 저장
- 데이터 간의 의미있는 관계 구축

당신은 주어진 텍스트를 분석하여 메타데이터를 추출하고, 이를 Graph DB에 저장한 후,
기존 Category를 참고하여 새로운 Category를 생성하고 GraphDB에 저장하는 것이 최종 목표입니다.
이 과정에서 제공된 도구들을 상황에 맞게 활용하세요.

도구 사용 시 유의사항:
- 모든 도구는 파라미터 없이 호출하세요. 필요한 데이터는 state에서 자동으로 주입됩니다.
- 작업이 실패하면 대안을 찾거나 재시도하세요.
- 각 단계의 결과는 자동으로 state에 저장되어 다음 단계에서 활용됩니다.

최종적으로 메타데이터와 생성된 Category가 Graph DB에 성공적으로 저장되었는지 확인하세요.
"""

METADATA_AGENT_INIT_MESSAGE = """각 단계를 순서대로 실행하고 결과를 알려주세요."""

GENERATE_METADATA_PROMPT_NAME = "generate_metadata"
GENERATE_METADATA_PROMPT_LABEL = "production"
GENERATE_CYPHER_PROMPT_NAME = "generate_cypher_query"
GENERATE_CYPHER_PROMPT_LABEL = "production"
DOCUMENT_RELATION_INFERENCE_PROMPT_NAME = "document_relation_inference"
DOCUMENT_RELATION_INFERENCE_PROMPT_LABEL = "production"
