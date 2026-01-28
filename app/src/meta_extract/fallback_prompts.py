from typing import Dict


class FallbackPrompts:
    """Container for fallback prompt templates used across LLM tasks."""

    META_EXTRACT: str = """
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

    CYPHER_GENERATE: str = """
    **반드시 generate_cypher_query tool을 사용하여 응답할것.**
    당신은 Neo4j Cypher 전문가입니다.
    다음 메타데이터와 텍스트를 기반으로 단일 Cypher 쿼리를 생성해주세요.

    요구사항:
    1. 하나의 완전한 쿼리로 Document 노드와 모든 관계를 생성
    2. UNWIND를 사용하여 topics와 tags를 효율적으로 처리  
    3. 문자열은 작은따옴표로 감싸고 특수문자는 이스케이프
    4. apoc.create.uuid()를 사용하여 고유 ID 생성
    5. 반드시 generate_cypher_query function tool을 호출하여 응답

    메타데이터: {metadata}
    텍스트: {text}

    예시:
    CREATE (d:Document) 
    SET d.id = 'doc_' + apoc.create.uuid(),
        d.text = '예시 텍스트 내용',
        d.description = '예시 설명',
        d.created_at = datetime()
    WITH d
    UNWIND ['주제1', '주제2'] AS topic_name
    MERGE (t:Topic) ON CREATE SET t.name = topic_name
    CREATE (d)-[:HAS_TOPIC]->(t)
    WITH d  
    UNWIND ['태그1', '태그2'] AS tag_name
    MERGE (tag:Tag) ON CREATE SET tag.name = tag_name
    CREATE (d)-[:HAS_TAG]->(tag)

    위 형식을 따라 실제 메타데이터 값으로 완전한 단일 쿼리를 생성하세요.
    다른 설명이나 여러 쿼리는 반환하지 마세요.
    """

    CATEGORY_GENERATE: str = """다음 topic 목록을 보고, 의미적으로 묶일 수 있는 상위 Category를 생성하라.

    조건:
    - Category는 일반명사로 표현
    - Topic → Category 관계는 BELONGS_TO
    - 동의어는 SAME_AS로 연결
    - 하위 개념은 SUBTYPE_OF로 연결
    - 의미적으로 유사하면 SIMILAR_TO로 연결

    기존 Category 목록 (가능하면 이 중 하나에 우선 매핑):
    {existing_categories}

    Topic 목록:
    {topics}

    출력 형식 (JSON):
    {{
      "categories": [
        {{
          "name": "카테고리명",
          "topics": [
            {{"topic": "토픽명", "relation": "BELONGS_TO"}}
          ]
        }}
      ]
    }}
    """

    DOCUMENT_RELATION_INFERENCE: str = """**반드시 infer_document_relations tool을 사용하여 응답할 것.**

당신은 문서 간 의미적 관계를 분석하는 전문가입니다.

새 문서와 그래프 구조 기반으로 찾은 관련 문서들 간의 관계를 분석하세요.

[새 문서]
ID: {new_doc_id}
설명: {new_doc_description}
주제: {new_doc_topics}
내용: {new_doc_text}

[관련 문서들]
{related_docs_info}

관계 타입 정의:
- EXTENDS: 새 문서가 기존 문서의 개념/내용을 확장
- USES: 새 문서가 기존 문서의 기술/지식/방법론을 활용
- PREREQUISITE_OF: 기존 문서가 새 문서 이해를 위한 선행 지식
- COMPARES_WITH: 유사한 주제를 다른 관점에서 비교
- APPLIES_TO: 새 문서의 개념이 기존 문서의 상황/사례에 적용
- SAME_DOMAIN: 같은 분야/도메인을 다룸 (공유 Topic/Category 많음)

분석 시 고려사항:
- 문서 내용의 의미적 관계
- 공유하는 Topic과 Tag 정보 (shared_topics, shared_tags)
- 그래프 점수 (graph_score) - 구조적 연관성
- 관계가 명확하지 않으면 포함하지 않음
- confidence는 0.0~1.0 사이 값

반드시 infer_document_relations function tool을 호출하여 응답하세요.
관계가 없으면 빈 배열을 반환하세요.
"""

    @classmethod
    def get_prompt(cls, task: str) -> str:
        """Get fallback prompt by task name."""
        prompt_map: Dict[str, str] = {
            "meta_extract": cls.META_EXTRACT,
            "cypher_generate": cls.CYPHER_GENERATE,
            "category_generate": cls.CATEGORY_GENERATE,
            "document_relation_inference": cls.DOCUMENT_RELATION_INFERENCE,
        }

        if task not in prompt_map:
            raise ValueError(
                f"Unknown task: {task}. Available tasks: {', '.join(prompt_map.keys())}"
            )

        return prompt_map[task]