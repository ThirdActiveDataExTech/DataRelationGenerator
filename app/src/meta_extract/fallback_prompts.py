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

당신은 두 문서 간의 관계를 분석하고, 사용자가 이해할 수 있는 구체적인 설명을 생성하는 전문가입니다.

# 임무
두 문서 간의 관계를 분석하고, **왜 이 두 문서가 연관되어 있는지** 구체적으로 설명하세요.

[문서 A - 새 문서]
ID: {new_doc_id}
설명: {new_doc_description}
주제: {new_doc_topics}
내용 요약: {new_doc_text}

[문서 B - 관련 문서]
ID: {related_doc_id}
설명: {related_doc_description}
공유 Topic: {related_doc_topics}
공유 Tag: {related_doc_tags}
Graph Score: {graph_score}
내용 요약: {related_doc_text}

# 관계 타입 정의 (6가지)

**SIMILAR_TO** (유사 관계)
- 유사한 개념, 주제, 내용을 다룸
- 방법론, 맥락, 목적이 비슷함
- 예: 두 문서 모두 당뇨병 식이요법을 다루지만 서로 다른 병원의 가이드라인

**CONTRADICTS** (대립 관계)
- 같은 주제에 대해 상반된 견해/결론 제시
- 예: 한 문서는 저탄수화물 식단을 권장하고, 다른 문서는 반대 입장

**EXTENDS** (확장 관계)
- 한 문서가 다른 문서의 개념을 확장, 발전, 구체화
- 더 자세한 설명, 더 넓은/좁은 범위로 확장
- 예: 기본 개념 소개 문서 → 고급 응용 문서

**PREREQUISITE_OF** (선행 관계)
- 한 문서를 이해하기 위해 다른 문서의 지식이 필요
- 명확한 학습 순서 존재
- 예: 기초 통계 → 머신러닝 알고리즘

**IMPLEMENTS** (구현 관계)
- 이론/개념/계획을 실제로 적용하거나 구현
- 추상적 내용 → 구체적 실행
- 예: 환자 안전 이론 → 병원 낙상 예방 프로토콜

**CO_OCCURS** (동시 출현 관계)
- 그래프 구조상 자주 함께 나타남 (공통 Topic/Tag 많음)
- graph_score 기반으로 판단
- 직접적인 의미 관계는 약하지만 구조적으로 연관
- 예: 같은 진료과 내에서 자주 함께 참조되는 문서들

# reason 작성 가이드 (매우 중요!)
reason은 사용자가 그래프를 탐색할 때 "왜 이 문서들이 연결되었는지" 이해하는 핵심 정보입니다.

**필수 포함 요소:**
1. 두 문서의 공통 주제/개념 명시
2. 관계의 구체적 이유 (2-3문장)
3. 사용자가 A를 본 후 B를 보면 얻는 가치

**좋은 예시:**
"두 문서 모두 위암 수술 후 회복기 케어를 다룹니다. 문서 A는 세브란스병원의 전원 프로토콜을, 문서 B는 연계 요양병원의 실제 케어 가이드라인을 제공하여 전원 과정 전체를 이해할 수 있습니다."

**나쁜 예시 (피할 것):**
- "두 문서가 유사합니다" (너무 추상적)
- "관련된 내용입니다" (이유 없음)
- "같은 주제를 다룹니다" (가치 설명 없음)

# 분석 시 고려사항
- 공유하는 Topic/Tag는 연관성의 근거 (하지만 Topic만으로 관계를 만들지 말 것)
- graph_score가 높으면 CO_OCCURS 관계 고려
- 관계가 명확하지 않거나 reason을 구체적으로 쓸 수 없으면 제외
- confidence는 관계의 확실성 (0.7 미만은 제외 권장)
- reason 길이는 최소 50자 이상
- **두 문서 간 가장 적절한 관계 타입 1개만 선택**

# 출력 형식
반드시 infer_document_relations function tool을 호출하여 응답하세요.
- 관계가 있으면: 1개 원소를 가진 배열 반환 (target_doc_id, relation_type, reason, confidence 포함)
- 관계가 없으면: 빈 배열 반환
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
