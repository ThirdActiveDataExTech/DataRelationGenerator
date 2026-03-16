from typing import Dict


class FallbackPrompts:
    """Container for fallback prompt templates used across LLM tasks."""

    META_EXTRACT: str = """
**반드시 tool call을 사용하여 응답할것.**

당신은 데이터 분석 전문가입니다.
다음 데이터를 분석하여 메타데이터를 JSON 형식으로 추출해주세요.

# 추출 항목

## 1. entities (엔티티)
문서에서 언급되는 **구체적인 개체**를 추출하세요.
- **인명**: 사람 이름, 직책 (예: "김철수 교수", "이영희 간호사")
- **기관명**: 조직, 병원, 부서 (예: "세브란스병원", "외과", "응급실")
- **지명**: 지역, 위치 (예: "서울", "강남구")
- **개념**: 질병명, 치료법, 절차명, 이론 (예: "위암", "내시경 수술", "회복기 케어")
- **제품/서비스명**: 장비, 시스템, 프로그램 (예: "MRI 장비", "원격 진료")

**기준**:
- 3-10개 추출
- 명사형으로 추출 (동사/형용사 제외)
- **원문 그대로 추출** (정규화하지 않음)
- 중요도 순으로 정렬

## 2. topics (주제)
데이터가 속하는 **추상적 개념이나 분야**를 추출하세요.
- 2-5개 추출
- 일반명사/추상명사 (예: "의료 연계", "수술 후 관리", "환자 안전")
- 검색 가능한 키워드

## 3. description (설명)
데이터를 한 문장으로 요약하세요 (최대 200자).

---
데이터: {data_text}
    """

    CYPHER_GENERATE: str = """
**반드시 generate_cypher_query tool을 사용하여 응답할것.**

당신은 Neo4j Cypher 전문가입니다.
다음 메타데이터와 텍스트를 기반으로 단일 Cypher 쿼리를 생성해주세요.

요구사항:
1. 하나의 완전한 쿼리로 Document 노드와 모든 관계를 생성
2. UNWIND를 사용하여 topics와 entities를 효율적으로 처리
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
MERGE (t:Topic {name: topic_name})
MERGE (d)-[:HAS_TOPIC]->(t)
WITH d
UNWIND ['엔티티1', '엔티티2'] AS entity_name
MERGE (e:Entity {name: entity_name})
MERGE (d)-[:MENTIONS]->(e)

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
주요 필드(Keys): {new_doc_keys}
내용 요약: {new_doc_text}

[문서 B - 관련 문서]
ID: {related_doc_id}
설명: {related_doc_description}
유사도 점수: {similarity_score}
내용 요약: {related_doc_text}

# 관계 타입 정의 (6가지)

**SIMILAR_TO** (유사 관계)
- 유사한 필드 구성과 내용을 가짐
- 같은 도메인/카테고리에 속하며 목적이 비슷함
- 예: 두 문서 모두 요양병원 정보를 담고 있으며 지역, 병상 수, 진료 과목 등 유사한 필드를 가짐

**CONTRADICTS** (대립 관계)
- 같은 필드에 대해 상반된 값이나 상태를 가짐
- 예: 한 문서는 "영업중" 상태이고, 다른 문서는 "폐업" 상태인 같은 사업장

**EXTENDS** (확장 관계)
- 한 문서가 다른 문서의 정보를 확장하거나 더 상세히 설명
- 추가 필드를 포함하거나 더 구체적인 값을 제공
- 예: 기본 병원 정보 → 상세 진료 과목 및 의료진 정보

**PREREQUISITE_OF** (선행 관계)
- 한 문서를 이해하기 위해 다른 문서의 정보가 필요
- 명확한 정보 의존성 존재
- 예: 병원 기본 정보 → 병원의 특정 부서 상세 정보

**IMPLEMENTS** (구현 관계)
- 계획/방침/기준을 실제 데이터로 구현
- 추상적 내용 → 구체적 실제 값
- 예: 병원 평가 기준 문서 → 실제 병원 평가 결과 데이터

**CO_OCCURS** (동시 출현 관계)
- 그래프 구조상 자주 함께 나타남 (공통 필드나 값이 많음)
- similarity_score가 높지만 명확한 의미 관계는 약함
- 구조적으로 연관되어 함께 검색/참조될 가능성이 높음
- 예: 같은 지역, 같은 사업 유형의 다른 사업장 정보들

# reason 작성 가이드 (매우 중요!)
reason은 사용자가 그래프를 탐색할 때 "왜 이 문서들이 연결되었는지" 이해하는 핵심 정보입니다.

**필수 포함 요소:**
1. 두 문서의 공통 주제/개념 명시
2. 관계의 구체적 이유 (2-3문장)
3. 사용자가 A를 본 후 B를 보면 얻는 가치

**좋은 예시:**
"두 문서 모두 요양병원 정보를 담고 있으며, '가평군' 지역에 위치하고 '재활의학과'를 운영한다는 공통점이 있습니다. 병상 수와 진료 과목 정보를 통해 환자가 적합한 요양병원을 선택하는 데 도움을 줍니다."

**나쁜 예시 (피할 것):**
- "두 문서가 유사합니다" (너무 추상적)
- "관련된 내용입니다" (이유 없음)
- "같은 주제를 다룹니다" (가치 설명 없음)

# 분석 시 고려사항
- 공유하는 필드(Keys)와 내용의 유사성을 근거로 활용
- 유사도 점수(similarity_score)가 높으면 더 강한 관계 고려
- 관계가 명확하지 않거나 reason을 구체적으로 쓸 수 없으면 제외
- confidence는 관계의 확실성 (0.7 미만은 제외 권장)
- reason 길이는 최소 50자 이상
- **두 문서 간 가장 적절한 관계 타입 1개만 선택**

# 출력 형식
반드시 infer_document_relations function tool을 호출하여 응답하세요.
- 관계가 있으면: 1개 원소를 가진 배열 반환 (target_doc_id, relation_type, reason, confidence 포함)
- 관계가 없으면: 빈 배열 반환
"""

    ENTITY_EXTRACT_FROM_QUERY: str = """
**반드시 tool call을 사용하여 응답할것.**

당신은 검색 쿼리에서 핵심 Entity를 추출하는 전문가입니다.

다음 검색 쿼리에서 구체적인 Entity(개체)를 추출하세요.

# Entity 정의
- 인명, 기관명, 지명, 질병명, 치료법, 제품/서비스명 등 구체적 개체
- 일반적인 형용사나 동사는 제외

# 예시
Query: "위암 수술 후 회복기 케어 가능한 세브란스 연계 병원"
→ Entities: ["위암", "수술", "회복기 케어", "세브란스", "세브란스병원", "연계 병원"]

Query: "강남구 1차 병원 찾기"
→ Entities: ["강남구", "1차 병원"]

---
Query: {query_text}

출력 형식 (JSON):
{{"entities": ["entity1", "entity2", ...]}}
"""

    QUERY_GRAPH_PARSING: str = """**반드시 parse_query_to_graph tool을 사용하여 응답할 것.**

당신은 검색 쿼리를 그래프 패턴으로 변환하는 전문가입니다.

다음 검색 쿼리를 분석하여 그래프 패턴으로 변환하세요:

쿼리: {query}

현재 그래프 데이터베이스 스키마:

노드 레이블:
- Document: 검색 대상 문서
- Entity: 엔티티 (지역, 진료과목, 병원명, 개념 등)
- Topic: 주제/토픽
- Category: 카테고리

관계 타입:
- MENTIONS: Document → Entity (문서가 특정 엔티티를 언급)
- HAS_TOPIC: Document/Entity → Topic (주제 보유)
- BELONGS_TO: Topic → Category (카테고리 소속)
- SIMILAR_TO, EXTENDS, CONTRADICTS, CO_OCCURS: Document 간 관계

지침:
1. 검색 쿼리에서 검색 조건들을 추출하세요
2. 각 조건의 type을 결정:
   - "entity": Entity 노드로 표현 가능한 구체적인 개체 (지명, 병원명, 진료과목 등)
   - "property": Document의 속성으로 표현되는 특성 (병원 유형, 상태 등)
3. 각 조건의 semantic (의미적 역할)을 명확히 기술하세요
4. 모든 조건은 검색에 실제로 활용 가능한 것만 추출하세요

예시:
쿼리: "서울에 있는 정형외과 의원"
→ conditions: [
    {{"type": "entity", "value": "서울", "semantic": "위치"}},
    {{"type": "entity", "value": "정형외과", "semantic": "진료과목"}},
    {{"type": "property", "value": "의원", "semantic": "의료기관유형"}}
  ]

쿼리: "가평군 요양병원 중 내과가 있는 곳"
→ conditions: [
    {{"type": "entity", "value": "가평군", "semantic": "위치"}},
    {{"type": "property", "value": "요양병원", "semantic": "의료기관유형"}},
    {{"type": "entity", "value": "내과", "semantic": "진료과목"}}
  ]

주의사항:
- 구체적인 값만 추출하세요 (모호한 표현은 제외)
- 중복된 의미의 조건은 하나만 추출하세요
- 조건이 없으면 빈 배열을 반환하세요
"""

    DOCUMENT_RELEVANCE_VERIFICATION: str = """**반드시 verify_document_relevance tool을 사용하여 응답할 것.**

당신은 검색 결과의 적합성을 평가하는 전문가입니다.

검색 쿼리: {query}

문서 내용:
{document}

# 임무
이 문서가 검색 쿼리의 의도를 만족하는지 엄격하게 평가하세요.

# 평가 기준

1. **명시적 조건 충족** (필수)
   - 쿼리에 명시된 모든 조건을 만족하는가?
   - 예: "가평군"이라고 했으면 실제로 가평군인가?
   - 예: "요양병원"이라고 했으면 정말 요양병원인가? (정신병원, 한의원 등은 제외)
   - 예: "병상 수 60 이상"이라고 했으면 실제로 60개 이상인가?

2. **암묵적 전제 조건** (중요)
   - 일반적으로 기대되는 조건을 만족하는가?
   - 예: 폐업/휴업/직권폐업한 곳은 부적합
   - 예: 유효하지 않거나 더 이상 운영하지 않는 경우 부적합
   - 예: 명백히 다른 카테고리인 경우 부적합

3. **의미적 관련성**
   - 쿼리의 의도와 문서의 주제가 일치하는가?
   - 단순히 키워드가 포함되었다고 관련 있는 것은 아님

# 평가 방법

**엄격하게 평가하세요:**
- 조건을 하나라도 만족하지 않으면 → is_relevant: false, score: 0.0~0.4
- 대부분 조건을 만족하지만 일부 애매하면 → is_relevant: true, score: 0.5~0.7
- 모든 조건을 명확히 만족하면 → is_relevant: true, score: 0.8~1.0

**reason 작성 가이드:**
- 구체적으로 어떤 조건을 만족/불만족하는지 명시
- 3문장 이내로 간결하게

# 출력
반드시 verify_document_relevance function tool을 호출하세요.
"""

    SCHEMA_ANALYSIS: str = """**반드시 analyze_schema tool을 사용하여 응답할 것.**

당신은 데이터 스키마 분석 전문가입니다.

# 임무
다음 데이터를 분석하여 스키마를 추론하세요.

## 샘플 데이터
{sample_data}

## 통계 정보
{statistics}

# 분석 가이드

## 1. 도메인 식별
데이터의 주제/분야를 파악하세요:
- 의료: 병원, 환자, 진료, 질병 관련
- 법률: 법원, 판례, 사건, 소송 관련
- 부동산: 아파트, 매물, 거래, 면적 관련
- 음식점: 레스토랑, 메뉴, 리뷰 관련
- 기타: 위 카테고리에 속하지 않는 경우

## 2. 중심 노드명 생성
데이터를 대표하는 한글 이름을 만드세요:
- 좋은 예: "병원현황", "판례정보", "부동산매물", "음식점정보"
- 나쁜 예: "데이터", "정보", "현황" (너무 추상적)

## 3. 필드별 분석

### 한글명 규칙
- 명확하고 간결하게 (2-4글자)
- 예: "CASE_NO" → "사건번호", "PRICE" → "가격"

### 데이터 타입 판단
샘플 값과 통계를 보고 타입을 정확히 추론:

**text**: 일반 텍스트
- 예: "서울중앙지방법원", "원고 승소"

**number**: 숫자 (정수/실수)
- 예: "1500", "84.5", "3"
- 통계: unique 개수가 다양, 숫자만 포함

**date**: 날짜
- 예: "2023-05-15", "20230515", "2023.05.15"
- 패턴: YYYY-MM-DD, YYYYMMDD 등

**phone**: 전화번호
- 예: "02-1234-5678", "01012345678"
- 패턴: 숫자와 하이픈, 10-11자리

**address**: 주소
- 예: "서울시 강남구 테헤란로 123"
- 키워드: 시, 구, 동, 로, 길

**email**: 이메일
- 예: "user@example.com"
- 패턴: @와 도메인

**url**: URL
- 예: "https://example.com"
- 패턴: http://, https://

## 4. 신뢰도 평가
다음 기준으로 0.0~1.0 점수 부여:
- 1.0: 모든 필드의 의미와 타입이 명확
- 0.8: 대부분 명확하지만 일부 애매함
- 0.6: 절반 정도 추론 가능
- 0.4 이하: 데이터가 불충분하거나 구조가 복잡

## 5. 주의사항
- unique_count가 적으면 → 카테고리/상태 필드 가능성
- null_ratio가 높으면 → 선택적 필드
- 샘플 값 길이가 일정하면 → 코드/ID 가능성

# 출력
반드시 analyze_schema function tool을 호출하세요.
"""

    GRAPH_QUERY_PARSE: str = """**반드시 parse_graph_query tool을 사용하여 응답할 것.**

당신은 사용자의 자연어 쿼리를 Key-Value 그래프 조건으로 파싱하는 전문가입니다.

# 임무
사용자 쿼리에서 검색 조건을 추출하여 구조화하세요.

사용자 쿼리: {query}

# 추출 규칙

1. **필드(field) 추출**
   - 비교 대상이 되는 속성명 추출
   - 예: "병상수", "지역", "업종", "직원수", "의료기관유형"
   - 자연어 표현을 필드명으로 정규화

2. **연산자(operator) 판단**
   - "~이상": ">="
   - "~초과": ">"
   - "~이하": "<="
   - "~미만": "<"
   - "~인", "~의", "~가": "="
   - "포함", "있는": "CONTAINS"
   - "아닌", "제외": "!="

3. **값(value) 추출**
   - 비교 대상 값
   - 예: "100", "경기도", "병원", "제조업"
   - 숫자는 문자열로 반환

# 예시

**Query:** "병상수가 100개 이상인 병원"
→ conditions: [
     {field: "병상수", operator: ">=", value: "100"},
     {field: "의료기관유형", operator: "CONTAINS", value: "병원"}
   ]

**Query:** "경기도에 있는 제조업 공장"
→ conditions: [
     {field: "지역", operator: "CONTAINS", value: "경기도"},
     {field: "업종", operator: "CONTAINS", value: "제조업"}
   ]

**Query:** "직원수 10명 이상 50명 이하"
→ conditions: [
     {field: "직원수", operator: ">=", value: "10"},
     {field: "직원수", operator: "<=", value: "50"}
   ]

**Query:** "가평군 요양병원"
→ conditions: [
     {field: "지역", operator: "CONTAINS", value: "가평군"},
     {field: "의료기관유형", operator: "CONTAINS", value: "요양병원"}
   ]

# 주의사항
- 명시적인 조건만 추출 (추측하지 말 것)
- 조건이 없으면 빈 배열 반환
- 필드명은 사용자가 이해할 수 있는 자연어로 유지

# 출력
반드시 parse_graph_query function tool을 호출하세요.
"""

    CLUSTER_NORMALIZE: str = """**반드시 normalize_cluster tool을 사용하여 응답할 것.**

당신은 유사한 값들을 분석하여 적절한 정규화 값을 생성하는 전문가입니다.

# 임무
다음은 클러스터링으로 묶인 유사한 값들의 리스트입니다.
이 값들을 대표할 수 있는 **하나의 정규화 값**을 선택하거나 생성하세요.

Values (클러스터): {cluster_values}

# 중요 원칙

**출력은 반드시 Value들 자체에서 나와야 합니다**
- "병상수", "날짜", "범위" 같은 추상적 메타 개념을 만들지 마세요
- 주어진 Value 중 하나를 선택하거나, Value들의 조합/정제 형태만 허용

# 정규화 전략

## 1. 가장 완전하고 표준적인 형태 선택

예시:
- ["꿈이있는 치과", "꿈이있는치과병원", "꿈이있는치과"]
  → "꿈이있는치과병원" (가장 완전한 형태)

- ["서울대병원", "서울대학교병원", "서울대 병원"]
  → "서울대학교병원" (가장 정식 명칭)

- ["성모병원", "가톨릭대 성모병원", "가톨릭대학교 성모병원"]
  → "가톨릭대학교 성모병원" (가장 완전한 형태)

## 2. 공백/특수문자가 적절히 포함된 형태 선택

예시:
- ["서울특별시 강남구", "서울 강남구", "강남구"]
  → "서울특별시 강남구" (가장 완전한 행정구역명)

- ["010-1234-5678", "01012345678", "010 1234 5678"]
  → "010-1234-5678" (가독성 좋은 형태)

## 3. 날짜는 가장 표준적인 형식 선택

예시:
- ["2026-03-03", "2026.03.03", "20260303"]
  → "2026-03-03" (표준 형식)

- ["2026/12/25", "2026-12-25"]
  → "2026-12-25" (ISO 표준)

## 4. 숫자는 가장 깔끔한 형태 선택

예시:
- ["100", "100명", "100 명"]
  → "100" (단위 제거, 숫자만)

- ["1,000", "1000", "1000원"]
  → "1000" (쉼표 제거, 숫자만)

# 출력 규칙

1. **반드시 하나의 문자열만 반환**
2. **원본 Value 중 하나를 선택하거나, Value들의 조합/정제 형태만 허용**
3. **Key 이름이나 추상적 메타 개념을 절대 출력하지 말 것**
4. 선택 기준: 완전성 > 표준성 > 가독성

# 출력
반드시 normalize_cluster function tool을 호출하세요.
"""

    KEY_VALUE_EXTRACT: str = """**반드시 extract_key_value_pairs tool을 사용하여 응답할 것.**

당신은 데이터에서 Key-Value 쌍을 추출하는 전문가입니다.

# 임무
다음 데이터에서 모든 필드명(Key)과 그 값(Value)을 추출하세요.

데이터:
{data_text}

# 추출 규칙

1. **Key (필드명)**
   - 원본 필드명을 그대로 추출 (예: "BIZPLC_NM", "SIGUN_NM", "SICKBD_CNT")
   - 영문 대문자 + 밑줄 형식 유지
   - 데이터에 있는 모든 필드를 누락 없이 추출

2. **Value (값)**
   - 각 필드의 실제 값을 정확하게 추출
   - 빈 값("")이나 null인 경우 빈 배열 반환
   - 복합 값(쉼표로 구분된 값)은 분리하여 배열로 반환

3. **복합 값 처리**
   - "내과, 외과, 재활의학과" → ["내과", "외과", "재활의학과"]
   - "서울특별시 강남구" → ["서울특별시 강남구"] (단일 값)

# 예시

입력:
{{
    "BIZPLC_NM": "노체리안드리자애병원",
    "SIGUN_NM": "가평군",
    "SICKBD_CNT": "83",
    "TREAT_SBJECT_CONT_INFO": "내과, 외과, 재활의학과",
    "BSN_STATE_NM": "영업중",
    "REFINE_ROADNM_ADDR": "",
    "LICENSG_DE": "2020-01-15"
}}

출력:
{{
    "key_values": {{
        "BIZPLC_NM": ["노체리안드리자애병원"],
        "SIGUN_NM": ["가평군"],
        "SICKBD_CNT": ["83"],
        "TREAT_SBJECT_CONT_INFO": ["내과", "외과", "재활의학과"],
        "BSN_STATE_NM": ["영업중"],
        "REFINE_ROADNM_ADDR": [],
        "LICENSG_DE": ["2020-01-15"]
    }}
}}

# 주의사항
- 모든 필드를 빠짐없이 추출
- 값의 자료형에 관계없이 문자열 배열로 반환
- 숫자는 문자열로 변환 ("83")
- 날짜도 문자열로 변환 ("2020-01-15")
- 빈 값은 빈 배열 ([])

# 출력
반드시 extract_key_value_pairs function tool을 호출하세요.
"""

    @classmethod
    def get_prompt(cls, task: str) -> str:
        """Get fallback prompt by task name."""
        prompt_map: Dict[str, str] = {
            "meta_extract": cls.META_EXTRACT,
            "cypher_generate": cls.CYPHER_GENERATE,
            "category_generate": cls.CATEGORY_GENERATE,
            "document_relation_inference": cls.DOCUMENT_RELATION_INFERENCE,
            "entity_extract_from_query": cls.ENTITY_EXTRACT_FROM_QUERY,
            "query_graph_parsing": cls.QUERY_GRAPH_PARSING,
            "document_relevance_verification": cls.DOCUMENT_RELEVANCE_VERIFICATION,
            "key_value_extract": cls.KEY_VALUE_EXTRACT,
            "cluster_normalize": cls.CLUSTER_NORMALIZE,
            "schema_analysis": cls.SCHEMA_ANALYSIS,
            "graph_query_parse": cls.GRAPH_QUERY_PARSE,
        }

        if task not in prompt_map:
            raise ValueError(
                f"Unknown task: {task}. Available tasks: {', '.join(prompt_map.keys())}"
            )

        return prompt_map[task]
