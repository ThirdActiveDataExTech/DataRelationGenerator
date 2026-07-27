from typing import Dict


class FallbackPrompts:
    """Container for fallback prompt templates used across LLM tasks."""


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

당신은 데이터에서 숨겨진 메타데이터를 발굴하는 전문가입니다.

# 임무
다음 데이터를 분석하여:
1. 데이터의 **도메인**과 **중심 개념**을 파악하고
2. **모든 원본 필드명을 의미있는 한글로 변환**하고
3. **긴 텍스트 필드에서 추출 가능한 메타데이터**를 설계하세요

**중요: 모든 필드명(원본 필드 + extractable_metadata)은 반드시 한글로 작성하세요.**

## 샘플 데이터
{sample_data}

## 통계 정보
{statistics}

---

# 핵심 임무 1: 원본 필드 한글 변환 (fields) ⭐⭐⭐

**반드시 모든 원본 필드를 한글로 번역하세요!**

예시 (병원 데이터):
```
원본: BIZPLC_NM → korean_name: "병원명"
원본: GENRL_AMBLNC_VCNT → korean_name: "일반구급차수"
원본: SICKBD_CNT → korean_name: "병상수"
```

---

# 핵심 임무 2: 추출 가능한 메타데이터 설계 (extractable_metadata) ⭐

**원칙:**
1. 긴 텍스트 필드("text", "content", "description" 등)의 **내용을 분석**
2. 원본 필드에 **없는 새로운 정보**만 extractable_metadata로 설계
3. **검색/분류/필터링에 유용**한 메타데이터 우선
4. **실제로 추출 가능한** 것만 포함

## 추출 가능한 메타데이터 예시 (도메인별)

### 예시 1: 의료연구 논문
```
text: "Female breast cancer incidence... Utah... Mormon... 1985-1999..."

extractable_metadata:
- 연구주제 (text_list): ["유방암 발병률", "종교와 건강"]
- 연구기간 (date_range): ["1985-1999"]
- 연구지역 (text): ["Utah"]
- 질병분류 (text_list): ["암", "유방암"]
- 연구대상 (category): ["여성", "종교인"]
```

### 예시 2: 병원 정보
```
text: "노인 전문 요양병원. 치매, 파킨슨병 환자 특화 케어..."

extractable_metadata:
- 전문분야 (text_list): ["노인의료", "요양케어"]
- 주요질환 (text_list): ["치매", "파킨슨병"]
- 대상환자 (category): ["노인", "만성질환자"]
```

### 예시 3: 법률 판례
```
text: "원고 승소. 손해배상 청구... 서울중앙지법 2022.05.10 선고..."

extractable_metadata:
- 판결결과 (category): ["원고 승소"]
- 사건유형 (category): ["손해배상"]
- 법원 (text): ["서울중앙지법"]
- 선고일 (date): ["2022-05-10"]
```

### 예시 4: 제조업 공장
```
text: "자동차 부품 전문 제조. ISO 9001 인증. 직원 150명..."

extractable_metadata:
- 제조품목 (text_list): ["자동차 부품"]
- 인증현황 (text_list): ["ISO 9001"]
- 규모 (number): ["150"]  // 직원수
```

### 예시 5: 전자상거래 상품
```
text: "삼성 갤럭시 최신 모델. 128GB, 블루. 배터리 5000mAh..."

extractable_metadata:
- 브랜드 (category): ["삼성"]
- 제품라인 (category): ["갤럭시"]
- 용량 (text): ["128GB"]
- 색상 (category): ["블루"]
- 사양 (text_list): ["배터리 5000mAh"]
```

## 메타데이터 타입 가이드
- **text**: 단일 텍스트 (예: 지역, 법원명)
- **text_list**: 여러 키워드/항목 (예: 주제, 질병, 인증)
- **number**: 숫자 (예: 직원수, 용량)
- **date**: 단일 날짜
- **date_range**: 기간 (예: "1985-1999")
- **category**: 분류/상태 (예: 결과, 유형, 브랜드)

## 추출 가능 메타데이터 판단 기준

**추출 가능**
- 텍스트에서 명시적으로 언급된 개체/값
- 분류/검색에 직접 활용 가능
- 구체적이고 명확한 정보

**추출 불가능**
- 원본 필드에 이미 있는 정보 (중복)
- 텍스트에 없는 추측/추론
- 너무 추상적이거나 모호한 개념

## 주의사항
- **이 예시들은 참고용**입니다. 실제 데이터 내용에 맞게 설계하세요
- 원본 필드 구조를 그대로 복사하지 마세요
- 텍스트가 짧거나 구조화된 데이터만 있으면 extractable_metadata는 빈 딕셔너리 {{}}
- examples는 샘플 데이터에서 실제로 추출한 값만 포함

---

# 출력 형식

반드시 analyze_schema tool 호출:
```
{{
    "domain": "도메인명 (예: 의료, 법률, ... )",
    "centrality_name": "거시적 도메인명 (예: 의료, 법률, 방송 등) - domain과 동일하거나 더 포괄적으로",
    "fields": {{
        "원본필드명": {{
            "korean_name": "의미있는 한글명",
            "data_type": "타입 (text, number, date 등)",
            "description": "필드 설명"
        }},
        ...
    }},
    "extractable_metadata": {{
        "메타필드명": {{
            "korean_name": "한글명",
            "source_field": "원본필드명 (text, content 등)",
            "data_type": "타입 (text, text_list, number, date, date_range, category)",
            "description": "이 메타데이터가 무엇인지 명확한 설명",
            "examples": ["샘플에서 실제 추출한 예시1", "예시2"]
        }},
        ...
    }},
    "confidence": 0.0~1.0,
    "notes": "추가 설명"
}}
```

**필수: 원본 필드(fields)도 반드시 출력하세요!**
- **모든 원본 필드의 korean_name을 의미있는 한글로 번역** (예: "GENRL_AMBLNC_VCNT" → "일반구급차수")
- 영문 약어는 풀어서 의미를 명확히 (CNT→수, NM→명, ADDR→주소 등)
- extractable_metadata만 집중해서 설계하세요.
- 추출 가능한 메타데이터가 없으면 extractable_metadata는 {{}}로 반환하세요.
"""

    QUERY_KEY_INFERENCE: str = """**반드시 infer_required_keys tool을 사용하여 응답할 것.**

당신은 사용자의 질문을 분석하여 필요한 그래프 Key들을 추론하는 전문가입니다.

# 임무
사용자 질문에서 **어떤 Key들이 검색에 필요한지** 추론하세요.

## 당신의 답변이 사용되는 방식 (중요!)

1. **Key 매칭**: 당신이 추론한 Key 이름(예: "지역")으로 그래프에서 실제 Key 노드를 **벡터 검색**으로 찾습니다
   - 예: "지역" → 실제 그래프의 "SIGUN_NM" Key 노드와 매칭

2. **Value 검색**: 매칭된 Key 내에서 당신이 제공한 search_terms(예: "강남구")와 유사한 Value를 **벡터 검색**으로 찾습니다
   - 예: "강남구" → "서울특별시 강남구", "  강남구청", "강남" 등의 유사 Value 검색

3. **Document 반환**: 찾은 Value들과 연결된 Document를 사용자에게 반환합니다

**따라서:**
- Key는 **의미적으로 넓은 개념**으로 추론하세요 (벡터 검색으로 유사 Key를 찾기 때문)
- search_terms는 **실제 검색할 구체적인 값**을 제공하세요

사용자 질문: {query}

# 그래프 구조
- Key: 데이터의 속성명 (예: "이름", "지역", "업종", "규모", "날짜", "상태")
- Value: Key에 대한 실제 값
- Document: 원본 데이터

# Key 추론 규칙

1. **검색어 분석**
   - 고유명사(이름, 기관명, 제품명) → "이름", "기관명", "제품명" 등
   - 지역/위치 → "지역", "주소", "시도", "시군구" 등
   - 숫자/규모 → "직원수", "규모", "면적", "가격" 등
   - 분야/업종/카테고리 → "업종", "분야", "유형", "카테고리" 등
   - 날짜/기간 → "날짜", "기간", "년도" 등
   - 상태 → "상태", "영업상태", "진행상황" 등

2. **Key 이름은 일반적인 한글 단어로**
   - 구체적이고 명확하게
   - 도메인에 관계없이 통용되는 용어 우선

3. **숫자 조건 처리 (매우 중요!)**
   - "이상", "초과", "미만", "이하" 같은 비교 연산이 포함된 경우
   - search_terms를 {{"operator": "연산자", "value": 숫자}} 형식으로 반환
   - 연산자 종류: ">=" (이상), ">" (초과), "<" (미만), "<=" (이하), "==" (같음)
   - 예: "100개 이상" → {{"operator": ">=", "value": 100}}
   - 예: "50명 미만" → {{"operator": "<", "value": 50}}
   - 예: "1억 초과" → {{"operator": ">", "value": 100000000}}

4. **검색어가 포함될 가능성이 높은 Key만 선택**
   - 너무 많이 선택하지 말 것 (1~3개 정도)

# 예시 (다양한 도메인)

**Query:** "삼성전자"
→ keys: ["기업명"]
→ search_terms: {{"기업명": "삼성전자"}}

**Query:** "서울 강남구 카페"
→ keys: ["지역", "업종"]
→ search_terms: {{"지역": "서울 강남구", "업종": "카페"}}

**Query:** "직원수 100명 이상인 제조업체"
→ keys: ["직원수", "업종"]
→ search_terms: {{"직원수": {{"operator": ">=", "value": 100}}, "업종": "제조업"}}

**Query:** "병상수가 100개 이상인 병원"
→ keys: ["병상수", "업종"]
→ search_terms: {{"병상수": {{"operator": ">=", "value": 100}}, "업종": "병원"}}

**Query:** "매출 1억 미만 기업"
→ keys: ["매출"]
→ search_terms: {{"매출": {{"operator": "<", "value": 100000000}}}}

**Query:** "2023년 손해배상 판결"
→ keys: ["년도", "사건유형"]
→ search_terms: {{"년도": "2023", "사건유형": "손해배상"}}

**Query:** "영업중인 음식점"
→ keys: ["상태", "업종"]
→ search_terms: {{"상태": "영업중", "업종": "음식점"}}

**Query:** "경기도 소재 연구소"
→ keys: ["지역", "유형"]
→ search_terms: {{"지역": "경기도", "유형": "연구소"}}

# 주의사항
- 질문과 **직접 관련**된 Key만 추론
- 추측하지 말고 명확한 것만
- Key 개수는 1~3개 정도가 적절
- search_terms는 실제로 검색할 값

# 출력
반드시 infer_required_keys function tool을 호출하세요.
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

    RELATION_KEY_ANALYSIS: str = """**반드시 analyze_relation_search_condition tool을 사용하여 응답할 것.**

당신은 Document의 Key-Value 메타데이터를 분석하여 관계 있는 Document를 찾기 위한 검색 조건을 생성하는 전문가입니다.

# 임무
주어진 Document의 Key-Value 메타데이터를 분석하여:
1. **사용자 관점에서 "왜" 관련 데이터를 찾을지 추론**
2. 도메인 특성에 맞는 중요 Key 선정
3. 각 Key의 특성 분류 (feature class)
4. 검색 전략 수립

# 사용자 의도 분석 (매우 중요!)

**데이터를 찾는 이유에 따라 중요 Key가 달라집니다:**

**병원/의료기관 → 지역 비교/벤치마킹**
- 우선순위: 지역 > 진료과목 > 병상수
- 이유: 같은 지역의 유사 규모 병원과 비교하려는 경우가 많음

**공장/제조업 → 경쟁사/협력사 분석**
- 우선순위: **생산품/업종 > 산업유형 > 지역** > 규모
- 이유: 같은 제품을 만드는 업체, 같은 산업군의 협력사를 찾으려는 경우가 많음
- **주의**: 지역은 2순위! 업종/생산품이 최우선

**학교/교육기관 → 교육 정책 비교**
- 우선순위: 학교유형 > 지역 > 학생수
- 이유: 같은 유형의 학교 간 정책/성과 비교

**기업/사업체 → 시장 분석**
- 우선순위: 업종/사업분야 > 지역 > 매출규모
- 이유: 같은 업종의 경쟁사, 시장 동향 파악

**판례/법률 → 법률 연구**
- 우선순위: 법조항 > 판결 유형 > 사건 유형
- 이유: 같은 법 조항의 판례 선례 찾기

**중요**: location(지역)이 항상 최우선이 아닙니다!
도메인 특성과 사용자 의도에 따라 우선순위가 완전히 달라집니다.

[분석 대상 Document]
ID: {document_id}

Key-Value 메타데이터:
```json
{key_values_json}
```

# Key 특성 분류 (feature_class)

**identity** (식별자)
- 거의 고유한 식별자
- 예: 병원명, 기관명, 상품명, 회사명
- 용도: 같은 문서 재발견, 중복 확인
- 관계 발굴 중요도: 낮음 (너무 고유해서 관계 찾기 어려움)

**location** (위치)
- 지역, 소재지, 시도, 구군 등
- 예: 서울특별시, 강남구, 부산시
- 용도: 같은 지역 문서 찾기 (후보 공간 축소에 매우 강함)
- 관계 발굴 중요도: 매우 높음

**categorical** (범주형)
- 카테고리, 분류, 유형 등
- 예: 진료과목, 전문분야, 기관유형, 사업상태
- 용도: 공통 속성 기반 관계 탐색
- 관계 발굴 중요도: 높음

**numeric** (숫자형)
- 수치 데이터
- 예: 병상수, 직원수, 매출, 연도
- 용도: 범위/근접 검색 (정확 매칭보다는 유사 범위)
- 관계 발굴 중요도: 중간

**free-text** (자유텍스트)
- 설명, 소개, 비고 등
- 예: 병원 소개, 서비스 설명, 특징
- 용도: 다른 근거가 약할 때 보조 신호
- 관계 발굴 중요도: 낮음 (보조)

# Key 선정 기준

**중요도가 높은 Key:**
- 식별력이 있으면서 너무 고유하지 않은 Key
- 정규화 가능성이 있는 Key
- 관계의 이유로 설명 가능한 Key
- 도메인에서 핵심 속성인 Key

**중요도가 낮은 Key:**
- 너무 고유한 값 (identity)
- 오타, 자유서술이 많은 Key
- 빈 값이 많은 Key
- 관계 설명력이 약한 Key

# 도메인별 Key 우선순위 예시

## 병원 도메인 (지역 비교/벤치마킹 목적)
상위 중요도:
- 지역/소재지: location (같은 지역 병원 찾기)
- 진료과목: categorical (같은 진료과 병원 찾기)
- 병상수: numeric (유사 규모 병원 찾기)

하위 중요도:
- 병원명: identity (너무 고유함)
- 비고: free-text (노이즈 많음)

## 공장/제조업 도메인 (경쟁사/협력사 분석 목적)
상위 중요도:
- **생산품정보: categorical (같은 제품 생산 업체 찾기) ← 최우선!**
- 산업유형/업종: categorical (같은 산업군 찾기)
- 지역: location (같은 산단/지역 업체 찾기) ← 2순위

중간 중요도:
- 직원수/부지면적: numeric (유사 규모 찾기)
- 공장등록일: numeric (설립 시기 유사성)

하위 중요도:
- 회사명: identity (너무 고유함)
- 관리기관명: categorical (관계 설명력 약함)

## 기타 도메인도 같은 원칙 적용
- 사용자가 "왜" 데이터를 찾는지 먼저 추론
- 그에 맞는 Key 우선순위 결정
- **지역은 항상 최우선이 아님!**

# 출력 예시

입력:
{{
    "지역": ["서울특별시 종로구"],
    "병원명": ["서울대학교병원"],
    "병상수": ["1500"],
    "진료과목": ["내과", "외과", "정형외과"],
    "설립유형": ["국립"],
    "비고": ["상급종합병원, 교육병원"]
}}

출력:
{{
    "important_keys": [
        {{
            "key_name": "지역",
            "feature_class": "location",
            "importance": 0.95,
            "reason": "같은 지역 병원을 찾는 데 매우 유효하며, 후보 공간 축소에 강력함"
        }},
        {{
            "key_name": "진료과목",
            "feature_class": "categorical",
            "importance": 0.85,
            "reason": "같은 진료과목을 가진 병원을 찾아 유사 기능 기관 발견 가능"
        }},
        {{
            "key_name": "병상수",
            "feature_class": "numeric",
            "importance": 0.80,
            "reason": "유사한 규모의 병원을 찾아 비교 가능한 기관 식별"
        }},
        {{
            "key_name": "설립유형",
            "feature_class": "categorical",
            "importance": 0.60,
            "reason": "같은 설립 유형 병원 간 운영 특성 유사성 발견"
        }}
    ],
    "search_strategy": "같은 지역(서울)에 있으면서 유사한 진료과목과 병상 규모를 가진 병원 찾기",
    "expected_relation_count": 10
}}

# 주의사항
- **사용자 의도를 먼저 추론** (비교? 경쟁사 분석? 법률 연구?)
- 도메인에 맞는 Key 우선순위 결정
- **location이 항상 최우선이 아님!** 공장/기업은 업종/생산품이 먼저
- identity는 가급적 제외 (너무 고유해서 관계 찾기 어려움)
- categorical 중에서도 도메인 핵심 속성 우선
- numeric은 중간 우선순위
- free-text는 보조로만 사용
- search_strategy는 도메인과 의도를 반영해 구체적으로 작성
- expected_relation_count는 도메인에 따라 5~15 정도 적절

# 출력
반드시 analyze_relation_search_condition function tool을 호출하세요.
"""

    FIELD_TYPE_CLASSIFICATION: str = """**반드시 classify_field_type tool을 사용하여 응답할 것.**

메타데이터 필드 타입을 분류하세요.

## 필드 정보
- 필드명: {key_name}
- 샘플 값:
{sample_values}

## 타입 선택 기준

### semantic (의미적 유사도 매칭)
- 자연어로 된 텍스트 필드
- 의미적 유사도 비교가 유효함
- 예시: 타이틀, 장르, 내용물 설명, 주제어, 권한

### numeric_date (숫자/날짜 범위 매칭)
- 날짜, 연도, 숫자 등 정량적 값
- 시간적/수치적 근접성으로 비교해야 함
- 예시: 발행일자, 등록연도, 병상수, 면적, 좌표

### proper_noun (고유명사 클러스터링)
- 기관명, 인명, 지명 등 고유명사
- 텍스트 구조 유사도가 아닌 의미적 범주로 분류해야 함
- 예시: 발행기관명, 제작자, 저작권자, 병원명, 시군명

## 주의사항
- 숫자처럼 보여도 의미적 비교가 필요하면 semantic (예: 전화번호)
- 날짜 형식(YYYY-MM-DD)이면 numeric_date
- 지명도 고유명사이므로 proper_noun
"""

    TABLE_RELATION_VERIFICATION: str = """**반드시 verify_table_relation tool을 사용하여 응답할 것.**

당신은 두 데이터 테이블 사이에 실제 데이터 근거에 기반한 관계가 있는지를 판단하는 전문가입니다.

# 테이블 A
이름: {table_a}
도메인: {domain_a}
설명: {description_a}
요약: {summary_a}
컬럼: {columns_a}
샘플 레코드:
{samples_a}

# 테이블 B
이름: {table_b}
도메인: {domain_b}
설명: {description_b}
요약: {summary_b}
컬럼: {columns_b}
샘플 레코드:
{samples_b}

임베딩 유사도: {embedding_score}

# 판단 원칙

**related=true의 근거는 반드시 두 테이블 안에 이미 존재해야 합니다.**
"두 테이블을 결합하는 새로운 서비스를 상상할 수 있다"는 것은 근거가 아닙니다.
근거를 아래 유형 중 하나로 분류하여 evidence_type에 명시하세요.

# 데이터 성격 구분 (판단 전 각 테이블을 분류하여 table_a_nature / table_b_nature에 기록)
- **record (기록 데이터)**: 실세계의 개체·사건(환자, 기관, 장비, 거래 등)의 상태를
  행 단위로 기록한 데이터
- **knowledge_content (지식·콘텐츠성 데이터)**: 지식이나 콘텐츠 자체가 행의 대상인
  데이터 (Q&A 지식, 영상·게시물 메타데이터, 학습용 어노테이션). 조회수·업로드일 같은
  부가 정보가 있어도 knowledge_content입니다.

# 근거 유형 (evidence_type)

## 1. shared_key
- 공통 컬럼(FK/PK)이 있거나 동일한 식별자 체계를 공유하는가?
- 반드시 **동일한** 식별자 체계여야 합니다. 서로 다른 식별자 체계는 shared_key가
  아닙니다 (예: 한쪽은 HOSPITAL_ID, 다른 쪽은 요양기호 → 불인정)
- 예: 두 테이블 모두 HOSPITAL_ID를 가짐

## 2. value_overlap
- 컬럼 이름은 달라도 값 도메인이 실제로 겹치는가?
- 반드시 샘플 레코드에서 겹치는 값을 직접 확인하고, reason에 그 값을 인용할 것
- **인정 (핵심 규칙)**: 한쪽이 record이고 그 기록의 필드 값이 상대 테이블의 값에
  등장하면, 상대가 knowledge_content여도 **반드시 value_overlap으로 인정**합니다.
  값이 질문·요약 등 텍스트 필드 안에 등장하는 경우도 포함합니다.
  이 경우를 "주제 중첩"으로 분류하지 마세요.
  - 예: 환자 기록의 진단명 "위암" ↔ 건강 콘텐츠 키워드 "위암" → value_overlap 인정
  - 예: 환자 기록의 진단명 "위암" ↔ 의료지식 Q&A의 "위암" 질문 → value_overlap 인정
  - 예: 환자 진단명 "위암" ↔ 이미지 라벨 "stomach_cancer" → value_overlap 인정
- **불인정**: 데이터를 생산·수집한 수단·방식의 명칭 일치는 값 중첩이 아닙니다.
  - 예: 한쪽은 내시경으로 촬영한 이미지, 다른 쪽은 "내시경" 장비 보유 목록 → 불인정

## 3. same_entity
- 두 테이블의 샘플에서 **동일한 개체 인스턴스**(같은 병원명, 같은 기관 식별자,
  같은 환자 ID)가 실제로 확인되는가?
- **양쪽 샘플 레코드에 같은 개체의 이름·식별자가 문자 그대로 존재해야 하며,
  reason에 양쪽 테이블에서 확인한 값을 각각 인용할 것.** 한쪽에만 등장하는 개체는
  근거가 아닙니다.
- 값의 종류가 같다는 사실(예: 둘 다 병원 이름을 담고 있음)만으로는 해당하지 않습니다
- 예: A의 채널명 "삼성서울병원TV" ↔ B의 병원명 목록에 "삼성서울병원"이 실제로 존재
  → 인정. B의 병원 목록에 그 병원이 없으면 → 불인정

## 4. parallel_schema
- 컬럼 구조가 동일하거나 거의 동일하고, 레코드의 키/주제가 서로 대응하는 동종 데이터셋인가?
- 예: 같은 주제를 난이도별/기간별로 나눈 분할본

## 5. usage_dependency (마지막 수단)
- 위 1~4 근거가 전혀 없지만, A를 단독으로 사용할 때 B 없이는 해석이 불가능하거나
  심각하게 불완전한 경우에만 해당
- "함께 쓰면 더 좋은 서비스를 만들 수 있다"는 가능성은 여기에 해당하지 않습니다

# related=false인 경우 (evidence_type=none)
- 위 근거 유형 중 어느 것도 두 테이블 안에서 확인할 수 없을 때
- 함께 사용하는 서비스 시나리오를 상상할 수 있다는 것만으로는 부족
- 단순히 같은 도메인이라는 이유만으로는 부족

# confidence 작성 방법
- 샘플 레코드에서 직접 확인한 근거(공통 컬럼, 겹치는 값)가 있으면 0.8 이상
- 근거가 간접적이거나 샘플에서 확인되지 않으면 0.8 미만

# relation_type 작성 방법
두 테이블이 어떤 방식으로 연결되는지 짧은 한국어 문장으로 작성하세요.
- shared_key: "A-B (1:N 참조)", "A-B (공통 식별자)"
- value_overlap: "A진단명-B라벨 값 중첩"
- same_entity: "동일 병원 참조"
- parallel_schema: "동일 스키마 병렬 데이터셋"

# 출력
반드시 verify_table_relation function tool을 호출하세요.
"""

    TABLE_METADATA_EXTRACTION: str = """**반드시 extract_table_metadata tool을 사용하여 응답할 것.**

당신은 데이터 테이블의 메타데이터를 분석하여 핵심 정보를 추출하는 전문가입니다.

# 테이블 정보
테이블명: {table_name}
컬럼 목록: {columns}
컬럼 설명: {column_descriptions}
샘플 레코드:
{sample_records}

# 임무

## 1. description
테이블이 무엇을 담고 있는지, 어떤 목적으로 쓰이는지를 1~2문장으로 요약하세요.
- 단순히 컬럼명을 나열하지 말 것
- "어떤 데이터가, 왜 필요한지"를 중심으로 작성

좋은 예: "암 환자의 진단코드, 수술 이력, 고혈압·당뇨 과거력 등 임상 정보를 담은 테이블로, 환자 맞춤형 의료 서비스 설계에 활용됩니다."
나쁜 예: "DIAGNOSIS_CODE, SURGERY_DATE, HYPERTENSION_HISTORY 컬럼을 포함합니다."

## 2. keywords
테이블을 대표하는 도메인 개념어 5~10개를 추출하세요.
- 컬럼명이 아니라 **의미 개념**으로 추출 (예: "진단코드" 대신 "암 진단", "수술 이력")
- 다른 테이블과 연결될 수 있는 개념어 우선 (예: "병원", "환자", "질환명")
- 한국어로 작성

# 출력
반드시 extract_table_metadata function tool을 호출하세요.
"""

    PROPER_NOUN_LABELING: str = """**반드시 label_proper_nouns tool을 사용하여 응답할 것.**

고유명사 값들을 범주별로 분류하세요.

## 필드명: {key_name}

## 값 목록:
{values_list}

## 선택 가능한 범주 (Taxonomy):
{taxonomy}

## 작업
각 값을 위 범주 중 하나로 분류하세요. 범주에 맞는 것이 없으면 "기타"를 선택하세요.

## 예시
- 필드: "발행기관명"
- 범주: ["정부기관", "문화기관", "항공사", "기타"]
- 한국문화진흥원 → 문화기관
- 한국항공 → 항공사
- KBS → 방송사 (없으면 기타)
"""

    @classmethod
    def get_prompt(cls, task: str) -> str:
        """Get fallback prompt by task name."""
        prompt_map: Dict[str, str] = {
            "document_relation_inference": cls.DOCUMENT_RELATION_INFERENCE,
            "document_relevance_verification": cls.DOCUMENT_RELEVANCE_VERIFICATION,
            "key_value_extract": cls.KEY_VALUE_EXTRACT,
            "cluster_normalize": cls.CLUSTER_NORMALIZE,
            "schema_analysis": cls.SCHEMA_ANALYSIS,
            "query_key_inference": cls.QUERY_KEY_INFERENCE,
            "relation_key_analysis": cls.RELATION_KEY_ANALYSIS,
            "field_type_classification": cls.FIELD_TYPE_CLASSIFICATION,
            "proper_noun_labeling": cls.PROPER_NOUN_LABELING,
            "table_relation_verification": cls.TABLE_RELATION_VERIFICATION,
            "table_metadata_extraction": cls.TABLE_METADATA_EXTRACTION,
        }

        if task not in prompt_map:
            raise ValueError(
                f"Unknown task: {task}. Available tasks: {', '.join(prompt_map.keys())}"
            )

        return prompt_map[task]
