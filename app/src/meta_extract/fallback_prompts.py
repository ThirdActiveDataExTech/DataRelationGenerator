from typing import Dict


class FallbackPrompts:
    """Container for fallback prompt templates used across LLM tasks."""

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

    TABLE_RELATION_VERIFICATION: str = """두 데이터셋 A, B 사이에 실제 근거가 있는 관계가 있는지 판정하고 verify_table_relation tool로 답하세요.

# 판정 원칙
- 근거는 주어진 정보(이름·설명·요약·컬럼·샘플) 안에 있어야 합니다. 함께 쓰는 서비스를 상상할 수 있다는 것, 같은 분야라는 것, 수집 수단의 이름이 같다는 것(예: CCTV 영상 ↔ CCTV 설치 대장)은 근거가 아닙니다.
- 메타데이터만 등록된 데이터셋은 컬럼·샘플이 비어 있는 것이 정상입니다. 한쪽이라도 그렇다면 아래 유형의 근거를 이름·설명·요약에서 찾고 인용은 비워 두세요. 비어 있다는 이유로 관계를 부정하거나 확신도를 낮추지 마세요.

# 데이터 성격 (table_a_nature, table_b_nature)
- record: 시설·기관·행정구역·거래 같은 실세계 개체나 사건을 행마다 기록한 데이터
- knowledge_content: 방송·영상·게시물 메타데이터, Q&A처럼 콘텐츠 자체가 행인 데이터

# 관계 유형 (level1_type)
A를 보는 사람에게 B가 무엇인지로 고릅니다. 근거를 댈 수 없으면 그 유형이 아니고, 어느 유형도 아니면 related=false 입니다.
- DETAIL_FOR: B가 A와 같은 개체의 다른 속성을 담습니다. 근거는 양쪽에 같은 이름으로 있는 식별자 컬럼이나, 양쪽 샘플에 모두 나오는 개체 이름·코드입니다. 예: 어린이집 현황 ↔ 어린이집 교직원 현황 (어린이집코드)
- STATISTIC_FOR: B가 개별 개체가 아니라 지역·기간·유형별로 묶은 수치이고, A를 해석할 기준이 됩니다. 근거는 양쪽에 모두 있는 지역·기간 값입니다. 예: 어린이집 현황 ↔ 시군구별 출생아 수 (A의 소재지와 B의 시군구에 모두 나오는 "성남시")
- SAME_KIND: B가 A와 같은 항목(같은 종류의 시설·업종 목록, 또는 같은 통계 지표)을 지역·기관·기간·분류 기준만 달리해 담아, 이어 붙이면 범위가 넓어집니다. 기관마다 항목 이름은 달라도 됩니다. 근거는 양쪽 이름·설명에 나오는 같은 대상입니다. 예: 광양시 약국 현황 ↔ 의정부시 약국 현황, 시도별 사망자 수 ↔ 성·연령별 사망자 수. 같은 시군에 다른 지표를 더하는 시군별 당뇨병 조절률 ↔ 투약순응률은 DETAIL_FOR 입니다.
- ADJACENT_TO: 같은 대상이 A와 B 중 하나를 고르거나 차례로 거치는 관계입니다. 인용은 필요 없지만 reason에 그 관계를 구체적으로 써야 합니다. 예: 어린이집 현황 ↔ 유치원 현황

# 근거 인용 (evidence_columns, evidence_values)
코드가 실제 컬럼·샘플과 대조해, 하나라도 틀리면 관계 전체를 기각합니다. 확인한 것만 넣고 없으면 비워 두세요.
- evidence_columns: A와 B에 똑같은 이름으로 있는 컬럼만. 뜻이 같아도 이름이 다르면(소재지 ↔ 시군구) 넣지 말고 값으로 인용하세요.
- evidence_values: A와 B의 샘플에 똑같이 나오는 값만, 나온 그대로 하나씩 (예: "성남시". "경기도 성남시 분당구"처럼 합치지 않기).

# 확신도 (confidence)
근거를 확인했으면 0.8 이상, 근거가 모호하면 그보다 낮게 주세요.

# 설명 필드
- relation_type: 두 데이터가 어떻게 이어지는지 한 줄. 예: "어린이집-교직원 (1:N, 어린이집코드 공통)"
- reason: 판정 근거. 검토자가 봅니다.
- usage: 두 데이터를 합치면 무엇을 알게 되는지 한 문장. 데이터를 찾는 사용자에게 그대로 보입니다. 관계가 없으면 빈 문자열.
  예: "회차마다 방영 주의 게시글·댓글 수가 붙어, 화제가 된 회차가 드러납니다."
  예: "어린이집코드로 붙이면 시설별 교사 1인당 아동 수가 나옵니다."

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

    DOCUMENT_METADATA_EXTRACTION: str = """**반드시 extract_table_metadata tool을 사용하여 응답할 것.**

당신은 문서(보고서·지침·통계연보·게시글 등)를 읽고 핵심 메타데이터를 만드는 전문가입니다.

# 문서
제목: {title}
본문 앞부분:
{excerpt}

# 임무

## 1. description
문서가 무엇을 다루는지, 누가 어떤 목적으로 볼 문서인지를 1~2문장으로 요약하세요.
- 목차·머리말을 옮겨 적지 말고 내용의 대상과 범위를 쓸 것
- 본문이 영어 등 외국어여도 한국어로 작성

좋은 예: "전국 의료기관의 병상 수와 시도별 분포를 정리한 연간 통계 보고서로, 지역별 의료 자원 비교에 쓰입니다."
나쁜 예: "제1장 개요, 제2장 현황, 제3장 부록으로 구성되어 있습니다."

## 2. keywords
문서를 대표하는 개념어 5~10개를 한국어로 추출하세요. 다른 데이터와 이어질 수 있는 대상·주제어(기관 종류, 지역, 질병, 제도명 등)를 우선하세요.

## 3. display_name
제목이 이미 자연스러우면 그대로, 파일명처럼 보이면 간결한 한국어 명사구로.

# 출력
반드시 extract_table_metadata function tool을 호출하세요.
"""

    DATASET_CATEGORY_CLASSIFICATION: str = """**반드시 classify_dataset_categories tool을 사용하여 응답할 것.**

당신은 데이터셋을 주제 분야로 분류하는 전문가입니다.
분류의 목적은 여러 데이터셋을 같은 분야로 **묶어** 사용자가 분야별로 찾게 하는 것입니다.

# 분류표 (국가지식정보 통합플랫폼 디지털 집현전 — 대분류: 중분류)
{taxonomy}

# 임무
데이터셋을 위 분류표의 '대분류 > 중분류'로 1~3개 분류하세요.

## 규칙
1. 반드시 분류표에 있는 '대분류 > 중분류' 그대로 쓰세요 (예: "의료 > 공중보건"). 새 분류를 만들지 마세요.
2. 데이터가 **무엇에 관한 것인지**(대상·주제)로 분류하세요. 제공 형식, 제공기관 이름, '통계'·'현황' 같은 자료 형태 표현만 보고 정하지 마세요.
   - 병원·약국 현황 → 의료, 시청률·방송 프로그램 → 문화 > 방송·미디어
   - 통계표: 시군구별 주민등록인구 → 사회 > 일반공공행정, 사망원인별 사망자수 → 의료 > 공중보건, 장애유형별 등록장애인수 → 사회 > 복지
3. 가장 대표적인 분류를 맨 앞에 두세요. 두 번째·세 번째는 분명히 걸쳐 있을 때만 붙이세요.

# 데이터셋
- 제목: {title}
- 요약: {summary}
- 키워드: {keywords}
- 컬럼: {columns}
"""

    DATASET_NATURE_CLASSIFICATION: str = """**반드시 classify_dataset_nature tool을 사용하여 응답할 것.**

공공데이터 파일 하나의 성격을 고르세요. 이름·설명·컬럼으로 판단합니다.
- aggregate: 행이 개별 개체가 아니라 지역·기간·성별·연령·유형 같은 기준으로 묶은 수치(인원·건수·비율·금액·평균)입니다. 기준이 열로 펼쳐진 표(연도가 열, 지역이 열)도 같습니다. 예: 시군구별 출생아 수, 월별 확진자 수, 상병별 환자 수 통계, 기관별 연간 이용 실적
- record: 행마다 실제 개체나 사건 하나를 기록합니다. 예: 약국·병원·업소 목록, 시험 응시자별 성적, 민원 접수 내역, 연구과제 목록
- reference: 코드표, 테이블·컬럼 정의서, 안내문처럼 다른 데이터를 설명하는 기준 정보입니다
- content: 방송·게시물·문서처럼 콘텐츠 자체가 행이거나 파일인 데이터입니다

# 데이터셋
- 이름: {title}
- 설명: {description}
- 컬럼: {columns}
"""

    DATASET_SEARCH_ANSWER: str = """**반드시 answer_dataset_search tool을 사용하여 응답할 것.**

당신은 데이터 카탈로그 검색 결과를 사용자에게 설명하는 전문가입니다.

# 사용자 질의
{query}

# 검색된 데이터셋 (유사도 내림차순)
{datasets}

# 데이터셋 간 관계 (기준: {anchor})
{relations}

# 임무
질의에 답하되, **위에 제시된 데이터셋과 관계만 근거로 삼으세요.**

## 규칙
- 목록에 없는 데이터셋을 언급하지 마세요. 존재하지 않는 데이터를 만들어내지 마세요
- 관계가 제시된 데이터셋은 **어떻게 이어지는지**를 말하세요.
  관계 유형의 뜻은 아래와 같습니다
  - DETAIL_FOR: 같은 개체의 다른 속성을 공급 (더 자세히 보기)
  - STATISTIC_FOR: 다른 입도의 집계·지표 (모수·추세로 맥락 보기)
  - SAME_KIND: 같은 종류의 다른 지역·기관·기간 데이터 (이어 붙이거나 견주기)
  - ADJACENT_TO: 같은 맥락의 다른 개체 (함께 놓고 비교하기)
- 관계가 없는 데이터셋은 **유사도로만 잡혔다**는 점을 밝히세요
- 질의에 답할 데이터가 부족하면 부족하다고 말하세요. 억지로 채우지 마세요
- 기관명·갱신주기 같은 사실은 제시된 설명에 있는 것만 쓰세요

## 답변 형식
한국어 3~5문장. 기준 데이터셋을 먼저 말하고, 무엇을 붙여 쓸 수 있는지 관계 유형에
따라 묶어 설명하세요. 불릿 없이 서술형으로 쓰세요.

suggested_next 에는 사용자가 이어서 볼 만한 데이터셋명을 **제시된 목록에서만**
최대 3개 고르세요.
"""

    DATASET_SEARCH_INTENT: str = """**반드시 extract_dataset_search_intent tool을 사용하여 응답할 것.**

당신은 자연어 질의를 데이터셋 카탈로그 검색 질의로 변환하는 전문가입니다.

# 임무
사용자 질의에서 **찾으려는 데이터의 주제**를 추출하여 임베딩 검색용 구문으로 변환하세요.

## 당신의 답변이 사용되는 방식 (중요!)
1. search_query 를 임베딩하여 등록된 데이터셋의 **요약문(table_summary) 임베딩과 코사인 유사도 검색**을 수행합니다
2. 따라서 search_query 는 질문·요청 형태가 아니라, **데이터셋 설명문과 닮은 서술형 명사구**여야 합니다

예시:
- 질의: "서울 지역 대형 병원의 장비 현황과 연결되는 데이터 찾아줘"
  → search_query: "병원 의료장비 보유 현황 및 의료기관 기본정보", keywords: ["병원", "의료장비", "의료기관"]
- 질의: "우리 회사 물류 데이터랑 같이 분석할만한 공공데이터 있어?"
  → search_query: "화물 운송 물류 현황 및 교통 인프라 정보", keywords: ["물류", "화물운송", "교통"]

## 규칙
- 지시·요청 표현("찾아줘", "연계 가능한", "~할만한", "데이터") 은 제거
- 핵심 주제 개념은 유지하고, 데이터셋 설명에 쓰일 법한 동의어·상위어로 보강
- search_query 는 한국어 한 문장(명사구) 이내로 간결하게

사용자 질의: {query}
"""

    @classmethod
    def get_prompt(cls, task: str) -> str:
        """Get fallback prompt by task name."""
        prompt_map: Dict[str, str] = {
            "key_value_extract": cls.KEY_VALUE_EXTRACT,
            "cluster_normalize": cls.CLUSTER_NORMALIZE,
            "schema_analysis": cls.SCHEMA_ANALYSIS,
            "relation_key_analysis": cls.RELATION_KEY_ANALYSIS,
            "field_type_classification": cls.FIELD_TYPE_CLASSIFICATION,
            "table_relation_verification": cls.TABLE_RELATION_VERIFICATION,
            "table_metadata_extraction": cls.TABLE_METADATA_EXTRACTION,
            "document_metadata_extraction": cls.DOCUMENT_METADATA_EXTRACTION,
            "dataset_search_intent": cls.DATASET_SEARCH_INTENT,
            "dataset_search_answer": cls.DATASET_SEARCH_ANSWER,
            "dataset_category_classification": cls.DATASET_CATEGORY_CLASSIFICATION,
            "dataset_nature_classification": cls.DATASET_NATURE_CLASSIFICATION,
        }

        if task not in prompt_map:
            raise ValueError(
                f"Unknown task: {task}. Available tasks: {', '.join(prompt_map.keys())}"
            )

        return prompt_map[task]
