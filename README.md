# 액티브 메타데이터 기반 데이터 관계 생성

> **[공개 SW] 액티브 메타데이터 기반 데이터 간 관계 자동 생성 기술**
> Key-Value 기반 그래프 모델을 활용한 도메인 독립적 데이터 관계 탐색

---

## 개요

다양한 데이터 소스의 메타데이터를 Key-Value 기반 그래프 구조로 변환하여, 데이터 간 관계를 자동으로 생성하는 시스템입니다.

![System Architecture](./docs/images/architecture.png)

정적 메타데이터 중심의 기존 방식을 넘어, 데이터의 특징과 사용 맥락을 반영하여 능동적으로 생성·확장·갱신되는 액티브 메타데이터 기반으로 데이터 간 의미적 관계와 잠재적 연관성을 탐색합니다.

---

## 주요 기능

![High-level Design](./docs/images/HighLevelDesign.png)

- **도메인 독립적 그래프 모델**: 특정 도메인 전용 구조 없이 Key-Value로 모든 데이터 표현
- **데이터 정규화**: 다양한 표현을 통합하여 동일 개체 언급 문서 자동 연결
- **그래프 + 전문 검색**: Neptune RDF 그래프 + OpenSearch 전문 검색 결합
- **추적 가능성**: 모든 값은 원본 문서로 추적 가능하며, 정규화 과정도 명시적 관계로 표현

---

## 데이터 구조

### Neptune RDF (그래프 저장소)

| Class | Properties | 설명 |
|-------|-----------|------|
| `kv:Document` | `kv:documentId`, `kv:binaryText` | 원본 데이터 레코드 |
| `kv:Key` | `kv:keyName`, `kv:name` | 데이터 필드명 (한글) |
| `kv:Value` | `kv:valueContent`, `kv:name` | 필드의 실제 값 |
| `kv:NormalizedValue` | `kv:name` | 정규화된 대표 값 |

**관계 (Predicates)**:
- `kv:extractedFrom`: Value → Document (값이 추출된 문서)
- `kv:hasValue`: Key → Value (키가 보유한 값)
- `kv:sameAs`: Value → NormalizedValue (정규화 관계)
- `kv:hasNormalizedValue`: Document → NormalizedValue (문서-정규화값 직접 연결)

### OpenSearch (검색 엔진)

| 인덱스 | 용도 |
|--------|------|
| `documents` | 전체 문서 전문 검색 |
| `keys` | 필드명(Key) 검색 |
| `values` | 필드값(Value) 검색 |

---

## 기술 스택

- **Backend**: FastAPI, Python 3.10-3.12
- **Graph Database**: AWS Neptune (RDF/SPARQL)
- **Search Engine**: AWS OpenSearch
- **LLM Framework**: LangChain, LangGraph
- **Package Manager**: uv

---

## 시작하기

```bash
# 의존성 설치
uv sync

# 환경 변수 설정 (.env 파일)
NEPTUNE_ENDPOINT=your-neptune-endpoint
OPENSEARCH_HOST=your-opensearch-host
ANTHROPIC_API_KEY=your-api-key

# 실행
uv run uvicorn app.main:app --reload
```

**접속**:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

---

## 기대 효과

- 도메인 독립적 구조로 다양한 데이터셋 적용 가능
- 그래프 기반 데이터 관계 자동 생성으로 탐색/추천/연계 자동화
- 정규화 기반 유사 문서 탐색 및 데이터 통합
