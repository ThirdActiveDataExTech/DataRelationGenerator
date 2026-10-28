-- 정형 데이터셋 저장소

CREATE EXTENSION IF NOT EXISTS vector;

-- 카탈로그 — 데이터셋 1건 = 1행
CREATE TABLE IF NOT EXISTS dcat_dataset
(
    dataset_id     TEXT PRIMARY KEY,                       -- dct:identifier
    source         TEXT        NOT NULL,                   -- 원천 포털 코드 (DATA_GO_KR / KOSIS / HIRA / FIRST_PARTY / OTHER)
    title          TEXT        NOT NULL,                   -- dct:title
    description    TEXT,                                   -- dct:description
    publisher      TEXT,                                   -- dct:publisher · 제공기관
    source_type    TEXT        NOT NULL DEFAULT 'structured_file',
    dataset_origin TEXT        NOT NULL DEFAULT 'admin',   -- admin / user
    publisher_code TEXT,                                   -- 원천 포털의 기관 코드
    keywords       JSONB       NOT NULL DEFAULT '[]',      -- dcat:keyword
    theme          JSONB       NOT NULL DEFAULT '[]',      -- dct:subject · 원천 분류 그대로 (공공 '대분류 - 소분류' 분해, KOSIS 주제 경로)
    category       JSONB       NOT NULL DEFAULT '[]',      -- 디지털 집현전 분류 '대분류 > 중분류' 1~3개 (첫 번째가 대표)
    language       TEXT        NOT NULL DEFAULT 'ko',      -- dct:language
    spatial        TEXT,                                   -- dct:spatial · 원문
    temporal_start DATE,                                   -- dct:temporal 시작
    temporal_end   DATE,                                   -- dct:temporal 종료 · NULL 이면 진행 중/미상
    temporal_resolution TEXT,                              -- dcat:temporalResolution · P1Y / P6M / P3M / P1M / P1D
    accrual_periodicity TEXT,                              -- dct:accrualPeriodicity · ANNUAL / MONTHLY / IRREGULAR …
    license        TEXT,                                   -- dct:license · KOGL-0~4 / OPEN / KOSIS-TERMS
    rights         TEXT,                                   -- dct:rights · 이용허락 원문
    item_count     INTEGER,                                -- 구성 데이터파일 수
    landing_page   TEXT,                                   -- dcat:landingPage
    access_url     TEXT,                                   -- dcat:accessURL
    external_ids   JSONB       NOT NULL DEFAULT '[]',      -- adms:identifier · 통합 카탈로그 식별자 등
    party          TEXT,                                   -- 1st / 3rd
    visibility     TEXT        NOT NULL DEFAULT 'PUBLIC',  -- PUBLIC / ORG / PRIVATE
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_dcat_dataset_publisher ON dcat_dataset (publisher);
CREATE INDEX IF NOT EXISTS idx_dcat_dataset_party ON dcat_dataset (party);
CREATE INDEX IF NOT EXISTS idx_dcat_dataset_keywords ON dcat_dataset USING gin (keywords);
CREATE INDEX IF NOT EXISTS idx_dcat_dataset_theme ON dcat_dataset USING gin (theme);
CREATE INDEX IF NOT EXISTS idx_dcat_dataset_category ON dcat_dataset USING gin (category);
CREATE INDEX IF NOT EXISTS idx_dcat_dataset_source ON dcat_dataset (source);
CREATE INDEX IF NOT EXISTS idx_dcat_dataset_periodicity ON dcat_dataset (accrual_periodicity);
CREATE INDEX IF NOT EXISTS idx_dcat_dataset_temporal ON dcat_dataset (temporal_start, temporal_end);

-- 데이터셋 현황 메타데이터 — DCAT 에 대응 속성이 없는 소스 고유 항목. 데이터셋 1건 = 1행
CREATE TABLE IF NOT EXISTS dcat_dataset_attr
(
    dataset_id             TEXT PRIMARY KEY REFERENCES dcat_dataset (dataset_id) ON DELETE CASCADE ON UPDATE CASCADE,
    list_type              TEXT,                           -- FILE / API / STD / STAT(KOSIS 통계표)
    stat_name              TEXT,                           -- KOSIS 통계명 — 같은 통계의 통계표 묶음
    spatial_unit           TEXT,                           -- 집계 단위 NATION / SIDO / SIGUNGU / EMD
    next_registration_date DATE,                           -- 원천 공시 차기 등록 예정일
    media_type             TEXT,
    provision_form         TEXT,                           -- 포털 다운로드 / 기관 URL 등
    api_type               TEXT,
    allowed_traffic        TEXT,                           -- 원문이 문장이라 text
    review_type            TEXT,
    national_core_yn       BOOLEAN,
    standard_data_yn       BOOLEAN,
    cost_charged           TEXT,                           -- 무료 / 유료
    cost_basis             TEXT,
    dept_name              TEXT,
    dept_phone             TEXT,
    retention_basis        TEXT,                           -- 보유근거 · 법적 근거
    collection_method      TEXT,                           -- 수집방법 · 작성방법
    data_limitation        TEXT,
    other_notes            TEXT,                           -- 기타 유의사항 · 통계표 주석
    updated_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_dcat_dataset_attr_list_type ON dcat_dataset_attr (list_type);
CREATE INDEX IF NOT EXISTS idx_dcat_dataset_attr_stat_name ON dcat_dataset_attr (stat_name);

-- 데이터셋 이력 메타데이터 — 지표의 시점 관측값 (변화량은 조회 시 계산)
CREATE TABLE IF NOT EXISTS active_meta_dataset
(
    id           BIGSERIAL PRIMARY KEY,
    dataset_id   TEXT          NOT NULL REFERENCES dcat_dataset (dataset_id) ON DELETE CASCADE ON UPDATE CASCADE,
    metric_code  TEXT          NOT NULL,                   -- 지표 표준값 · FK 는 003_metric_registry.sql
    metric_value NUMERIC(20, 4) NOT NULL,
    observed_at  TIMESTAMPTZ   NOT NULL,
    source       TEXT          NOT NULL,                   -- PORTAL_CATALOG / PORTAL_LOG / CRAWLER
    UNIQUE (dataset_id, metric_code, observed_at)
);
CREATE INDEX IF NOT EXISTS idx_dcat_dataset_external_ids ON dcat_dataset USING gin (external_ids);

-- 데이터파일 현황 메타데이터 — 파일 1건 = 1행 (v0.1 OpenSearch datasets 인덱스를 흡수)
CREATE TABLE IF NOT EXISTS meta_datagokr_file
(
    file_id           TEXT PRIMARY KEY,                    -- 내용 해시 기반
    dataset_id        TEXT        NOT NULL REFERENCES dcat_dataset (dataset_id) ON DELETE CASCADE ON UPDATE CASCADE,
    file_name         TEXT        NOT NULL,                -- 관계 서비스 API 의 table_name
    display_name      TEXT,                                -- LLM 생성 한국어 표시명
    file_format       TEXT,                                -- CSV / XLSX / JSON / API / KOSIS
    byte_size         BIGINT,
    content_sha256    TEXT        NOT NULL UNIQUE,         -- 중복 등록 판별 (메타데이터만이면 구조 해시)
    storage_key       TEXT,                                -- 원본 보관 경로
    record_count      BIGINT,
    column_count      INTEGER,
    column_names      JSONB       NOT NULL DEFAULT '[]',
    column_profile    JSONB       NOT NULL DEFAULT '{}',   -- 항목별 기술통계량 · KOSIS 분류별 값·항목별 단위
    coverage_start    DATE,                                -- 수록기간 시작
    coverage_end      DATE,                                -- 수록기간 종료 (공공은 파일명 날짜 접미)
    domain            TEXT,                                -- 대표 대분류 (dcat_dataset.category 첫 번째의 대분류)
    table_summary     TEXT,                                -- 임베딩 소스 텍스트
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- 관계 서비스는 file_name(table_name)을 키로 조회·대체 등록한다
CREATE UNIQUE INDEX IF NOT EXISTS uq_meta_datagokr_file_name ON meta_datagokr_file (file_name);
CREATE INDEX IF NOT EXISTS idx_meta_datagokr_file_dataset ON meta_datagokr_file (dataset_id);
CREATE INDEX IF NOT EXISTS idx_meta_datagokr_file_domain ON meta_datagokr_file (domain);
CREATE INDEX IF NOT EXISTS idx_meta_datagokr_file_columns ON meta_datagokr_file USING gin (column_names);

-- 데이터파일 임베딩 — 1:1 분리 (재임베딩이 본 테이블에 잠금을 걸지 않는다)
CREATE TABLE IF NOT EXISTS meta_datagokr_file_vec
(
    file_id            TEXT PRIMARY KEY REFERENCES meta_datagokr_file (file_id) ON DELETE CASCADE,
    summary_embedding  vector(768),                        -- table_summary
    keywords_embedding vector(768),                        -- 키워드 평균
    embed_text         TEXT,
    model_name         TEXT        NOT NULL,
    embedded_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_meta_datagokr_file_vec_summary
    ON meta_datagokr_file_vec USING hnsw (summary_embedding vector_cosine_ops);

-- 데이터셋 간 관계 — 쌍+유형+method당 1행 · 재생성 시 같은 method 행만 대체 · 정본은 dcat_dataset_relation_canonical 뷰(004)
CREATE TABLE IF NOT EXISTS dcat_dataset_relation
(
    id                BIGSERIAL PRIMARY KEY,
    source_dataset_id TEXT        NOT NULL REFERENCES dcat_dataset (dataset_id) ON DELETE CASCADE ON UPDATE CASCADE,
    target_dataset_id TEXT        NOT NULL REFERENCES dcat_dataset (dataset_id) ON DELETE CASCADE ON UPDATE CASCADE,
    relation_type     TEXT        NOT NULL,                -- 관계 유형 정의서 4종(:하위라벨)
    score             REAL        NOT NULL,                -- 0~1
    method            TEXT        NOT NULL,                -- rule / embedding / llm (와이즈넛) · gnn (모비젠)
    status            TEXT        NOT NULL,                -- PENDING / APPROVED / REJECTED
    evidence          JSONB       NOT NULL DEFAULT '{}',   -- 매칭 컬럼, 유사도, LLM 사유
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT dcat_dataset_relation_pair_type_method_key
        UNIQUE (source_dataset_id, target_dataset_id, relation_type, method),
    CONSTRAINT dcat_dataset_relation_method_check CHECK (method IN ('rule', 'embedding', 'llm', 'gnn'))
);
CREATE INDEX IF NOT EXISTS idx_dcat_dataset_relation_type ON dcat_dataset_relation (relation_type);
CREATE INDEX IF NOT EXISTS idx_dcat_dataset_relation_status ON dcat_dataset_relation (status);
CREATE INDEX IF NOT EXISTS idx_dcat_dataset_relation_target ON dcat_dataset_relation (target_dataset_id);
