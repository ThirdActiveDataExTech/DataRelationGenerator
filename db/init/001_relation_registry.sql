-- 관계 유형 표준값 — 001_dcat_dataset.sql 다음, 시드 적재(002) 전에 실행된다 (이름 순)
-- 이미 떠 있는 원장에도 그대로 실행한다. 재실행해도 안전하며, 이 파일의 표준값으로 갱신된다:
--   docker compose -f db/docker-compose.yml exec -T postgres psql -U wisenut -d datagokr -v ON_ERROR_STOP=1 \
--     < db/init/001_relation_registry.sql
-- 기존 원장에 적용한 뒤에는 db/export_seed.sh 로 시드를 다시 내보낸다 (예전 시드는 유형 FK 에 걸린다).

-- 관계 유형 1개 = 1행 · dcat_dataset_relation / meta_file_relation 이 참조
CREATE TABLE IF NOT EXISTS relation_type_registry
(
    relation_type   TEXT PRIMARY KEY,
    label           TEXT  NOT NULL,                        -- 화면 표기명
    scope           TEXT  NOT NULL,                        -- dataset / file / both (저장 테이블 구분)
    applies_to      JSONB NOT NULL,                        -- 적용 source_type 배열 · ["*"] = 전체 공통
    required_fields JSONB NOT NULL,                        -- 판정에 쓰는 항목
    description     TEXT                                   -- 판정 근거 · 방향
);
CREATE INDEX IF NOT EXISTS idx_relation_type_registry_scope ON relation_type_registry (scope);
CREATE INDEX IF NOT EXISTS idx_relation_type_registry_applies ON relation_type_registry USING gin (applies_to);

-- 표준값 시드 — 관계 유형 4종(DETAIL_FOR·STATISTIC_FOR·ADJACENT_TO·SAME_KIND) + 비정형 공용 2종 + 파일 단위 2종.
-- 문서형 파일(unstructured_text)은 컬럼이 없어 제목·설명·키워드로 판정하는 인접 맥락에 든다.
-- 비정형 전용 코드는 모비젠이 추가한다
INSERT INTO relation_type_registry (relation_type, label, scope, applies_to, required_fields, description)
VALUES ('DETAIL_FOR', '속성 보완', 'both', '["structured_file"]', '["column_names", "column_profile"]',
        '동일 대상의 추가 속성. 공통 식별 컬럼으로 결합하며 결합 컬럼은 meta_file_relation 에 기록. source = 속성을 주는 쪽, target = 기준'),
       ('STATISTIC_FOR', '집계 맥락', 'dataset', '["structured_file"]', '["column_names", "column_profile"]',
        '비교 기준이 되는 통계. source = 집계표, target = 기준'),
       ('ADJACENT_TO', '인접 맥락', 'dataset', '["structured_file", "unstructured_text"]', '["title", "description", "keywords"]',
        '연관 주제의 인접 데이터. 대칭'),
       ('SAME_KIND', '동종 판본', 'dataset', '["structured_file"]', '["column_names"]',
        '같은 표준을 따르거나 컬럼 구성이 같은 타 지역·기관 데이터. 대칭'),
       ('SEMANTICALLY_RELATED', '의미 유사', 'dataset', '["*"]', '["title", "description"]',
        '제목·설명 임베딩(dcat_dataset_vec)의 코사인 유사도'),
       ('KEYWORD_OVERLAP', '키워드 유사', 'dataset', '["*"]', '["keywords"]',
        '키워드 집합의 자카드 계수'),
       ('TIMESERIES_SUCCESSOR', '시계열 연속', 'file', '["structured_file"]', '["coverage_start", "coverage_end"]',
        '같은 데이터셋의 수록기간이 이어지는 파일. 컬럼 없이 파일 간 관계로 기록'),
       ('COLUMN_NAME_SIMILAR', '컬럼명 유사', 'file', '["structured_file"]', '["column_names"]',
        '컬럼 논리·물리명의 문자열·의미 유사도. 데이터셋 관계로 후보 쌍을 좁힌 뒤 산출')
ON CONFLICT (relation_type) DO UPDATE SET
    label = EXCLUDED.label, scope = EXCLUDED.scope, applies_to = EXCLUDED.applies_to,
    required_fields = EXCLUDED.required_fields, description = EXCLUDED.description;

-- 데이터셋 간 관계 — '유형:하위라벨' 한 문자열을 유형(표준값 FK) + 하위라벨(개방형)로 나눈다.
-- 앱은 조회 시 '유형:하위라벨'로 다시 합쳐 다룬다
ALTER TABLE dcat_dataset_relation ADD COLUMN IF NOT EXISTS relation_label TEXT;
UPDATE dcat_dataset_relation
SET relation_type  = split_part(relation_type, ':', 1),
    relation_label = NULLIF(substr(relation_type, strpos(relation_type, ':') + 1), '')
WHERE strpos(relation_type, ':') > 0;
ALTER TABLE dcat_dataset_relation DROP CONSTRAINT IF EXISTS dcat_dataset_relation_relation_type_fkey,
    ADD CONSTRAINT dcat_dataset_relation_relation_type_fkey FOREIGN KEY (relation_type)
        REFERENCES relation_type_registry (relation_type) ON UPDATE CASCADE;

-- 데이터파일·컬럼 간 관계 — 컬럼명이 NULL 이면 파일 간, 값이 있으면 컬럼 간
CREATE TABLE IF NOT EXISTS meta_file_relation
(
    id             BIGSERIAL PRIMARY KEY,
    source_file_id TEXT        NOT NULL REFERENCES meta_datagokr_file (file_id) ON DELETE CASCADE,
    source_column  TEXT,
    target_file_id TEXT        NOT NULL REFERENCES meta_datagokr_file (file_id) ON DELETE CASCADE,
    target_column  TEXT,
    relation_type  TEXT        NOT NULL REFERENCES relation_type_registry (relation_type) ON UPDATE CASCADE,
    score          REAL        NOT NULL,                   -- 0~1
    evidence       JSONB       NOT NULL DEFAULT '{}',      -- 값 중첩률, 타입 일치, 명칭 유사도
    status         TEXT        NOT NULL,                   -- PENDING / APPROVED / REJECTED
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- 파일 간 관계(컬럼 NULL)도 쌍+유형당 1행
    UNIQUE NULLS NOT DISTINCT (source_file_id, source_column, target_file_id, target_column, relation_type)
);
CREATE INDEX IF NOT EXISTS idx_meta_file_relation_type ON meta_file_relation (relation_type);
CREATE INDEX IF NOT EXISTS idx_meta_file_relation_target ON meta_file_relation (target_file_id);
