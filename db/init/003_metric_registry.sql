-- 지표 표준값 — 시드 적재(002) 다음에 실행된다 (이름 순). 적재된 지표 코드를 표준값으로 맞춘 뒤 FK 를 건다
-- 이미 떠 있는 원장에도 그대로 실행한다. 재실행해도 안전하다:
--   docker compose -f db/docker-compose.yml exec -T postgres psql -U wisenut -d datagokr -v ON_ERROR_STOP=1 \
--     < db/init/003_metric_registry.sql

-- 지표 코드 1개 = 1행 · active_meta_dataset / active_meta_file 이 참조
CREATE TABLE IF NOT EXISTS metric_registry
(
    metric_code    TEXT PRIMARY KEY,
    label          TEXT  NOT NULL,                         -- 화면 표기명
    layer          TEXT  NOT NULL,                         -- dataset / file
    value_type     TEXT  NOT NULL DEFAULT 'count',         -- count / ratio / index
    unit           TEXT,                                   -- 건 / 회 / 개
    applies_to     JSONB NOT NULL,                         -- 적용 source_type 배열 · ["*"] = 전체 공통
    default_source TEXT,                                   -- PORTAL_CATALOG / PORTAL_LOG / CRAWLER / API_*
    description    TEXT                                    -- 지표 정의 · 산출 방식
);
CREATE INDEX IF NOT EXISTS idx_metric_registry_layer ON metric_registry (layer);
CREATE INDEX IF NOT EXISTS idx_metric_registry_applies ON metric_registry USING gin (applies_to);

-- 표준값 시드 — 정형은 와이즈넛이 넣고, 비정형 코드는 모비젠이 추가한다
INSERT INTO metric_registry (metric_code, label, layer, value_type, unit, applies_to, default_source, description)
VALUES ('download_request_count', '다운로드·활용신청 건수', 'dataset', 'count', '건', '["structured_file"]', 'PORTAL_CATALOG',
        '원천 포털 목록개방현황의 누적 건수. 증분은 직전 관측값과의 차이로 조회 시 산출'),
       ('view_count', '조회수', 'dataset', 'count', '회', '["structured_file"]', 'PORTAL_CATALOG',
        '원천 포털 누적 조회수'),
       ('file_count', '파일 수', 'dataset', 'count', '개', '["structured_file"]', 'CRAWLER',
        '데이터셋을 구성하는 파일 개수'),
       ('record_count', '데이터 건수', 'file', 'count', '건', '["structured_file"]', 'CRAWLER',
        '파일의 전체 행 수. 재수집 시 값이 변한다'),
       ('portal_download_count', '포털 다운로드 건수', 'file', 'count', '건', '["*"]', 'PORTAL_LOG',
        '케이웨어 cetus_dataset_file.down_cnt 의 시점 관측값'),
       ('portal_search_hit_count', '포털 검색 노출 건수', 'dataset', 'count', '회', '["*"]', 'PORTAL_LOG',
        '케이웨어 검색 로그 기반 노출 건수')
ON CONFLICT (metric_code) DO NOTHING;

-- 데이터셋 이력 메타데이터 — 지표 코드를 표준값으로 맞추고 FK 를 건다
UPDATE active_meta_dataset SET metric_code = 'download_request_count' WHERE metric_code = 'download_request';
ALTER TABLE active_meta_dataset DROP CONSTRAINT IF EXISTS active_meta_dataset_metric_code_fkey,
    ADD CONSTRAINT active_meta_dataset_metric_code_fkey FOREIGN KEY (metric_code)
        REFERENCES metric_registry (metric_code) ON UPDATE CASCADE;

-- 데이터파일 이력 메타데이터 — 구조는 active_meta_dataset 과 같고 참조 대상만 파일
CREATE TABLE IF NOT EXISTS active_meta_file
(
    id           BIGSERIAL PRIMARY KEY,
    file_id      TEXT          NOT NULL REFERENCES meta_datagokr_file (file_id) ON DELETE CASCADE,
    metric_code  TEXT          NOT NULL REFERENCES metric_registry (metric_code) ON UPDATE CASCADE,
    metric_value NUMERIC(20, 4) NOT NULL,
    observed_at  TIMESTAMPTZ   NOT NULL,
    source       TEXT          NOT NULL,                   -- PORTAL_LOG / CRAWLER
    UNIQUE (file_id, metric_code, observed_at)
);
