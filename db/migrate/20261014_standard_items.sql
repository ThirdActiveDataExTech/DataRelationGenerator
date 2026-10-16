-- 표준 항목 정의서 반영 — 이미 떠 있는 원장용 (새 볼륨은 init/001 이 같은 구조로 만든다)
--   cat db/migrate/20261014_standard_items.sql db/init/001_dcat_dataset.sql \
--     | docker compose -f db/docker-compose.yml exec -T postgres psql -U wisenut -d datagokr -v ON_ERROR_STOP=1
-- 새 테이블(dcat_dataset_attr·active_meta_dataset)과 인덱스는 뒤따르는 init/001(전부 IF NOT EXISTS)이 만든다.
-- 재실행해도 안전하다. 끝나면 db/export_seed.sh 로 시드를 다시 내보낸다.
BEGIN;

ALTER TABLE dcat_dataset
    ADD COLUMN IF NOT EXISTS publisher_code      TEXT,
    ADD COLUMN IF NOT EXISTS category            JSONB NOT NULL DEFAULT '[]',
    ADD COLUMN IF NOT EXISTS language            TEXT  NOT NULL DEFAULT 'ko',
    ADD COLUMN IF NOT EXISTS spatial             TEXT,
    ADD COLUMN IF NOT EXISTS temporal_start      DATE,
    ADD COLUMN IF NOT EXISTS temporal_end        DATE,
    ADD COLUMN IF NOT EXISTS temporal_resolution TEXT,
    ADD COLUMN IF NOT EXISTS accrual_periodicity TEXT,
    ADD COLUMN IF NOT EXISTS license             TEXT,
    ADD COLUMN IF NOT EXISTS rights              TEXT,
    ADD COLUMN IF NOT EXISTS item_count          INTEGER;

-- 지금까지 theme 에 넣던 디지털 집현전 분류를 category 로 옮기고, theme 은 원천 분류 자리로 비운다
UPDATE dcat_dataset SET category = theme, theme = '[]'
WHERE category = '[]' AND theme::text LIKE '%>%';

ALTER TABLE meta_datagokr_file
    ADD COLUMN IF NOT EXISTS byte_size      BIGINT,
    ADD COLUMN IF NOT EXISTS coverage_start DATE,
    ADD COLUMN IF NOT EXISTS coverage_end   DATE;

-- dataset_id 형식을 바꿔도 파일·관계가 따라가도록
ALTER TABLE meta_datagokr_file DROP CONSTRAINT IF EXISTS meta_datagokr_file_dataset_id_fkey,
    ADD CONSTRAINT meta_datagokr_file_dataset_id_fkey FOREIGN KEY (dataset_id)
        REFERENCES dcat_dataset (dataset_id) ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE dcat_dataset_relation DROP CONSTRAINT IF EXISTS dcat_dataset_relation_source_dataset_id_fkey,
    ADD CONSTRAINT dcat_dataset_relation_source_dataset_id_fkey FOREIGN KEY (source_dataset_id)
        REFERENCES dcat_dataset (dataset_id) ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE dcat_dataset_relation DROP CONSTRAINT IF EXISTS dcat_dataset_relation_target_dataset_id_fkey,
    ADD CONSTRAINT dcat_dataset_relation_target_dataset_id_fkey FOREIGN KEY (target_dataset_id)
        REFERENCES dcat_dataset (dataset_id) ON DELETE CASCADE ON UPDATE CASCADE;

COMMIT;
