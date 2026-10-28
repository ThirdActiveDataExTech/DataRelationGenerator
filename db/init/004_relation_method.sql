-- 관계를 method 단위로 보관 — 003 다음에 실행된다 (이름 순)
-- 같은 쌍·유형에 규칙 기반(rule / embedding / llm)과 모델 기반(gnn) 결과를 함께 두고, 조회는 정본 뷰를 읽는다
-- 이미 떠 있는 원장에도 그대로 실행한다. 재실행해도 안전하다:
--   docker compose -f db/docker-compose.yml exec -T postgres psql -U wisenut -d datagokr -v ON_ERROR_STOP=1 \
--     < db/init/004_relation_method.sql

-- 쌍+유형당 1행 → 쌍+유형+method당 1행. method 값은 4종으로 고정
ALTER TABLE dcat_dataset_relation
    DROP CONSTRAINT IF EXISTS dcat_dataset_relation_source_dataset_id_target_dataset_id_r_key,
    DROP CONSTRAINT IF EXISTS dcat_dataset_relation_pair_type_method_key,
    ADD CONSTRAINT dcat_dataset_relation_pair_type_method_key
        UNIQUE (source_dataset_id, target_dataset_id, relation_type, method),
    DROP CONSTRAINT IF EXISTS dcat_dataset_relation_method_check,
    ADD CONSTRAINT dcat_dataset_relation_method_check CHECK (method IN ('rule', 'embedding', 'llm', 'gnn'));

-- 정본 — 쌍+유형당 1행. APPROVED 우선 → score 최대 → created_at 최신
DROP VIEW IF EXISTS dcat_dataset_relation_canonical;
CREATE VIEW dcat_dataset_relation_canonical AS
SELECT DISTINCT ON (source_dataset_id, target_dataset_id, relation_type) *
FROM dcat_dataset_relation
ORDER BY source_dataset_id, target_dataset_id, relation_type,
         (status = 'APPROVED') DESC, score DESC, created_at DESC;
