#!/bin/sh
# 현재 원장을 시드로 내보낸다 — 데이터셋·속성·파일·임베딩·관계·이력 6개 테이블의 데이터(스키마 제외).
# 새 볼륨으로 띄우면 init/002_load_seed.sh 가 이 파일을 자동으로 적재한다.
#
#   db/export_seed.sh      # → $SEED_DIR/datagokr_seed.sql.gz (SEED_DIR 은 db/.env, 없으면 db/seed)
set -eu

cd "$(dirname "$0")"
[ -f .env ] && . ./.env
seed_dir="${SEED_DIR:-./seed}"
mkdir -p "$seed_dir"
out="$seed_dir/datagokr_seed.sql.gz"

docker compose exec -T postgres pg_dump --username wisenut --dbname datagokr \
  --data-only --no-owner \
  --table dcat_dataset --table dcat_dataset_attr --table meta_datagokr_file --table meta_datagokr_file_vec \
  --table dcat_dataset_relation --table active_meta_dataset \
  | gzip -9 > "$out"

counts=$(docker compose exec -T postgres psql --username wisenut --dbname datagokr --tuples-only --no-align -c \
  "SELECT (SELECT count(*) FROM dcat_dataset) || ' 데이터셋, ' || (SELECT count(*) FROM dcat_dataset_relation) || ' 관계'")
echo "$out ($(du -h "$out" | cut -f1)) — $counts"
