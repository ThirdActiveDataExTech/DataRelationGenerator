#!/bin/sh
# 시드 적재 — /seed 의 *.sql.gz 를 이름 순으로 넣는다 (볼륨이 비어 있을 때 한 번만 실행됨).
# 시드가 없으면 빈 원장으로 뜬다. 시드 만들기: db/export_seed.sh
set -eu

found=0
for f in /seed/*.sql.gz; do
  [ -e "$f" ] || continue
  found=1
  echo "[seed] $f 적재"
  gunzip -c "$f" | psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" --quiet
done

[ "$found" = 1 ] || echo "[seed] /seed 에 *.sql.gz 없음 — 빈 원장으로 시작"
