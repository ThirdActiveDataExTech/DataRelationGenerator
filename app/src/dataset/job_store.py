"""백그라운드 job 상태 영속 저장소 (SQLite).

파이프라인/KV그래프 job 상태를 인메모리 dict 대신 SQLite에 보관하여
프로세스 재시작 후에도 폴링 API(GET .../{job_id})가 상태를 잃지 않게 한다.
자동 트리거(Airflow DAG)가 job 완료를 폴링하는 전제 조건.
"""

import json
import sqlite3

from app.config import settings

_TABLE = "background_job"


class JobStore:
    """job_id → (kind, payload JSON) SQLite 저장소.

    ponytail: 호출마다 커넥션 생성 — job API는 저빈도 폴링이라 풀링 불필요.
    """

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or settings.SQLITE_FILE_NAME
        with self._conn() as conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {_TABLE} ("
                "job_id TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL)"
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=30)

    def put(self, job_id: str, kind: str, data: dict) -> None:
        """job 상태 저장 (upsert)."""
        with self._conn() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO {_TABLE} (job_id, kind, payload) VALUES (?, ?, ?)",
                (job_id, kind, json.dumps(data, ensure_ascii=False)),
            )

    def update(self, job_id: str, **fields) -> None:
        """기존 payload에 필드 병합 (없는 job 이면 무시)."""
        with self._conn() as conn:
            row = conn.execute(
                f"SELECT payload FROM {_TABLE} WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None:
                return
            data = json.loads(row[0])
            data.update(fields)
            conn.execute(
                f"UPDATE {_TABLE} SET payload = ? WHERE job_id = ?",
                (json.dumps(data, ensure_ascii=False), job_id),
            )

    def get(self, job_id: str) -> dict | None:
        """job 상태 조회. 없으면 None."""
        with self._conn() as conn:
            row = conn.execute(
                f"SELECT payload FROM {_TABLE} WHERE job_id = ?", (job_id,)
            ).fetchone()
        return json.loads(row[0]) if row else None


_store: JobStore | None = None


def get_job_store() -> JobStore:
    global _store
    if _store is None:
        _store = JobStore()
    return _store
