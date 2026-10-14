"""PostgreSQL(pgvector) 접속 — 데이터셋 원장."""

from collections.abc import Sequence

import psycopg
from psycopg.rows import DictRow, dict_row

from app.config import settings


def connect() -> psycopg.Connection[DictRow]:
    """원장 접속. `with connect() as conn:` 블록이 끝나면 커밋(예외 시 롤백) 후 닫힌다.

    ponytail: 호출마다 새 접속. 동시 요청이 늘면 psycopg_pool.ConnectionPool 로 교체
    """
    return psycopg.Connection[DictRow].connect(settings.POSTGRES_DSN, row_factory=dict_row)


def to_vector(values: Sequence[float]) -> str | None:
    """pgvector 입력 리터럴 ('[0.1,0.2]'). 빈 임베딩은 NULL."""
    return f"[{','.join(map(str, values))}]" if values else None


def from_vector(text: str | None) -> list[float]:
    """pgvector 텍스트 출력('[0.1,0.2]') → float 목록."""
    return [float(v) for v in text.strip("[]").split(",")] if text else []
