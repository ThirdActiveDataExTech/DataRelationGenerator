"""데이터셋 간 관계 저장소 — dcat_dataset_relation (승인 상태의 정본)."""

from typing import Any, Literal

from psycopg.types.json import Jsonb

from app.clients.postgres import connect
from app.exceptions.service import DatasetNotFoundError, RelationNotFoundError
from app.schemas.dataset import DatasetRelation

RelationStatus = Literal["PENDING", "APPROVED", "REJECTED"]

# 원장은 유형(표준값 FK)과 하위라벨을 나눠 갖고, 앱은 '유형:하위라벨' 한 문자열로 다룬다
_TYPE_WITH_LABEL = "concat_ws(':', r.relation_type, r.relation_label) AS relation_type"

_REVIEW_SELECT = f"""
SELECT r.id, fa.file_name AS a, fb.file_name AS b, fa.display_name AS name_a, fb.display_name AS name_b,
       {_TYPE_WITH_LABEL}, r.score, r.method, r.status, r.evidence, r.created_at,
       da.external_ids AS ids_a, db.external_ids AS ids_b
FROM dcat_dataset_relation r
JOIN meta_datagokr_file fa ON fa.dataset_id = r.source_dataset_id
JOIN meta_datagokr_file fb ON fb.dataset_id = r.target_dataset_id
JOIN dcat_dataset da ON da.dataset_id = r.source_dataset_id
JOIN dcat_dataset db ON db.dataset_id = r.target_dataset_id
"""


def _catalog_id(ids: list[str] | None) -> str:
    """통합 카탈로그 식별자 — 원천 식별자('datagokr:'·'kosis:' 접두)를 뺀 첫 값."""
    return next((i for i in ids or [] if ":" not in i), "")


def _review_row(r: dict) -> dict[str, Any]:
    return {
        "id": r["id"], "table_a": r["a"], "table_b": r["b"],
        "name_a": r["name_a"] or r["a"], "name_b": r["name_b"] or r["b"],
        "relation_type": r["relation_type"], "score": r["score"], "method": r["method"], "status": r["status"],
        "evidence": r["evidence"], "created_at": r["created_at"].isoformat(),
        "catalog_identifier_a": _catalog_id(r["ids_a"]), "catalog_identifier_b": _catalog_id(r["ids_b"]),
    }


class RelationRepository:
    """데이터셋 쌍의 관계. 관계 서비스는 데이터셋을 file_name(table_name)으로 가리킨다."""

    def replace(self, relation: DatasetRelation, status: RelationStatus) -> None:
        """쌍(방향 무관)의 기존 관계를 지우고 새로 기록한다 — 재생성 시 대체.

        Raises:
            DatasetNotFoundError: 어느 한쪽 데이터셋이 원장에 없음
        """
        with connect() as conn:
            rows = conn.execute(
                "SELECT file_name, dataset_id FROM meta_datagokr_file WHERE file_name = ANY(%s)",
                ([relation.table_a, relation.table_b],),
            ).fetchall()
            ids = {r["file_name"]: r["dataset_id"] for r in rows}
            missing = next((t for t in (relation.table_a, relation.table_b) if t not in ids), None)
            if missing:
                raise DatasetNotFoundError(missing)
            source, target = ids[relation.table_a], ids[relation.table_b]
            relation_type, _, label = relation.relation_type.partition(":")
            conn.execute(
                """
                DELETE FROM dcat_dataset_relation
                WHERE (source_dataset_id = %(s)s AND target_dataset_id = %(t)s)
                   OR (source_dataset_id = %(t)s AND target_dataset_id = %(s)s)
                """,
                {"s": source, "t": target},
            )
            conn.execute(
                """
                INSERT INTO dcat_dataset_relation (source_dataset_id, target_dataset_id, relation_type,
                                                   relation_label, score, method, status, evidence)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (source, target, relation_type, label or None, relation.score, relation.method, status,
                 Jsonb(relation.evidence)),
            )

    def pairs(self) -> set[tuple[str, str]]:
        """판정 이력이 있는 쌍(상태 무관) — 정렬된 (table_a, table_b) 튜플."""
        with connect() as conn:
            rows = conn.execute(
                """
                SELECT fa.file_name AS a, fb.file_name AS b
                FROM dcat_dataset_relation r
                JOIN meta_datagokr_file fa ON fa.dataset_id = r.source_dataset_id
                JOIN meta_datagokr_file fb ON fb.dataset_id = r.target_dataset_id
                """
            ).fetchall()
        return {tuple(sorted((r["a"], r["b"]))) for r in rows}

    def edges(self, table_names: list[str] | None = None) -> list[dict[str, Any]]:
        """확정(APPROVED) 관계. table_names 를 주면 그중 하나가 끼어 있는 관계만 (최신 순).

        Returns:
            [{"source","target","relation_type","score","method","evidence","created_at",
              "catalog_identifier_a","catalog_identifier_b"}]
        """
        with connect() as conn:
            rows = conn.execute(
                f"""
                SELECT fa.file_name AS source, fb.file_name AS target, {_TYPE_WITH_LABEL}, r.score, r.method,
                       r.evidence, r.created_at, da.external_ids AS ids_a, db.external_ids AS ids_b
                FROM dcat_dataset_relation r
                JOIN meta_datagokr_file fa ON fa.dataset_id = r.source_dataset_id
                JOIN meta_datagokr_file fb ON fb.dataset_id = r.target_dataset_id
                JOIN dcat_dataset da ON da.dataset_id = r.source_dataset_id
                JOIN dcat_dataset db ON db.dataset_id = r.target_dataset_id
                WHERE r.status = 'APPROVED'
                  AND (%(names)s::text[] IS NULL OR fa.file_name = ANY(%(names)s) OR fb.file_name = ANY(%(names)s))
                ORDER BY r.created_at DESC
                """,
                {"names": table_names},
            ).fetchall()
        return [
            {
                "source": r["source"],
                "target": r["target"],
                "relation_type": r["relation_type"],
                "score": r["score"],
                "method": r["method"],
                "evidence": r["evidence"],
                "created_at": r["created_at"].isoformat(),
                "catalog_identifier_a": _catalog_id(r["ids_a"]),
                "catalog_identifier_b": _catalog_id(r["ids_b"]),
            }
            for r in rows
        ]

    def review_rows(self, status: RelationStatus | None = None) -> list[dict[str, Any]]:
        """판정 화면용 관계 목록 (상태 필터, 최근 순)."""
        with connect() as conn:
            rows = conn.execute(
                _REVIEW_SELECT + " WHERE %(s)s::text IS NULL OR r.status = %(s)s ORDER BY r.created_at DESC",
                {"s": status},
            ).fetchall()
        return [_review_row(r) for r in rows]

    def set_status(self, relation_id: int, status: RelationStatus) -> dict[str, Any]:
        """관계 상태를 바꾸고 바뀐 행을 돌려준다.

        Raises:
            RelationNotFoundError: 없는 관계 id
        """
        with connect() as conn:
            if not conn.execute(
                "UPDATE dcat_dataset_relation SET status = %s WHERE id = %s RETURNING id", (status, relation_id)
            ).fetchone():
                raise RelationNotFoundError(relation_id)
            row = conn.execute(_REVIEW_SELECT + " WHERE r.id = %s", (relation_id,)).fetchone()
        return _review_row(row)


def get_relation_repository() -> RelationRepository:
    return RelationRepository()
