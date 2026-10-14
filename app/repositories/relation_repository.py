"""데이터셋 간 관계 저장소 — dcat_dataset_relation (승인 상태의 정본)."""

from typing import Any, Literal

from psycopg.types.json import Jsonb

from app.clients.postgres import connect
from app.exceptions.service import DatasetNotFoundError
from app.schemas.dataset import DatasetRelation

RelationStatus = Literal["PENDING", "APPROVED", "REJECTED"]


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
                                                   score, method, status, evidence)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (source, target, relation.relation_type, relation.score, relation.method, status,
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
                """
                SELECT fa.file_name AS source, fb.file_name AS target, r.relation_type, r.score, r.method,
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
                "catalog_identifier_a": (r["ids_a"] or [""])[0],
                "catalog_identifier_b": (r["ids_b"] or [""])[0],
            }
            for r in rows
        ]


def get_relation_repository() -> RelationRepository:
    return RelationRepository()
