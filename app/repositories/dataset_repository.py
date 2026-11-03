"""데이터셋 원장 저장소 — PostgreSQL(pgvector).

DatasetProfile 1건을 정의서의 세 테이블에 나눠 둔다.
- dcat_dataset: 카탈로그 레코드 (제목·설명·제공기관·파티·식별자)
- meta_datagokr_file: 파일 프로파일 (컬럼·샘플·요약·도메인) — 관계 서비스의 table_name 은 file_name
- meta_datagokr_file_vec: 요약·키워드 임베딩 (KNN 은 summary_embedding 코사인)
"""

import hashlib
import json
from urllib.parse import urlparse

from psycopg import errors
from psycopg.types.json import Jsonb

from app.clients.postgres import connect, from_vector, to_vector
from app.config import settings
from app.exceptions.service import DatasetNotFoundError, DuplicateDatasetContentError
from app.schemas.dataset import DatasetProfile, DatasetSearchResult

_SOURCE_BY_HOST = {"data.go.kr": "DATA_GO_KR", "kosis.kr": "KOSIS", "opendata.hira.or.kr": "HIRA"}

_PROFILE_SELECT = """
SELECT d.dataset_id, f.file_name, f.display_name, d.external_ids, d.party, d.publisher, d.landing_page,
       d.source, d.source_type, d.language, f.domain, d.category, f.record_count, f.column_count, f.column_names,
       f.column_profile, d.description, f.table_summary, d.keywords, d.created_at, f.storage_key, {vectors}
FROM meta_datagokr_file f
JOIN dcat_dataset d USING (dataset_id)
LEFT JOIN meta_datagokr_file_vec v USING (file_id)
"""
_WITH_VECTORS = (
    "v.summary_embedding::text AS summary_embedding, v.keywords_embedding::text AS keywords_embedding, v.embed_text"
)
_NO_VECTORS = "NULL AS summary_embedding, NULL AS keywords_embedding, NULL AS embed_text"


def _source_code(profile: DatasetProfile) -> str:
    """원천 포털 코드 (dcat_dataset.source).

    ponytail: 제공처 URL 로 추정한다. 수집 경로가 출처 코드를 넘겨주면 그 값으로 대체
    """
    if profile.party == "1st":
        return "FIRST_PARTY"
    host = urlparse(profile.landing_page).hostname or ""
    return next((code for suffix, code in _SOURCE_BY_HOST.items() if host.endswith(suffix)), "OTHER")


def _content_sha256(profile: DatasetProfile) -> str:
    """파일 내용 해시. 레코드가 없으면(메타데이터-only) 데이터셋 이름과 요약문이 내용이다.

    메타데이터만 있는 데이터셋은 같은 설명을 공유할 수 있어(표준데이터의 지자체별 목록) 이름을 함께 넣는다.
    """
    content = (
        {"columns": profile.column_names, "sample_records": profile.sample_records, "record_count": profile.record_count}
        if profile.record_count
        else {"table_name": profile.table_name, "summary": profile.table_summary}
    )
    return hashlib.sha256(json.dumps(content, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def _column_profile(profile: DatasetProfile) -> dict:
    """meta_datagokr_file.column_profile — 정형은 샘플 레코드, 비정형 텍스트는 본문 특징을 함께 둔다."""
    return {"sample_records": profile.sample_records, **({"text_profile": profile.text_profile} if profile.text_profile else {})}


def _to_profile(row: dict) -> DatasetProfile:
    ids = row["external_ids"] or []
    return DatasetProfile(
        dataset_id=row["dataset_id"],
        table_name=row["file_name"],
        display_name=row["display_name"] or "",
        # 원천 식별자는 'datagokr:'·'kosis:' 처럼 출처 접두가 붙는다 — 접두 없는 값이 통합 카탈로그 식별자
        catalog_identifier=next((i for i in ids if ":" not in i), ""),
        party=row["party"] or "",
        provider=row["publisher"] or "",
        landing_page=row["landing_page"] or "",
        source=row["source"],
        source_type=row["source_type"],
        language=row["language"],
        domain=row["domain"] or "",
        categories=row["category"],
        record_count=row["record_count"] or 0,
        column_count=row["column_count"] or 0,
        column_names=row["column_names"],
        sample_records=row["column_profile"].get("sample_records", []),
        text_profile=row["column_profile"].get("text_profile", {}),
        description=row["description"] or "",
        table_summary=row["table_summary"] or "",
        embed_text=row["embed_text"] or "",
        table_embedding=from_vector(row["summary_embedding"]),
        keywords=row["keywords"],
        keywords_embedding=from_vector(row["keywords_embedding"]),
        created_at=row["created_at"].isoformat(),
        raw_storage_key=row["storage_key"] or "",
    )


class DatasetRepository:
    """데이터셋 원장 (dcat_dataset · meta_datagokr_file · meta_datagokr_file_vec)."""

    def upsert(self, profile: DatasetProfile) -> None:
        """데이터셋을 등록하거나 같은 dataset_id 를 대체 등록한다 (데이터셋 1건 = 파일 1건).

        Raises:
            DuplicateDatasetContentError: 같은 내용의 파일이 다른 이름으로 이미 있음
        """
        content_sha = _content_sha256(profile)
        file_id = f"file-{content_sha[:24]}"
        dataset = {
            "dataset_id": profile.dataset_id,
            "source": _source_code(profile),
            "title": profile.table_name,
            "description": profile.description,
            "publisher": profile.provider,
            "keywords": Jsonb(profile.keywords),
            "category": Jsonb(profile.categories),
            "landing_page": profile.landing_page,
            "external_ids": Jsonb([profile.catalog_identifier] if profile.catalog_identifier else []),
            "party": profile.party,
            "source_type": profile.source_type,
            "language": profile.language,
            "created_at": profile.created_at or None,
        }
        try:
            with connect() as conn:
                conn.execute(
                    """
                    INSERT INTO dcat_dataset (dataset_id, source, title, description, publisher, keywords, category,
                                              landing_page, external_ids, party, source_type, language, visibility,
                                              created_at)
                    VALUES (%(dataset_id)s, %(source)s, %(title)s, %(description)s, %(publisher)s, %(keywords)s,
                            %(category)s, %(landing_page)s, %(external_ids)s, %(party)s, %(source_type)s,
                            %(language)s, 'PUBLIC', COALESCE(%(created_at)s::timestamptz, now()))
                    ON CONFLICT (dataset_id) DO UPDATE SET
                        source = EXCLUDED.source, title = EXCLUDED.title, description = EXCLUDED.description,
                        publisher = EXCLUDED.publisher, keywords = EXCLUDED.keywords, category = EXCLUDED.category,
                        landing_page = EXCLUDED.landing_page, external_ids = EXCLUDED.external_ids,
                        party = EXCLUDED.party, source_type = EXCLUDED.source_type, language = EXCLUDED.language,
                        updated_at = now()
                    """,
                    dataset,
                )
                conn.execute("DELETE FROM meta_datagokr_file WHERE dataset_id = %s", (profile.dataset_id,))
                conn.execute(
                    """
                    INSERT INTO meta_datagokr_file (file_id, dataset_id, file_name, display_name, content_sha256,
                                                    storage_key, record_count, column_count, column_names,
                                                    column_profile, domain, table_summary)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        file_id, profile.dataset_id, profile.table_name, profile.display_name, content_sha,
                        profile.raw_storage_key, profile.record_count, profile.column_count,
                        Jsonb(profile.column_names),
                        # ponytail: 항목별 기술통계량(타입·결측률·상위값) 산출 전까지 샘플 레코드를 둔다
                        Jsonb(_column_profile(profile)),
                        profile.domain, profile.table_summary,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO meta_datagokr_file_vec (file_id, summary_embedding, keywords_embedding,
                                                        embed_text, model_name)
                    VALUES (%s, %s::vector, %s::vector, %s, %s)
                    """,
                    (
                        file_id, to_vector(profile.table_embedding), to_vector(profile.keywords_embedding),
                        profile.embed_text or profile.table_summary, settings.EMBEDDING_MODEL_NAME,
                    ),
                )
        except errors.UniqueViolation as e:
            raise DuplicateDatasetContentError(profile.table_name) from e

    def update_text_analysis(self, profile: DatasetProfile) -> None:
        """이미 등록된 데이터셋에 비정형 텍스트 분석 결과(형식·언어·본문 특징·분류·임베딩)만 덮어쓴다.

        upsert 는 파일 행을 지우고 다시 넣어 표준 항목(파일 형식·크기·기간)을 잃으므로 쓰지 않는다.

        Raises:
            DatasetNotFoundError: 파일 행이 없는 데이터셋
        """
        with connect() as conn:
            conn.execute(
                "UPDATE dcat_dataset SET source_type = %s, language = %s, category = %s, updated_at = now() "
                "WHERE dataset_id = %s",
                (profile.source_type, profile.language, Jsonb(profile.categories), profile.dataset_id),
            )
            row = conn.execute(
                "UPDATE meta_datagokr_file SET column_profile = column_profile || %s, domain = %s, updated_at = now() "
                "WHERE dataset_id = %s RETURNING file_id",
                (Jsonb({"text_profile": profile.text_profile}), profile.domain, profile.dataset_id),
            ).fetchone()
            if row is None:
                raise DatasetNotFoundError(profile.dataset_id)
            conn.execute(
                """
                INSERT INTO meta_datagokr_file_vec (file_id, summary_embedding, keywords_embedding, embed_text, model_name)
                VALUES (%s, %s::vector, %s::vector, %s, %s)
                ON CONFLICT (file_id) DO UPDATE SET
                    summary_embedding = EXCLUDED.summary_embedding, keywords_embedding = EXCLUDED.keywords_embedding,
                    embed_text = EXCLUDED.embed_text, model_name = EXCLUDED.model_name, embedded_at = now()
                """,
                (
                    row["file_id"], to_vector(profile.table_embedding), to_vector(profile.keywords_embedding),
                    profile.embed_text, settings.EMBEDDING_MODEL_NAME,
                ),
            )

    def search_knn(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        domain: str | None = None,
        party: str = "",
        provider: str = "",
        exclude_table_name: str = "",
    ) -> list[DatasetSearchResult]:
        """요약 임베딩 코사인 KNN (도메인/파티/제공기관 조건 필터). score 는 코사인 유사도."""
        with connect() as conn:
            rows = conn.execute(
                """
                SELECT f.file_name, f.display_name, f.domain, d.description, f.table_summary,
                       d.party, d.publisher, d.landing_page,
                       1 - (v.summary_embedding <=> %(q)s::vector) AS score
                FROM meta_datagokr_file f
                JOIN dcat_dataset d USING (dataset_id)
                JOIN meta_datagokr_file_vec v USING (file_id)
                WHERE v.summary_embedding IS NOT NULL
                  AND (%(domain)s::text IS NULL OR f.domain = %(domain)s)
                  AND (%(party)s = '' OR d.party = %(party)s)
                  AND (%(provider)s = '' OR d.publisher = %(provider)s)
                  AND f.file_name <> %(exclude)s
                ORDER BY v.summary_embedding <=> %(q)s::vector
                LIMIT %(k)s
                """,
                {
                    "q": to_vector(query_embedding), "domain": domain or None, "party": party,
                    "provider": provider, "exclude": exclude_table_name, "k": top_k,
                },
            ).fetchall()
        return [
            DatasetSearchResult(
                table_name=r["file_name"],
                display_name=r["display_name"] or "",
                domain=r["domain"] or "",
                description=r["description"] or "",
                table_summary=r["table_summary"] or "",
                score=r["score"],
                party=r["party"] or "",
                provider=r["publisher"] or "",
                landing_page=r["landing_page"] or "",
            )
            for r in rows
        ]

    def get(self, table_name: str) -> DatasetProfile | None:
        """table_name(file_name)으로 단건 조회 (임베딩 포함)."""
        with connect() as conn:
            row = conn.execute(
                _PROFILE_SELECT.format(vectors=_WITH_VECTORS) + " WHERE f.file_name = %s", (table_name,)
            ).fetchone()
        return _to_profile(row) if row else None

    def get_by_identifier(self, catalog_identifier: str) -> DatasetProfile | None:
        """통합 카탈로그 식별자(catalog_entry.identifier)로 단건 조회."""
        if not catalog_identifier:
            return None
        with connect() as conn:
            row = conn.execute(
                _PROFILE_SELECT.format(vectors=_WITH_VECTORS)
                + " WHERE d.external_ids ? %s ORDER BY d.updated_at DESC LIMIT 1",
                (catalog_identifier,),
            ).fetchone()
        return _to_profile(row) if row else None

    def list_domain_embeddings(self) -> list[tuple[str, str, list[float]]]:
        """도메인 분류 centroid 계산용 (table_name, domain, 요약 임베딩) 목록."""
        with connect() as conn:
            rows = conn.execute(
                """
                SELECT f.file_name, f.domain, v.summary_embedding::text AS embedding
                FROM meta_datagokr_file f JOIN meta_datagokr_file_vec v USING (file_id)
                WHERE v.summary_embedding IS NOT NULL
                """
            ).fetchall()
        return [(r["file_name"], r["domain"] or "", from_vector(r["embedding"])) for r in rows]

    def list_all(self) -> list[DatasetProfile]:
        """등록된 모든 데이터셋 (임베딩 제외, 최근 등록 순)."""
        with connect() as conn:
            rows = conn.execute(
                _PROFILE_SELECT.format(vectors=_NO_VECTORS) + " ORDER BY d.created_at DESC"
            ).fetchall()
        return [_to_profile(r) for r in rows]

    def search(
        self,
        query: str = "",
        query_embedding: list[float] | None = None,
        category: str = "",
        subcategory: str = "",
        party: str = "",
        provider: str = "",
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[int, list[dict]]:
        """분야·조건 필터 + 검색어(제목·요약·키워드 일치 또는 의미 유사) 목록.

        검색어가 있으면 텍스트 일치를 먼저, 그다음 의미 유사도 순. 없으면 최근 등록 순.

        Returns:
            (전체 건수, 행 목록) — 행은 카드 표시용 필드 + relation_count(확정 관계 수) + similarity
        """
        with connect() as conn:
            rows = conn.execute(
                """
                WITH base AS (
                    SELECT f.file_name, f.display_name, d.description, f.table_summary, d.publisher, d.party,
                           d.source, d.category, f.domain, f.record_count, f.column_count, d.landing_page,
                           d.keywords, d.created_at,
                           (SELECT count(*) FROM dcat_dataset_relation r
                            WHERE r.status = 'APPROVED'
                              AND d.dataset_id IN (r.source_dataset_id, r.target_dataset_id)) AS relation_count,
                           CASE WHEN %(qvec)s::vector IS NULL THEN NULL
                                ELSE 1 - (v.summary_embedding <=> %(qvec)s::vector) END AS similarity,
                           %(like)s::text IS NOT NULL AND (
                               f.file_name ILIKE %(like)s OR f.display_name ILIKE %(like)s
                               OR f.table_summary ILIKE %(like)s OR d.keywords::text ILIKE %(like)s
                           ) AS text_hit
                    FROM meta_datagokr_file f
                    JOIN dcat_dataset d USING (dataset_id)
                    LEFT JOIN meta_datagokr_file_vec v USING (file_id)
                    WHERE (%(category)s = '' OR EXISTS (
                              SELECT 1 FROM jsonb_array_elements_text(d.category) t
                              WHERE split_part(t, ' > ', 1) = %(category)s))
                      AND (%(subcategory)s = '' OR d.category ? %(subcategory)s)
                      AND (%(party)s = '' OR d.party = %(party)s)
                      AND (%(provider)s = '' OR d.publisher = %(provider)s)
                )
                SELECT *, count(*) OVER () AS total
                FROM base
                WHERE %(like)s::text IS NULL OR text_hit OR similarity >= %(min_similarity)s
                ORDER BY text_hit DESC, similarity DESC NULLS LAST, created_at DESC
                LIMIT %(limit)s OFFSET %(offset)s
                """,
                {
                    "qvec": to_vector(query_embedding or []), "like": f"%{query}%" if query else None,
                    "category": category, "subcategory": subcategory, "party": party, "provider": provider,
                    "min_similarity": settings.SEARCH_MIN_SIMILARITY, "limit": limit, "offset": offset,
                },
            ).fetchall()
        return (rows[0]["total"] if rows else 0), rows

    def category_counts(self) -> dict[str, int]:
        """분야별 데이터셋 수 — '대분류' 와 '대분류 > 중분류' 키를 함께 (다중 분류라 대분류는 중복 없이 센다)."""
        with connect() as conn:
            rows = conn.execute(
                """
                SELECT t AS category, count(DISTINCT d.dataset_id) AS n
                FROM dcat_dataset d, jsonb_array_elements_text(d.category) t GROUP BY t
                UNION ALL
                SELECT split_part(t, ' > ', 1), count(DISTINCT d.dataset_id)
                FROM dcat_dataset d, jsonb_array_elements_text(d.category) t GROUP BY 1
                """
            ).fetchall()
        return {r["category"]: r["n"] for r in rows}

    def delete(self, table_name: str) -> bool:
        """데이터셋 삭제 — 파일·임베딩·관계는 ON DELETE CASCADE 로 함께 정리된다."""
        with connect() as conn:
            cur = conn.execute(
                "DELETE FROM dcat_dataset WHERE dataset_id IN "
                "(SELECT dataset_id FROM meta_datagokr_file WHERE file_name = %s)",
                (table_name,),
            )
        return cur.rowcount > 0


def get_dataset_repository() -> DatasetRepository:
    return DatasetRepository()
