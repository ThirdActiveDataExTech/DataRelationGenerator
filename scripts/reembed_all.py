"""원장 임베딩 일괄 재계산 — 임베딩 모델(EMBEDDING_MODEL_NAME)을 바꾼 뒤 실행한다.

요약 임베딩은 저장된 임베딩 입력(embed_text), 키워드 임베딩은 키워드별 임베딩의 평균이다 (등록 경로와 같은 규칙).
이미 현재 모델로 계산된 행은 건너뛰므로(읽을 때와 쓸 때 모두) 중단 후 다시 실행하면 이어서 하고,
다른 적재 작업과 동시에 돌려도 그쪽이 먼저 쓴 결과를 덮지 않는다.

    PYTHONPATH=. uv run python scripts/reembed_all.py [--batch 256]
"""

import argparse

from app.clients.postgres import connect, to_vector
from app.config import settings
from app.src.embedding.embedding_service import embedding_service
from app.utils.vector import mean_vector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=int, default=256, help="한 번에 계산·저장할 파일 수")
    args = parser.parse_args()

    model = settings.EMBEDDING_MODEL_NAME
    with connect() as conn:
        rows = conn.execute(
            """SELECT f.file_id, coalesce(nullif(v.embed_text, ''), f.table_summary, d.title) AS text, d.keywords
               FROM meta_datagokr_file f JOIN dcat_dataset d USING (dataset_id)
               LEFT JOIN meta_datagokr_file_vec v USING (file_id)
               WHERE v.model_name IS DISTINCT FROM %s ORDER BY f.file_id""",
            (model,),
        ).fetchall()
    print(f"{model} 로 다시 계산할 파일 {len(rows)}건")

    for start in range(0, len(rows), args.batch):
        chunk = rows[start:start + args.batch]
        summaries = embedding_service.generate_embeddings([r["text"] for r in chunk])
        words = sorted({k for r in chunk for k in r["keywords"] if k.strip()})
        word_vec = dict(zip(words, embedding_service.generate_embeddings(words))) if words else {}
        params = []
        for r, summary in zip(chunk, summaries):
            kw = [word_vec[k] for k in r["keywords"] if k in word_vec]
            params.append((r["file_id"], to_vector(summary), to_vector(mean_vector(kw) if kw else []), r["text"], model))
        with connect() as conn, conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO meta_datagokr_file_vec (file_id, summary_embedding, keywords_embedding, embed_text, model_name)
                   VALUES (%s, %s::vector, %s::vector, %s, %s)
                   ON CONFLICT (file_id) DO UPDATE SET
                       summary_embedding = EXCLUDED.summary_embedding, keywords_embedding = EXCLUDED.keywords_embedding,
                       embed_text = EXCLUDED.embed_text, model_name = EXCLUDED.model_name, embedded_at = now()
                   WHERE meta_datagokr_file_vec.model_name IS DISTINCT FROM EXCLUDED.model_name""",
                params,
            )
        print(f"  {start + len(chunk)}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
