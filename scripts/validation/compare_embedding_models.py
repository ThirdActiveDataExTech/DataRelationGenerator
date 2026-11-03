"""임베딩 모델 비교 — 원장 임베딩 입력(embed_text)을 모델마다 새로 계산해 같은 조건으로 잰다.

1. 관계 후보 탐색: 확정 관계(APPROVED)의 양끝이 서로를 원장 전체 중 몇 위로 찾는가 (Hits@1·@10, MRR)
2. 교차 언어 검색: 국문 데이터셋 제목을 영문으로 옮긴 질의(crosslingual_queries.tsv)로 원래 데이터셋 순위.
   같은 데이터셋의 국문 제목 질의를 대조군으로 함께 잰다
3. 같은 문서의 언어판 찾기: 원장 문서형 데이터셋의 본문 발췌만으로, 같은 문서의 다른 언어판이 문서 전체 중 몇 위인가
   (제목·설명은 등록 기관이 국문으로 붙여 언어판끼리 같으므로 빼고 본문만 비교한다)
4. 임계값 환산: 기존 모델의 임계값(0.4)과 같은 비율을 통과시키는 새 모델 값.
   - 관계 후보(데이터셋 쌍): 확정 관계 쌍의 통과율(재현율)을 같게 / 무작위 쌍의 통과율을 같게
   - 검색(질의–데이터셋): 정답 데이터셋의 통과율을 같게 / 무관 데이터셋의 통과율을 같게

    PYTHONPATH=. uv run python scripts/validation/compare_embedding_models.py [--out result.json]
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from app.clients.postgres import connect

MODELS = {"jhgan/ko-sroberta-multitask": "", "intfloat/multilingual-e5-base": "query: "}  # 모델 → 입력 접두
OLD_THRESHOLDS = (0.4,)  # 관계 후보(Tier 2)·검색 의미 유사도 하한 — 기존 모델 기준값
QUERIES = Path(__file__).with_name("crosslingual_queries.tsv")
_RANDOM_PAIRS = 50_000
# 원장에 함께 등록된 같은 문서의 언어판 (dataset_id 묶음)
LANGUAGE_VERSIONS = {
    "국제보건의료재단 내 건강 지킴이 8개 언어판": [
        f"urn:datagokr:{n}" for n in (15052205, 15052218, 15052212, 15052211, 15052206, 15052204, 15052209, 15052201)
    ],
}


def load_documents() -> tuple[list[str], list[str]]:
    """문서형 데이터셋의 (dataset_id 목록, 본문 발췌 목록)."""
    with connect() as conn:
        rows = conn.execute(
            """SELECT d.dataset_id, f.column_profile -> 'text_profile' ->> 'excerpt' AS excerpt
               FROM dcat_dataset d JOIN meta_datagokr_file f USING (dataset_id)
               WHERE d.source_type = 'unstructured_text' ORDER BY d.dataset_id"""
        ).fetchall()
    return [r["dataset_id"] for r in rows], [r["excerpt"] for r in rows]


def load_corpus() -> tuple[list[str], list[str], dict[str, str], list[tuple[str, str]]]:
    """(dataset_id 목록, 임베딩 입력 목록, dataset_id → 제목, 확정 관계 쌍)."""
    with connect() as conn:
        rows = conn.execute(
            """SELECT d.dataset_id, d.title, coalesce(v.embed_text, f.table_summary, d.title) AS text
               FROM dcat_dataset d JOIN meta_datagokr_file f USING (dataset_id)
               LEFT JOIN meta_datagokr_file_vec v USING (file_id) ORDER BY d.dataset_id"""
        ).fetchall()
        pairs = conn.execute(
            "SELECT source_dataset_id, target_dataset_id FROM dcat_dataset_relation WHERE status = 'APPROVED'"
        ).fetchall()
    return (
        [r["dataset_id"] for r in rows],
        [r["text"] for r in rows],
        {r["dataset_id"]: r["title"] for r in rows},
        [(p["source_dataset_id"], p["target_dataset_id"]) for p in pairs],
    )


def ranks(queries: np.ndarray, corpus: np.ndarray, targets: list[int], exclude: list[int | None]) -> np.ndarray:
    """질의마다 정답의 순위(1부터). exclude 는 질의 자신(관계 탐색) — 후보에서 뺀다."""
    sims = queries @ corpus.T
    for i, e in enumerate(exclude):
        if e is not None:
            sims[i, e] = -np.inf
    target_sims = sims[np.arange(len(targets)), targets]
    return (sims > target_sims[:, None]).sum(axis=1) + 1


def summarize(r: np.ndarray) -> dict:
    return {"n": int(len(r)), "hits@1": float((r <= 1).mean()), "hits@10": float((r <= 10).mean()),
            "mrr": float((1 / r).mean())}


def language_versions(encode, doc_ids: list[str], excerpts: list[str]) -> dict:
    """언어판마다 같은 묶음의 다른 언어판 순위 (본문 발췌끼리, 문서형 데이터셋 전체가 후보)."""
    corpus = encode(excerpts)
    index = {d: i for i, d in enumerate(doc_ids)}
    src, dst = [], []
    for members in LANGUAGE_VERSIONS.values():
        present = [index[m] for m in members if m in index]
        src += [a for a in present for b in present if a != b]
        dst += [b for a in present for b in present if a != b]
    return summarize(ranks(corpus[src], corpus, dst, src)) if src else {"n": 0}


def evaluate(name: str, prefix: str, ids, texts, titles, pairs, queries, documents) -> dict:
    model = SentenceTransformer(name)
    encode = lambda xs: model.encode([prefix + x for x in xs], batch_size=64, normalize_embeddings=True,  # noqa: E731
                                     show_progress_bar=False)
    corpus = encode(texts)
    index = {d: i for i, d in enumerate(ids)}

    both = [(index[a], index[b]) for a, b in pairs if a in index and b in index]
    src = [a for a, b in both] + [b for a, b in both]
    dst = [b for a, b in both] + [a for a, b in both]
    relation = ranks(corpus[src], corpus, dst, src)

    known = [(index[q["dataset_id"]], q["query_en"]) for q in queries if q["dataset_id"] in index]
    targets = [i for i, _ in known]
    english = ranks(encode([q for _, q in known]), corpus, targets, [None] * len(known))
    korean = ranks(encode([titles[ids[i]] for i in targets]), corpus, targets, [None] * len(known))

    rng = np.random.default_rng(0)
    a, b = rng.integers(0, len(ids), _RANDOM_PAIRS), rng.integers(0, len(ids), _RANDOM_PAIRS)
    keep = a != b
    random_sims = np.einsum("ij,ij->i", corpus[a[keep]], corpus[b[keep]])
    related_sims = np.einsum("ij,ij->i", corpus[[x for x, _ in both]], corpus[[y for _, y in both]])
    ko_queries = encode([titles[ids[i]] for i in targets])
    query_target_sims = np.einsum("ij,ij->i", ko_queries, corpus[targets])
    query_random_sims = (ko_queries @ corpus[rng.integers(0, len(ids), 500)].T).ravel()
    return {
        "model": name,
        "relation_candidates": summarize(relation),
        "crosslingual_en_query": summarize(english),
        "same_language_ko_query": summarize(korean),
        "language_versions": language_versions(encode, *documents),
        "random_pair_quantiles": {q: float(np.quantile(random_sims, q / 100)) for q in (50, 90, 95, 99)},
        "related_pair_quantiles": {q: float(np.quantile(related_sims, q / 100)) for q in (10, 50)},
        "_sims": {"pair": (related_sims, random_sims), "search": (query_target_sims, query_random_sims)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, help="결과 JSON 저장 경로")
    args = parser.parse_args()

    ids, texts, titles, pairs = load_corpus()
    with QUERIES.open(encoding="utf-8") as f:
        queries = list(csv.DictReader(f, delimiter="\t"))
    print(f"원장 {len(ids)}건, 확정 관계 {len(pairs)}쌍, 교차 언어 질의 {len(queries)}건")

    documents = load_documents()
    print(f"문서형 데이터셋 {len(documents[0])}건 (언어판 묶음 {len(LANGUAGE_VERSIONS)}개)")
    results = [evaluate(name, prefix, ids, texts, titles, pairs, queries, documents) for name, prefix in MODELS.items()]
    old, new = results
    for use, (old_hit, old_miss) in old["_sims"].items():
        new_hit, new_miss = new["_sims"][use]
        for t in OLD_THRESHOLDS:
            recall, passed = float((old_hit >= t).mean()), float((old_miss >= t).mean())
            new.setdefault("mapped_thresholds", {})[f"{use}@{t}"] = {
                "old_recall": recall, "same_recall_threshold": float(np.quantile(new_hit, 1 - recall)),
                "old_unrelated_pass": passed, "same_unrelated_pass_threshold": float(np.quantile(new_miss, 1 - passed)),
            }
    for r in results:
        r.pop("_sims")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if args.out:
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
