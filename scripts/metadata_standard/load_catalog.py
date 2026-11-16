"""표준 항목 적재기 — 공공데이터포털·KOSIS·심평원·보건의료 빅데이터 통합 플랫폼 메타데이터를 표준 항목으로 바꿔
원장(dcat_*)에 넣는다.

    # 공공데이터포털 — 목록개방현황(시드 데이터 레포 source/)에서 분류체계로 골라 적재
    PYTHONPATH=. uv run --with truststore python scripts/metadata_standard/load_catalog.py datagokr \\
        --listing $SEED_DIR/source/datagokr_목록개방현황_20260930.csv.gz [--themes ...] [--limit N] [--dry-run]

    # KOSIS — 주제별 목록에서 주제 아래 통계표를 내려가며 적재
    KOSIS_API_KEY=... PYTHONPATH=. uv run --with truststore python scripts/metadata_standard/load_catalog.py kosis \\
        [--topics 인구 보건 복지] [--limit N] [--dry-run]

    # 심평원 보건의료빅데이터개방시스템 — 공공데이터 목록·상세 화면 (공공데이터포털 교차 게시분은 식별자만 연결)
    PYTHONPATH=. uv run --with truststore python scripts/metadata_standard/load_catalog.py hira [--limit N] [--dry-run]

    # 보건의료 빅데이터 통합 플랫폼 — 기관별 데이터 카탈로그와 컬럼 정의 (비회원 공개분)
    PYTHONPATH=. uv run --with truststore python scripts/metadata_standard/load_catalog.py hcdl [--limit N] [--dry-run]

    # 공공 일반 파일(CSV)의 컬럼명 — 원본 파일 앞부분(16KB)만 받아 헤더를 읽는다
    PYTHONPATH=. uv run --with truststore python scripts/metadata_standard/load_catalog.py columns [--limit N]

    # 공공 문서형 파일(PDF·HWP·HWPX·DOCX·ODT·HTML·TXT) — 원문 본문으로 비정형 텍스트 특징·분류·임베딩을 붙인다
    PYTHONPATH=. uv run --with truststore python scripts/metadata_standard/load_catalog.py texts [--limit N] [--redownload]

새 데이터셋은 분야 분류(LLM 1회)·임베딩을 붙여 등록하고, 이미 있으면 표준 항목만 갱신한다(--refresh 면 분류·임베딩도).
같은 원천 데이터셋이 예전 식별자(UUID)로 있으면 식별자를 URN 으로 바꿔 이어받는다 — 파일·관계는 ON UPDATE CASCADE.
"""

import argparse
import csv
import html
import http.cookiejar
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd
from psycopg.types.json import Jsonb

sys.path.insert(0, str(Path(__file__).parent))
from survey_sources import fetch, kosis  # noqa: E402 — 같은 폴더의 조사 스크립트 (truststore 주입 포함)

from app.clients.postgres import connect  # noqa: E402
from app.exceptions.service import DocumentTextExtractionError  # noqa: E402
from app.repositories.dataset_repository import DatasetRepository  # noqa: E402
from app.schemas.dataset import DatasetProfile  # noqa: E402
from app.src.dataset.dataset_service import DatasetService  # noqa: E402
from app.src.dataset.domain_classifier import CategoryClassifier, major_of  # noqa: E402
from app.src.dataset.text_extractor import DOCUMENT_FORMATS, detect_format, extract_text, text_profile  # noqa: E402
from app.src.dataset.standard_items import (  # noqa: E402
    HIRA_DETAIL_URL,
    StandardRecord,
    datagokr_record,
    display_name,
    hcdl_record,
    hira_record,
    kosis_record,
)
from app.src.embedding.embedding_service import embedding_service  # noqa: E402
from app.utils.vector import mean_vector  # noqa: E402

# 1차 적재 범위 — 실증 관련 분야 (보건의료·보육·방송, KOSIS 인구·보건·복지)
DATAGOKR_THEMES = ("보건의료 - 보건의료", "보건의료 - 건강보험", "사회복지 - 보육·가족및여성", "과학기술 - 방송통신")
KOSIS_TOPICS = ("인구", "보건", "복지")
KOSIS_META = ("ORG", "PRD", "ITM", "CMMT", "SOURCE")
HIRA_LIST = "https://opendata.hira.or.kr/op/opc/selectOpenDataList.do?pageIndex={page}"
HIRA_PAGE_SIZE = 10
HCDL_API = "https://hcdl.mohw.go.kr/api"

DATASET_COLUMNS = (
    "source", "title", "description", "publisher", "publisher_code", "keywords", "theme", "spatial",
    "temporal_start", "temporal_end", "temporal_resolution", "accrual_periodicity", "license", "rights",
    "landing_page", "external_ids", "item_count",
)
JSON_COLUMNS = {"keywords", "theme", "external_ids", "column_names", "column_profile"}


def _value(column: str, value):
    return Jsonb(value) if column in JSON_COLUMNS else value


def _norm(text: str | None) -> str:
    return re.sub(r"[\s_()\[\]·,.\-]", "", text or "")


def adopt(conn, rec: StandardRecord, landing_patterns: list[str]) -> None:
    """같은 원천 데이터셋이 다른 식별자로 있으면 식별자를 이 레코드의 URN 으로 바꾼다.

    원문 링크만으로는 판단하지 않는다 — 예전 적재분 중 링크가 다른 데이터셋을 가리키는 것이 있어
    이름(목록명·파일데이터명)도 겹칠 때만 이어받는다.
    """
    urn = rec.dataset["dataset_id"]
    if conn.execute("SELECT 1 FROM dcat_dataset WHERE dataset_id = %s", (urn,)).fetchone():
        return
    where = " AND ".join("d.landing_page LIKE %s" for _ in landing_patterns)
    old = conn.execute(
        f"SELECT d.dataset_id, d.title, f.file_name FROM dcat_dataset d "
        f"LEFT JOIN meta_datagokr_file f USING (dataset_id) WHERE {where}",
        tuple(landing_patterns),
    ).fetchone()
    if not old:
        return
    d = rec.dataset
    new_names = {_norm(d["title"]), _norm(display_name(d["title"], d["publisher"])), _norm(rec.file["file_name"])}
    old_names = {_norm(old["title"]), _norm(old["file_name"])}
    if any(a and b and (a in b or b in a) for a in new_names for b in old_names):
        conn.execute("UPDATE dcat_dataset SET dataset_id = %s WHERE dataset_id = %s", (urn, old["dataset_id"]))
    else:
        print(f"  이어받지 않음 — 링크는 같지만 이름이 다름: {old['title']} ≠ {d['title']}")


def enrich(rec: StandardRecord) -> None:
    """분야 분류·임베딩을 붙여 데이터셋을 등록한다 (기존 관계 서비스 등록 경로와 같은 행을 만든다)."""
    d, f = rec.dataset, rec.file
    summary = d["description"] or d["title"]
    embedding = embedding_service.generate_embedding(summary)
    keywords = d["keywords"]
    columns = f.get("column_names") or []
    categories = CategoryClassifier().classify(d["title"], summary, keywords, columns, embedding)
    DatasetRepository().upsert(DatasetProfile(
        dataset_id=d["dataset_id"], table_name=f["file_name"], display_name=display_name(d["title"], d["publisher"]),
        party="3rd", provider=d["publisher"], landing_page=d["landing_page"], source=d["source"],
        domain=major_of(categories[0]) if categories else "", categories=categories,
        # record_count 는 표준 항목 갱신에서 넣는다 — 여기서 넣으면 메타데이터-only 해시 규칙이 깨진다
        record_count=0, column_count=len(columns), column_names=columns, sample_records=[],
        description=d["description"], table_summary=summary, table_embedding=embedding, keywords=keywords,
        keywords_embedding=mean_vector([embedding_service.generate_embedding(k) for k in keywords]) if keywords else [],
    ))


def write_standard(conn, rec: StandardRecord, observed_at: date) -> None:
    """표준 항목을 원장 4개 테이블에 쓴다 (등록된 데이터셋 기준 갱신)."""
    d, dataset_id = rec.dataset, rec.dataset["dataset_id"]
    # 통합 카탈로그 식별자(접두 없음)는 남기고 원천 식별자만 갈아 끼운다
    current = conn.execute("SELECT external_ids FROM dcat_dataset WHERE dataset_id = %s", (dataset_id,)).fetchone()
    d = {**d, "external_ids": [i for i in (current["external_ids"] if current else []) if ":" not in i] + d["external_ids"]}
    sets = [c for c in DATASET_COLUMNS if d.get(c) is not None]
    for c in ("created_at", "updated_at"):
        if d.get(c):
            sets.append(c)
    conn.execute(
        f"UPDATE dcat_dataset SET {', '.join(f'{c} = %({c})s' for c in sets)} WHERE dataset_id = %(dataset_id)s",
        {**{c: _value(c, d[c]) for c in sets}, "dataset_id": dataset_id},
    )
    attr = {k: v for k, v in rec.attr.items() if v is not None}
    cols = ", ".join(attr)
    conn.execute(
        f"""INSERT INTO dcat_dataset_attr (dataset_id, {cols}) VALUES (%(dataset_id)s, {', '.join(f'%({c})s' for c in attr)})
            ON CONFLICT (dataset_id) DO UPDATE SET {', '.join(f'{c} = EXCLUDED.{c}' for c in attr)}, updated_at = now()""",
        {**attr, "dataset_id": dataset_id},
    )
    f = {k: v for k, v in rec.file.items() if v is not None and k not in {"file_name", "structure_sha256"}}
    if f.get("column_names"):
        f["column_count"] = len(f["column_names"])
    if "column_profile" in f:  # 기존 기술통계(샘플 등)는 두고 KOSIS 구조를 합친다
        f["column_profile"] = Jsonb(f["column_profile"])
        profile_set = "column_profile = column_profile || %(column_profile)s"
        f_sets = [profile_set] + [f"{c} = %({c})s" for c in f if c != "column_profile"]
    else:
        f_sets = [f"{c} = %({c})s" for c in f]
    if f_sets:
        conn.execute(
            f"UPDATE meta_datagokr_file SET {', '.join(f_sets)}, updated_at = now() WHERE dataset_id = %(dataset_id)s",
            {**{c: (_value(c, v) if c != "column_profile" else v) for c, v in f.items()}, "dataset_id": dataset_id},
        )
    for code, value in rec.metrics.items():
        if str(value).isdigit():
            conn.execute(
                """INSERT INTO active_meta_dataset (dataset_id, metric_code, metric_value, observed_at, source)
                   VALUES (%s, %s, %s, %s, 'PORTAL_CATALOG') ON CONFLICT DO NOTHING""",
                (dataset_id, code, int(value), observed_at),
            )


def load(records: list[tuple[StandardRecord, list[str]]], observed_at: date, refresh: bool, workers: int) -> None:
    """레코드 목록을 원장에 넣는다. (레코드, 예전 식별자를 찾을 landing_page 패턴)"""

    def one(item: tuple[StandardRecord, list[str]]) -> str:
        try:
            return _one(item)
        except Exception as e:  # noqa: BLE001 — 한 건 실패로 전체를 멈추지 않는다
            print(f"  실패 {item[0].dataset['dataset_id']} {item[0].dataset['title'][:30]}: {type(e).__name__} {e}"[:200])
            return "실패"

    def _one(item: tuple[StandardRecord, list[str]]) -> str:
        rec, patterns = item
        with connect() as conn:
            adopt(conn, rec, patterns)
            row = conn.execute(
                """SELECT d.category <> '[]' AND v.summary_embedding IS NOT NULL AS ready
                   FROM dcat_dataset d LEFT JOIN meta_datagokr_file f USING (dataset_id)
                   LEFT JOIN meta_datagokr_file_vec v USING (file_id) WHERE d.dataset_id = %s""",
                (rec.dataset["dataset_id"],),
            ).fetchone()
        status = "갱신"
        if refresh or not row or not row["ready"]:
            enrich(rec)
            status = "등록" if not row else "보강"
        with connect() as conn:
            write_standard(conn, rec, observed_at)
        return status

    done = Counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, status in enumerate(pool.map(one, records), 1):
            done[status] += 1
            if i % 100 == 0 or i == len(records):
                print(f"  {i}/{len(records)} {dict(done)}", flush=True)


def fill_report(records: list[StandardRecord]) -> None:
    """표준 항목별 채움률 — 정의서 필수·권장 항목 확인용."""
    n = len(records)
    for table in ("dataset", "attr", "file"):
        counts = Counter(k for r in records for k, v in getattr(r, table).items() if v not in (None, "", []))
        keys = list(dict.fromkeys(k for r in records for k in getattr(r, table)))
        print(f"[{table}] " + " · ".join(f"{k} {counts[k] / n:.2f}" for k in keys))


# ── 소스별 ───────────────────────────────────────────

def datagokr(args) -> list[tuple[StandardRecord, list[str]]]:
    df = pd.read_csv(args.listing, dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    # 이미 원장에 있는 공공데이터포털 데이터셋은 분야와 상관없이 표준 항목을 받게 한다
    with connect() as conn:
        existing = {
            r["pk"] for r in conn.execute(
                "SELECT substring(landing_page from '/data/([0-9]+)/') AS pk FROM dcat_dataset "
                "WHERE landing_page LIKE '%%data.go.kr/data/%%'"
            ).fetchall()
        }
    df = df[df["분류체계"].isin(args.themes) | df["목록키"].isin(existing)]
    if args.limit:
        df = df.head(args.limit)
    print(f"공공데이터포털 {len(df)}건 — {df['목록유형'].value_counts().to_dict()}")
    return [(datagokr_record(row), [f"%/data/{row['목록키']}/%"]) for row in df.to_dict("records")]


def kosis_collect(args) -> list[tuple[StandardRecord, list[str]]]:
    key = os.environ["KOSIS_API_KEY"]
    roots = kosis("statisticsList.do?method=getList", key, vwCd="MT_ZTITLE", parentListId="")
    tables = []
    for root in (r for r in roots if r.get("LIST_NM") in args.topics):
        queue, found = [(root["LIST_ID"], [root["LIST_NM"]])], 0
        while queue and (not args.limit or found < args.limit):
            parent, path = queue.pop(0)
            nodes = kosis("statisticsList.do?method=getList", key, vwCd="MT_ZTITLE", parentListId=parent)
            for node in nodes if isinstance(nodes, list) else []:
                if node.get("TBL_ID") and (not args.limit or found < args.limit):
                    tables.append((node, path))
                    found += 1
                elif node.get("LIST_ID") and not node.get("TBL_ID"):
                    queue.append((node["LIST_ID"], path + [node["LIST_NM"]]))
    print(f"KOSIS 통계표 {len(tables)}개 — 주제 {list(args.topics)}")
    expl_cache: dict[str, list | dict] = {}
    out = []
    for i, (table, path) in enumerate(tables, 1):
        meta = {t: kosis("statisticsData.do?method=getMeta", key, orgId=table["ORG_ID"], tblId=table["TBL_ID"], type=t)
                for t in KOSIS_META}
        stat_id = table.get("STAT_ID", "")
        if stat_id and stat_id not in expl_cache:
            expl_cache[stat_id] = kosis("statisticsExplData.do?method=getList", key, statId=stat_id, metaItm="All")
        rec = kosis_record(table, path, meta, expl_cache.get(stat_id, []))
        out.append((rec, [f"%orgId={table['ORG_ID']}%", f"%tblId={table['TBL_ID']}%"]))
        if i % 50 == 0:
            print(f"  메타 {i}/{len(tables)}", flush=True)
    return out


def _cells(markup: str) -> dict:
    """상세 화면 표의 항목명(th) → 값(td) 텍스트."""
    def text(fragment: str) -> str:
        return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", fragment))).strip()

    # 주석 처리된 칸(심평원 담당자명)이 다음 칸과 한 항목으로 읽히지 않게 주석부터 뗀다
    markup = re.sub(r"<!--.*?-->", "", markup, flags=re.S)
    return {text(k): text(v) for k, v in re.findall(r"<th[^>]*>(.*?)</th>\s*<td[^>]*>(.*?)</td>", markup, re.S)}


def hira_collect(args) -> list[tuple[StandardRecord, list[str]]]:
    first = fetch(HIRA_LIST.format(page=1))
    if not (found := re.search(r"전체\s*<span>(\d+)</span>건", first)):
        raise SystemExit("심평원 공공데이터 목록에서 전체 건수를 찾지 못했습니다 — 목록 화면 구성 변경 확인")
    total = int(found[1])
    pages = [first] + [fetch(HIRA_LIST.format(page=p)) for p in range(2, -(-total // HIRA_PAGE_SIZE) + 1)]
    snos = list(dict.fromkeys(s for page in pages for s in re.findall(r'<a href="(\d+)" class="select">', page)))
    snos = snos[: args.limit] if args.limit else snos
    print(f"심평원 공공데이터 {len(snos)}건 (목록 {total}건)")
    records = [hira_record({**_cells(fetch(HIRA_DETAIL_URL.format(sno=s))), "sno": s}) for s in snos]
    return [(r, [f"%opendata.hira.or.kr%sno={r.dataset['dataset_id'].rsplit(':', 1)[1]}"]) for r in link_cross_posted(records, args.dry_run)]


def link_cross_posted(records: list[StandardRecord], dry_run: bool) -> list[StandardRecord]:
    """공공데이터포털에 같은 데이터로 이미 적재된 것은 새로 만들지 않고 원천 식별자만 붙인다. 나머지를 돌려준다.

    심평원은 공공데이터 대부분을 공공데이터포털에도 올리고 상세에 그 주소를 적는다 — 같은 데이터가 두 행이 되면
    관계 생성이 둘을 서로의 판본으로 잇는다.
    """
    counterpart = {
        r.dataset["dataset_id"]: next((f"urn:{i}" for i in r.dataset["external_ids"] if i.startswith("datagokr:")), None)
        for r in records
    }
    with connect() as conn:
        loaded = {
            row["dataset_id"] for row in conn.execute(
                "SELECT dataset_id FROM dcat_dataset WHERE dataset_id = ANY(%s)", ([c for c in counterpart.values() if c],)
            ).fetchall()
        }
        linked = [r for r in records if counterpart[r.dataset["dataset_id"]] in loaded]
        for r in [] if dry_run else linked:
            own_id = r.dataset["external_ids"][0]
            conn.execute(
                "UPDATE dcat_dataset SET external_ids = external_ids || %s, updated_at = now() "
                "WHERE dataset_id = %s AND NOT external_ids ? %s",
                (Jsonb([own_id]), counterpart[r.dataset["dataset_id"]], own_id),
            )
    print(f"  공공데이터포털 적재분과 같은 데이터 {len(linked)}건 — 식별자 연결{'(시험)' if dry_run else ''}, 신규 {len(records) - len(linked)}건")
    return [r for r in records if r not in linked]


def hcdl_collect(args) -> list[tuple[StandardRecord, list[str]]]:
    def api(path: str, **query) -> list[dict]:
        params = {"q": json.dumps(query, separators=(",", ":")), "page": 1, "size": 10000, "total": 0}
        return json.loads(fetch(f"{HCDL_API}/{path}?{urllib.parse.urlencode(params)}"))

    insts = json.loads(fetch(f"{HCDL_API}/catalog/countByInst"))
    items = [i for inst in insts for i in api("catalog/list", prvdInstSpcd=inst["instSpcd"]) if i.get("dltnYn") != "Y"]
    items = items[: args.limit] if args.limit else items
    print(f"보건의료 빅데이터 통합 플랫폼 {len(items)}건 (기관 {len(insts)}곳 등록 {sum(i['cnt'] for i in insts)}건 중 비회원 공개분)")
    out = []
    for item in items:
        columns = api("column/list", clcd=item["clcd"], prvdInstSpcd=item["prvdInstSpcd"], nmbrPublYn=item["nmbrPublYn"])
        out.append((hcdl_record(item, columns), [f"%/{item['clcd']}/{item['prvdInstSpcd']}"]))
    return out


def columns(args) -> None:
    """공공 일반 파일(CSV)의 헤더 — 원본 앞부분 16KB 만 받아 컬럼명을 채운다."""
    with connect() as conn:
        rows = conn.execute(
            """SELECT f.dataset_id, d.landing_page FROM meta_datagokr_file f JOIN dcat_dataset d USING (dataset_id)
               WHERE d.source = 'DATA_GO_KR' AND f.file_format = 'CSV' AND f.column_names = '[]'
               ORDER BY d.dataset_id LIMIT %s""",
            (args.limit or 1_000_000,),
        ).fetchall()
    print(f"헤더를 읽을 CSV {len(rows)}건")

    def one(row) -> bool:
        try:
            page = fetch(row["landing_page"])
            url = re.search(r'"contentUrl"\s*:\s*"([^"]+)"', page)
            if not url:
                return False
            req = urllib.request.Request(url[1], headers={"User-Agent": "Mozilla/5.0", "Range": "bytes=0-16383"})
            with urllib.request.urlopen(req, timeout=30) as r:
                head = r.read(16384)
        except Exception:  # noqa: BLE001 — 개별 파일 실패는 건너뛴다
            return False
        # 헤더는 첫 줄만 필요하다 — 16KB 경계에서 잘린 글자와 무관하게 첫 줄만 디코딩 (줄바꿈은 \r\n · \n · \r)
        first = re.split(rb"\r\n|\n|\r", head, maxsplit=1)[0]
        text = next((t for t in (_decode(first, enc) for enc in ("utf-8-sig", "cp949")) if t is not None), "")
        try:
            header = [h.strip() for h in next(csv.reader(io.StringIO(text)), []) if h.strip()]
        except csv.Error:
            return False
        if not header:
            return False
        with connect() as conn:
            conn.execute(
                "UPDATE meta_datagokr_file SET column_names = %s, column_count = %s, updated_at = now() WHERE dataset_id = %s",
                (Jsonb(header), len(header), row["dataset_id"]),
            )
        return True

    with ThreadPoolExecutor(max_workers=4) as pool:
        ok = sum(pool.map(one, rows))
    print(f"컬럼명 채움 {ok}/{len(rows)}")


# ── 공공 문서형 파일(비정형 텍스트) 본문 ──────────────────────

PORTAL = "https://www.data.go.kr"
_MAX_DOCUMENT_BYTES = 50 * 1024 * 1024
_ATTACHMENT_TRIES = 3
_TEMPLATE_MIN_PAGES = 3  # 게시판 화면 본문에서 이만큼 여러 화면에 똑같이 나오는 줄은 누리집 공통 틀로 본다
_DOWN_ARGS = re.compile(r"fn_fileDataDown\(([^)]*)\)")
_ANCHOR = re.compile(r'<a\b([^>]*)>(.*?)</a>', re.I | re.S)
_HREF = re.compile(r'href="([^"#][^"]*)"', re.I)
_DOC_EXT = re.compile(r"\.(?:pdf|hwpx?|docx|odt|txt)\b", re.I)
_DOWN = re.compile(r"down|atchfile", re.I)
_TAG = re.compile(r"<[^>]+>")


class DownloadLimited(Exception):
    """포털 다운로드 한도 — 자동등록 방지 문자를 요구한다. 우회하지 않고 수집을 멈춘다."""


class _Session:
    """쿠키를 유지하는 포털 세션 (다운로드 경로가 같은 세션의 조회 결과를 요구한다)."""

    def __init__(self):
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))

    def get(self, url: str, data: dict | None = None, ajax: bool = False) -> bytes:
        headers = {"User-Agent": "Mozilla/5.0", **({"X-Requested-With": "XMLHttpRequest"} if ajax else {})}
        body = urllib.parse.urlencode(data).encode() if data else None
        with self.opener.open(urllib.request.Request(url, data=body, headers=headers), timeout=60) as r:
            content = r.read(_MAX_DOCUMENT_BYTES + 1)
        if len(content) > _MAX_DOCUMENT_BYTES:
            raise ValueError(f"파일이 {_MAX_DOCUMENT_BYTES // 2**20}MB 를 넘음")
        return content


def download_document(session: _Session, landing_page: str) -> bytes:
    """원문 파일을 받는다.

    - 포털 등록 파일: 화면과 같은 순서(파일 조회 → 다운로드 한도 확인 → 다운로드)
    - 제공기관 링크: 링크 화면이 문서면 그대로, 게시판 화면이면 첨부 문서 링크를 앞에서부터 시도하고
      없으면 화면 글자(게시글)를 쓴다
    """
    page = session.get(landing_page).decode("utf-8", "replace")
    if m := _DOWN_ARGS.search(page):
        pk, detail_pk, atch, sn = (a.strip(" '\"") for a in m[1].split(",")[:4])
        info = json.loads(session.get(f"{PORTAL}/tcs/dss/selectFileDataDownload.do", {
            "publicDataPk": pk, "publicDataDetailPk": detail_pk, "atchFileId": atch, "fileDetailSn": sn,
            "publicDataTyCode": "PR0051",
        }, ajax=True))
        if not info.get("status"):
            raise ValueError(f"포털 파일 조회 실패: {info.get('error')}")
        file_key = {"atchFileId": info["atchFileId"], "fileDetailSn": info["fileDetailSn"]}
        if json.loads(session.get(f"{PORTAL}/cmm/cmm/check-limit.json", file_key, ajax=True)).get("needCaptcha"):
            raise DownloadLimited()
        return session.get(f"{PORTAL}/cmm/cmm/fileDownload.do?{urllib.parse.urlencode(file_key)}")

    pk = re.search(r"/data/(\d+)/", landing_page)
    if not pk:
        raise ValueError(f"공공데이터포털 목록 주소가 아님: {landing_page}")
    pk = pk[1]
    link = json.loads(session.get(f"{PORTAL}/tcs/dss/selectLinkUrl.do?publicDataPk={pk}", ajax=True))
    if not link.get("status"):
        raise ValueError(f"제공기관 링크 없음: {link.get('errorDc')}")
    data = session.get(link["linkUrl"])
    if detect_format(data) != "HTML":
        return data
    for href in _attachment_links(data.decode("utf-8", "replace"))[:_ATTACHMENT_TRIES]:
        try:
            candidate = session.get(urllib.parse.urljoin(link["linkUrl"], html.unescape(href)))
        except (urllib.error.URLError, TimeoutError, ValueError):
            continue
        if detect_format(candidate) not in {"HTML", "TXT", ""}:
            return candidate
    return data


def _attachment_links(markup: str) -> list[str]:
    """게시판 화면의 첨부 문서 링크 — 문서 확장자가 보이는 링크(파일명이 제목·글자에 있는 다운로드 버튼)를 먼저.

    ponytail: href 가 있는 링크만. 자바스크립트 함수로 받는 첨부(onclick)는 기관마다 달라 화면 글자로 대신한다
    """
    ranked = []
    for attrs, label in _ANCHOR.findall(markup):
        href = _HREF.search(attrs)
        if not href or href[1].startswith("javascript"):
            continue
        visible = f"{attrs} {_TAG.sub('', label)}"
        if score := 2 * bool(_DOC_EXT.search(visible)) + bool(_DOWN.search(href[1])):
            ranked.append((-score, href[1]))
    return [h for _, h in sorted(ranked, key=lambda x: x[0])]


def texts(args) -> None:
    """공공 문서형 파일 본문 → 비정형 텍스트 특징(발췌·언어)·분류·다국어 임베딩.

    본문 특징은 원장(column_profile.text_profile)에 남기므로, 다시 실행하면 원문을 받지 않고 분류·임베딩만
    새로 한다(--redownload 면 원문부터). LLM 을 쓸 수 없으면 분류는 기존 값을 두고 본문·임베딩만 반영한다.
    ponytail: datagokr --refresh 는 파일 행을 새로 써서 본문 특징이 지워진다 — 그 뒤에는 이 명령을 --redownload 로 다시 실행
    """
    with connect() as conn:
        rows = conn.execute(
            """SELECT d.dataset_id, d.title, d.description, d.keywords, d.category, d.landing_page, f.file_format,
                      f.column_profile -> 'text_profile' AS features
               FROM meta_datagokr_file f JOIN dcat_dataset d USING (dataset_id)
               WHERE d.source = 'DATA_GO_KR' AND upper(f.file_format) = ANY(%s)
                 AND (%s::text IS NULL OR f.column_profile -> 'text_profile' ->> 'format' = %s)
               ORDER BY d.dataset_id LIMIT %s""",
            (sorted(DOCUMENT_FORMATS), args.only_format, args.only_format, args.limit or 1_000_000),
        ).fetchall()
    llm_ready = _llm_ready()
    print(f"문서형 파일 {len(rows)}건 (LLM {'사용' if llm_ready else '불가 — 분류는 기존 값 유지'})")

    session, service, repo, done = _Session(), DatasetService(), DatasetRepository(), Counter()
    for i, row in enumerate(rows, 1):
        features = None if args.redownload else row["features"]
        try:
            if not features:
                time.sleep(args.interval)  # 포털·제공기관 서버 부하를 줄인다
                features = text_profile(*extract_text(download_document(session, row["landing_page"]), row["file_format"]))
            profile = service.document_profile(
                row["title"], features, row["description"] or "", row["keywords"],
                categories=None if llm_ready else row["category"],
            )
        except DownloadLimited:
            print("  포털 다운로드 한도 — 자동등록 방지 문자를 요구해 멈춥니다. 다시 실행하면 이어서 받습니다.")
            break
        except (DocumentTextExtractionError, urllib.error.URLError, TimeoutError, ValueError, TypeError) as e:
            done["실패"] += 1
            reason = e.result["reason"] if isinstance(e, DocumentTextExtractionError) else f"{type(e).__name__} {e}"
            print(f"  실패 {row['dataset_id']} {row['title'][:30]}: {reason}"[:200])
            continue
        profile.dataset_id = row["dataset_id"]
        repo.update_text_analysis(profile)
        done[features["format"]] += 1
        if i % 20 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)} {dict(done)}", flush=True)
    print(f"완료 {dict(done)}, 화면 공통 문구를 지운 게시판 본문 {_strip_site_template(service, repo, llm_ready)}건")


def _strip_site_template(service: DatasetService, repo: DatasetRepository, llm_ready: bool) -> int:
    """게시판 화면(HTML) 본문에서 여러 화면에 똑같이 나오는 줄(누리집 공통 안내·주소·저작권)을 지우고 다시 임베딩한다.

    ponytail: 저장된 발췌(앞 1,500자) 기준 — 틀 줄을 지운 만큼 본문 뒷부분을 되살리지는 않는다
    """
    with connect() as conn:
        rows = conn.execute(
            """SELECT d.dataset_id, d.title, d.description, d.keywords, d.category, f.column_profile -> 'text_profile' AS features
               FROM meta_datagokr_file f JOIN dcat_dataset d USING (dataset_id)
               WHERE f.column_profile -> 'text_profile' ->> 'format' = 'HTML'"""
        ).fetchall()
    pages = Counter(line for r in rows for line in set(r["features"]["excerpt"].splitlines()))
    template = {line for line, n in pages.items() if n >= _TEMPLATE_MIN_PAGES}
    changed = 0
    for r in rows:
        lines = r["features"]["excerpt"].splitlines()
        kept = [line for line in lines if line not in template]  # 전부 틀이면 본문 없음(빈 발췌)
        if len(kept) == len(lines):
            continue
        profile = service.document_profile(
            r["title"], {**r["features"], "excerpt": "\n".join(kept)}, r["description"] or "", r["keywords"],
            categories=None if llm_ready else r["category"],
        )
        profile.dataset_id = r["dataset_id"]
        repo.update_text_analysis(profile)
        changed += 1
    return changed


def _llm_ready() -> bool:
    from openai import APIConnectionError

    from app.dependencies import chatopenai_client

    try:
        chatopenai_client.invoke("1")
        return True
    except APIConnectionError:
        return False


def _decode(data: bytes, encoding: str) -> str | None:
    try:
        return data.decode(encoding)
    except UnicodeDecodeError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="source", required=True)
    p1 = sub.add_parser("datagokr")
    p1.add_argument("--listing", required=True, help="목록개방현황 CSV(.gz)")
    p1.add_argument("--themes", nargs="+", default=list(DATAGOKR_THEMES))
    p2 = sub.add_parser("kosis")
    p2.add_argument("--topics", nargs="+", default=list(KOSIS_TOPICS))
    p3 = sub.add_parser("columns")
    p4 = sub.add_parser("texts")
    p4.add_argument("--redownload", action="store_true", help="본문 특징이 있어도 원문을 다시 받는다")
    p4.add_argument("--interval", type=float, default=1.0, help="원문 요청 사이 대기(초)")
    p4.add_argument("--only-format", help="이미 읽은 본문 형식이 이것인 데이터셋만 (예: 추출 규칙을 바꾼 HTML 만 다시 받기)")
    p5 = sub.add_parser("hira")
    p6 = sub.add_parser("hcdl")
    for p in (p1, p2, p3, p4, p5, p6):
        p.add_argument("--limit", type=int, default=0, help="소스(KOSIS 는 주제)별 최대 건수 — 시험 적재용")
    for p in (p1, p2, p5, p6):
        p.add_argument("--dry-run", action="store_true", help="변환·채움률만 보고 원장에 쓰지 않는다")
        p.add_argument("--refresh", action="store_true", help="이미 있는 데이터셋도 분류·임베딩을 다시 붙인다")
        p.add_argument("--workers", type=int, default=6)
        p.add_argument("--observed-at", type=date.fromisoformat, default=date.today(),
                       help="이력 지표 관측일 — 목록개방현황 판 날짜를 넣는다")
    args = parser.parse_args()

    if args.source in ("columns", "texts"):
        (columns if args.source == "columns" else texts)(args)
        return
    collect = {"datagokr": datagokr, "kosis": kosis_collect, "hira": hira_collect, "hcdl": hcdl_collect}
    records = collect[args.source](args)
    fill_report([r for r, _ in records])
    if not args.dry_run:
        load(records, args.observed_at, args.refresh, args.workers)


if __name__ == "__main__":
    main()
