"""원천 메타데이터 항목 실물 조사 — 공공데이터포털·KOSIS 가 실제로 주는 항목과 채움률.

표준 항목 정의(카탈로그·데이터셋·데이터파일)의 근거 자료를 만든다. 문서상 항목이 아니라
실제 응답에 오는 항목을 기준으로 삼기 위해 채널별 표본을 받아 항목별 채움률·예시값을 낸다.

채널
- 공공데이터포털 목록조회 API(15077093)·검색 API — 수집 파이프라인이 받아 둔 parquet
- 공공데이터포털 데이터셋별 DCAT XML·Schema.org JSON — 공개 주소, 키 불필요
- KOSIS 통계목록 · 통계표 메타(getMeta) · 통계설명자료 — KOSIS_API_KEY 필요

    KOSIS_API_KEY=... uv run --no-project --with truststore --with pandas --with pyarrow \\
        python scripts/metadata_standard/survey_sources.py <parquet 폴더> <출력 폴더>

출력: <출력 폴더>/raw/*.json (원본 응답), <출력 폴더>/fields.json (채널별 항목·채움률·예시)
"""

import json
import os
import random
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import pandas as pd
import truststore

truststore.inject_into_ssl()  # 사내망 인증서 검사 환경 — 시스템 신뢰 저장소 사용

DATAGOKR_SAMPLE = {"OPEN API": 30, "STD": 30, "FILE": 40}
KOSIS_TABLES_PER_TOPIC = 3
DELAY = 0.2
NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dct": "http://purl.org/dc/terms/",
    "dcat": "http://www.w3.org/ns/dcat#",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "vcard": "http://www.w3.org/2006/vcard/ns#",
}
PREFIX = {v: k for k, v in NS.items()}


def fetch(url: str) -> str:
    time.sleep(DELAY)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", errors="replace")


def qname(tag: str) -> str:
    uri, _, local = tag[1:].partition("}")
    return f"{PREFIX.get(uri, uri)}:{local}"


def summarize(records: list[dict], n: int | None = None) -> dict:
    """[{항목: 값}] → {항목: {fill, example}} (값이 비어 있지 않은 비율)"""
    n = n or len(records)
    filled, example = defaultdict(int), {}
    for rec in records:
        for key, value in rec.items():
            text = str(value).strip() if value is not None else ""
            if text and text.lower() not in {"nan", "none", "[]"}:
                filled[key] += 1
                example.setdefault(key, text[:120])
    return {k: {"fill": round(filled[k] / n, 3), "example": example[k]} for k in sorted(filled, key=lambda k: -filled[k])}


# ── 공공데이터포털 ─────────────────────────────────────

def dcat_fields(xml_text: str) -> dict:
    """DCAT XML 의 Dataset/DataService 하위 요소 → {경로: 값}"""
    root = ET.fromstring(xml_text)
    out = {}
    for node in root.iter():
        if qname(node.tag) not in {"dcat:Dataset", "dcat:DataService"}:
            continue
        kind = qname(node.tag)
        for child in node.iter():
            if child is node:
                continue
            lang = child.get("{http://www.w3.org/XML/1998/namespace}lang")
            key = f"{kind}/{qname(child.tag)}" + (f"@{lang}" if lang else "")
            value = (child.text or "").strip() or child.get(f"{{{NS['rdf']}}}resource", "")
            if value:
                out[key] = value
    return out


def schema_org_fields(doc: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in doc.items():
        if key.startswith("@"):
            continue
        if isinstance(value, dict):
            out.update(schema_org_fields(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def survey_datagokr(parquet_dir: Path, raw: Path) -> dict:
    listing = pd.read_parquet(parquet_dir / "standard_list.parquet")
    search = pd.read_parquet(parquet_dir / "standard_search.parquet")
    result = {
        "datagokr.list_api": {"n": len(listing), "fields": summarize(listing.to_dict("records"))},
        "datagokr.search_api": {"n": len(search), "fields": summarize(search.drop(columns=["columns"]).to_dict("records"))},
    }

    listing["kind"] = listing.apply(
        lambda r: "OPEN API" if r.list_type == "OPEN API" else ("STD" if r.ownership_grounds == "공공데이터 제공표준" else "FILE"),
        axis=1,
    )
    rng = random.Random(7)
    dcat_recs, schema_recs, raw_dump = defaultdict(list), defaultdict(list), {}
    for kind, size in DATAGOKR_SAMPLE.items():
        pool = listing[listing.kind == kind].list_id.astype(str).unique().tolist()
        for pk in rng.sample(pool, min(size, len(pool))):
            entry = {"kind": kind}
            try:
                entry["dcat"] = fetch(f"https://www.data.go.kr/dcat/metadata/{pk}")
                dcat_recs[kind].append(dcat_fields(entry["dcat"]))
            except Exception as e:  # noqa: BLE001 — 조사용: 실패도 기록
                entry["dcat_error"] = str(e)
            for category in (["openapi"] if kind == "OPEN API" else ["fileData", "standard"]):
                try:
                    doc = json.loads(fetch(f"https://www.data.go.kr/catalog/{pk}/{category}.json"))
                    entry["schema_org"] = doc
                    schema_recs[kind].append(schema_org_fields(doc))
                    break
                except Exception as e:  # noqa: BLE001
                    entry[f"schema_org_error_{category}"] = str(e)
            raw_dump[pk] = entry
    (raw / "datagokr_dcat_schemaorg.json").write_text(json.dumps(raw_dump, ensure_ascii=False, indent=1))
    for kind in DATAGOKR_SAMPLE:
        result[f"datagokr.dcat[{kind}]"] = {"n": len(dcat_recs[kind]), "fields": summarize(dcat_recs[kind])}
        result[f"datagokr.schema_org[{kind}]"] = {"n": len(schema_recs[kind]), "fields": summarize(schema_recs[kind])}
    return result


# ── KOSIS ────────────────────────────────────────────

def kosis(path: str, key: str, **params) -> list | dict:
    params.update(apiKey=key, format="json", jsonVD="Y")
    return json.loads(fetch(f"https://kosis.kr/openapi/{path}&{urllib.parse.urlencode(params)}"))


def kosis_tables(key: str) -> list[dict]:
    """주제별 목록(MT_ZTITLE) 대분류마다 앞쪽 하위 목록을 따라 내려가 통계표를 몇 개씩 뽑는다."""
    tables = []
    for topic in kosis("statisticsList.do?method=getList", key, vwCd="MT_ZTITLE", parentListId=""):
        found, queue = [], [topic["LIST_ID"]]
        while queue and len(found) < KOSIS_TABLES_PER_TOPIC:
            nodes = kosis("statisticsList.do?method=getList", key, vwCd="MT_ZTITLE", parentListId=queue.pop(0))
            if not isinstance(nodes, list):
                continue
            found += [n for n in nodes if n.get("TBL_ID")][: KOSIS_TABLES_PER_TOPIC - len(found)]
            queue += [n["LIST_ID"] for n in nodes if n.get("LIST_ID") and not n.get("TBL_ID")][:2]
        tables += [{**t, "TOPIC": topic.get("LIST_NM")} for t in found]
    return tables


def survey_kosis(key: str, raw: Path) -> dict:
    tables = kosis_tables(key)
    meta_types = ["TBL", "ORG", "PRD", "ITM", "CMMT", "UNIT", "SOURCE", "WGT", "NCD"]
    meta_recs, expl_recs, raw_dump, seen_stats = defaultdict(list), [], {}, set()
    for t in tables:
        entry = {"list": t}
        for mtype in meta_types:
            try:
                rows = kosis("statisticsData.do?method=getMeta", key, orgId=t["ORG_ID"], tblId=t["TBL_ID"], type=mtype)
            except Exception as e:  # noqa: BLE001
                rows = {"err": str(e)}
            entry[mtype] = rows if mtype not in {"NCD", "ITM"} else (rows[:5] if isinstance(rows, list) else rows)
            # 한 통계표 안의 여러 행은 항목 하나로 합친다 (있으면 채움)
            merged = {}
            for row in rows if isinstance(rows, list) else []:
                merged.update({k: v for k, v in row.items() if str(v).strip()})
            meta_recs[mtype].append(merged)
        stat_id = t.get("STAT_ID")
        if stat_id and stat_id not in seen_stats:
            seen_stats.add(stat_id)
            try:
                rows = kosis("statisticsExplData.do?method=getList", key, statId=stat_id, metaItm="All")
                entry["EXPL"] = rows
                merged = {}
                for row in rows if isinstance(rows, list) else []:
                    merged.update({k: v for k, v in row.items() if str(v).strip()})
                expl_recs.append(merged)
            except Exception as e:  # noqa: BLE001
                entry["EXPL"] = {"err": str(e)}
        raw_dump[t["TBL_ID"]] = entry
    (raw / "kosis_meta.json").write_text(json.dumps(raw_dump, ensure_ascii=False, indent=1))
    result = {"kosis.statistics_list": {"n": len(tables), "fields": summarize(tables)}}
    for mtype in meta_types:
        result[f"kosis.meta[{mtype}]"] = {"n": len(tables), "fields": summarize(meta_recs[mtype], len(tables))}
    result["kosis.stat_explanation"] = {"n": len(seen_stats), "fields": summarize(expl_recs, len(seen_stats))}
    return result


def main() -> None:
    parquet_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    result = survey_datagokr(parquet_dir, raw)
    result.update(survey_kosis(os.environ["KOSIS_API_KEY"], raw))
    (out / "fields.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    for channel, info in result.items():
        print(f"{channel}: n={info['n']}, 항목 {len(info['fields'])}개")


if __name__ == "__main__":
    main()
