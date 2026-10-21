"""원천 메타데이터 → 표준 항목 변환

공공데이터포털 목록개방현황 1행, KOSIS 통계표 1개를 원장 4개 테이블의 행으로 바꾼다.
LLM·임베딩이 필요한 파생값(분야 분류·임베딩)은 여기서 만들지 않는다 — 적재기가 붙인다.
"""

import calendar
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date

PLACEHOLDER = {"", "-", "--", "없음", "해당없음", "해당 없음", "N/A", "null", "NULL", "."}

PERIODICITY = {
    "수시": "IRREGULAR", "부정기": "IRREGULAR",
    "연간": "ANNUAL", "1년": "ANNUAL", "년": "ANNUAL",
    "반기": "SEMIANNUAL", "분기": "QUARTERLY",
    "월간": "MONTHLY", "월": "MONTHLY", "주간": "WEEKLY",
    "일간": "DAILY", "일": "DAILY", "1회한": "ONCE",
}
RESOLUTION = {"년": "P1Y", "반기": "P6M", "분기": "P3M", "월": "P1M", "일": "P1D"}
SPATIAL_UNIT = (("읍면동", "EMD"), ("시군구", "SIGUNGU"), ("시도", "SIDO"), ("전국", "NATION"))
TABULAR = {"CSV", "XLSX", "XLS", "JSON", "XML"}
KOSIS_RIGHTS = "KOSIS 통계정보 이용지침 — 상업적 이용·재배포 가능, 통계수치 임의 변경 금지, 무료"


@dataclass
class StandardRecord:
    """원장 4개 테이블에 들어갈 표준 항목 (테이블 → 컬럼 → 값)."""

    dataset: dict
    attr: dict
    file: dict
    metrics: dict = field(default_factory=dict)  # metric_code → 관측값


def clean(value) -> str:
    """자리표시 값('-'·'해당없음' 등)과 결측을 빈 문자열로."""
    text = "" if value is None else str(value).strip()
    return "" if text in PLACEHOLDER or text.lower() == "nan" else text


def _last_day(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def _span(year: int, month: int | None = None, day: int | None = None) -> tuple[date, date]:
    """연·월·일 중 주어진 정밀도의 기간 첫날·마지막날."""
    if day:
        d = date(year, month, day)
        return d, d
    if month:
        return date(year, month, 1), _last_day(year, month)
    return date(year, 1, 1), date(year, 12, 31)


_DATE = re.compile(r"((?:19|20)\d{2})\s*(?:[년./-]\s*(\d{1,2}))?\s*(?:[월./-]\s*(\d{1,2}))?")


def parse_period(text: str) -> tuple[date | None, date | None]:
    """공공 시간범위 자유 텍스트 → (시작, 종료). 날짜 2개면 시작·종료, 1개면 그 기간, 없으면 (None, None).

    "2021년 1월 ~ 2025년 12월", "1952년~2025년", "~ 2025.03.11.", "2025년 7월 1일 ~ 8월 31일"
    """
    found = []
    for m in _DATE.finditer(clean(text)):
        year, month, day = (int(g) if g else None for g in m.groups())
        try:
            found.append(_span(year, month if month and 1 <= month <= 12 else None, day))
        except ValueError:
            continue
    if not found:
        return None, None
    open_start = clean(text).startswith("~")
    return (None if open_start else found[0][0]), found[-1][1]


def kosis_period(prd_se: str, value: str) -> tuple[date | None, date | None]:
    """KOSIS 수록시점 → (첫날, 마지막날). 'YYYY' · 'YYYY.MM' · 'YYYYMM' · 'YYYY n/4' · 'YYYY 1/2'."""
    value = clean(value)
    m = re.match(r"^((?:19|20)\d{2})\s*(\d)/([24])", value)
    if m:
        year, part, parts = int(m[1]), int(m[2]), int(m[3])
        months = 12 // parts
        return date(year, months * (part - 1) + 1, 1), _last_day(year, months * part)
    m = re.match(r"^((?:19|20)\d{2})\.?(\d{2})?\.?(\d{2})?$", value)
    if not m:
        return None, None
    year, month, day = (int(g) if g else None for g in m.groups())
    if prd_se == "일" and month and day:
        return _span(year, month, day)
    return _span(year, month)


def file_date_suffix(name: str) -> date | None:
    """파일데이터명 끝 날짜 접미(_YYYYMMDD · _YYYYMM · _YYYY) → 그 기간의 마지막날."""
    m = re.search(r"_((?:19|20)\d{2})(\d{2})?(\d{2})?$", clean(name))
    if not m:
        return None
    year, month, day = (int(g) if g else None for g in m.groups())
    try:
        return _span(year, month, day)[1]
    except ValueError:
        return None


def periodicity_code(text: str) -> str | None:
    """갱신 주기 원문 → 코드 (수시 IRREGULAR · 연간 ANNUAL …)."""
    text = re.sub(r"\s*\(.*\)$", "", clean(text))  # '수시 (1회성 데이터)'
    return PERIODICITY.get(text)


def license_code(text: str) -> str | None:
    """이용허락범위 원문 → 공공누리 유형 코드 (KOGL-n · 제한 없음 OPEN)."""
    text = clean(text)
    if not text:
        return None
    if "제한 없음" in text:
        return "OPEN"
    m = re.search(r"제\s*(\d)\s*유형", text)
    return f"KOGL-{m[1]}" if m else None


def spatial_unit(*texts: str) -> str | None:
    """공표범위·분류명에서 가장 세밀한 집계 단위 (EMD · SIGUNGU · SIDO · NATION)."""
    joined = " ".join(clean(t) for t in texts)
    return next((code for word, code in SPATIAL_UNIT if word in joined), None)


def split_theme(text: str) -> list[str]:
    """공공 분류체계 '대분류 - 소분류' → [대분류, 소분류]."""
    return [part.strip() for part in clean(text).split(" - ") if part.strip()]


def display_name(title: str, publisher: str) -> str:
    """'기관_목록명' 에서 기관 접두를 뗀 짧은 이름."""
    title = clean(title)
    prefix = f"{clean(publisher)}_"
    return title[len(prefix):] if publisher and title.startswith(prefix) else title


def yn(text: str) -> bool | None:
    """Y/N → boolean, 그 외 None."""
    return {"Y": True, "N": False}.get(clean(text).upper())


def _date(text: str) -> date | None:
    try:
        return date.fromisoformat(clean(text)[:10])
    except ValueError:
        return None


def structure_hash(payload: dict) -> str:
    """메타데이터만 있는 데이터파일의 내용 해시 (구조 JSON 기준)."""
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


# ── 공공데이터포털 ─────────────────────────────────────

LIST_TYPE = {"FILE": "FILE", "API": "API", "STD": "STD"}


def datagokr_record(row: dict) -> StandardRecord:
    """목록개방현황 1행 → 표준 항목. 키는 목록개방현황 컬럼명 그대로."""
    list_id = clean(row["목록키"])
    list_type = LIST_TYPE.get(clean(row["목록유형"]), clean(row["목록유형"]))
    title, publisher = clean(row["목록명"]), clean(row["제공기관"])
    start, end = parse_period(row.get("시간범위"))
    file_name = clean(row.get("파일데이터명")) if list_type == "FILE" else ""
    file_format = clean(row.get("확장자(데이터포맷)")).upper() or None
    record_count = clean(row.get("전체행"))
    rights = clean(row.get("이용허락범위"))
    return StandardRecord(
        dataset={
            "dataset_id": f"urn:datagokr:{list_id}",
            "source": "DATA_GO_KR",
            "title": title,
            "description": clean(row.get("설명")),
            "publisher": publisher,
            "publisher_code": clean(row.get("제공기관코드")) or None,
            "keywords": [k.strip() for k in clean(row.get("키워드")).split(",") if k.strip()],
            "theme": split_theme(row.get("분류체계")),
            "spatial": clean(row.get("공간범위")) or None,
            "temporal_start": start,
            "temporal_end": end,
            "temporal_resolution": None,
            "accrual_periodicity": periodicity_code(row.get("업데이트 주기")),
            "license": license_code(rights),
            "rights": rights or None,
            "landing_page": clean(row.get("목록 URL")),
            "external_ids": [f"datagokr:{list_id}"],
            "created_at": _date(row.get("등록일")),
            "updated_at": _date(row.get("수정일")),
            "item_count": 1,
        },
        attr={
            "list_type": list_type,
            "next_registration_date": _date(row.get("차기 등록 예정일")),
            "media_type": clean(row.get("매체유형")) or None,
            "provision_form": clean(row.get("제공형태")) or None,
            "api_type": clean(row.get("API 유형")) or None,
            "allowed_traffic": clean(row.get("신청가능 트래픽")) or None,
            "review_type": clean(row.get("심의 유형")) or None,
            "national_core_yn": yn(row.get("국가중점여부")),
            "standard_data_yn": yn(row.get("표준데이터여부")),
            "cost_charged": clean(row.get("비용부과유무")) or None,
            "cost_basis": clean(row.get("비용부과기준 및 단위")) or None,
            "dept_name": clean(row.get("관리 부서명")) or None,
            "dept_phone": clean(row.get("관리부서 전화번호")) or None,
            "retention_basis": clean(row.get("보유근거")) or None,
            "collection_method": clean(row.get("수집방법")) or None,
            "data_limitation": clean(row.get("데이터 한계")) or None,
            "other_notes": clean(row.get("기타 유의사항")) or None,
        },
        file={
            "file_name": file_name or title,
            "file_format": "API" if list_type == "API" else file_format,
            # 비표 형식(pdf·jpg 등)의 전체행 '1' 은 자리표시 값
            "record_count": int(record_count) if record_count.isdigit() and file_format in TABULAR else None,
            "coverage_start": start,
            "coverage_end": file_date_suffix(file_name) or end,
        },
        metrics={
            "download_request": clean(row.get("다운로드_활용신청건수")),
            "view_count": clean(row.get("조회수")),
        },
    )


# ── KOSIS ────────────────────────────────────────────

def _rows(value) -> list:
    """KOSIS 응답 목록. 오류·빈 응답(dict·None)은 빈 목록."""
    return value if isinstance(value, list) else []


def _merge(rows) -> dict:
    """KOSIS 응답(같은 키가 여러 행에 흩어진 목록) → 첫 값 우선 dict."""
    out: dict = {}
    for row in _rows(rows):
        for key, value in row.items():
            if clean(value) and key not in out:
                out[key] = clean(value)
    return out


def kosis_record(table: dict, path: list[str], meta: dict, expl: list | dict) -> StandardRecord:
    """KOSIS 통계표 1개 → 표준 항목.

    Args:
        table: 통계목록 노드 (TBL_ID · TBL_NM · ORG_ID · STAT_ID · SEND_DE)
        path: 주제별 목록 경로 (상위 → 하위 목록명)
        meta: getMeta 응답 {ORG, PRD, ITM, CMMT, SOURCE}
        expl: 통계설명자료 응답
    """
    org_id, tbl_id = table["ORG_ID"], table["TBL_ID"]
    org = _merge(meta.get("ORG"))
    source = _merge(meta.get("SOURCE"))
    ex = _merge(expl)
    prd_rows = _rows(meta.get("PRD"))
    # 수록주기가 여러 개면(월·년) 가장 촘촘한 것
    order = ["일", "월", "분기", "반기", "년", "부정기"]
    prd = min(prd_rows, key=lambda r: order.index(r["PRD_SE"]) if r.get("PRD_SE") in order else 99, default={})
    start = kosis_period(prd.get("PRD_SE", ""), prd.get("STRT_PRD_DE", ""))[0]
    end = kosis_period(prd.get("PRD_SE", ""), prd.get("END_PRD_DE", ""))[1]

    itm = _rows(meta.get("ITM"))
    classes = list(dict.fromkeys(r["OBJ_NM"] for r in itm if r.get("OBJ_ID") != "ITEM" and r.get("OBJ_NM")))
    items = list(dict.fromkeys(r["ITM_NM"] for r in itm if r.get("OBJ_ID") == "ITEM" and r.get("ITM_NM")))
    units = {r["ITM_NM"]: r["UNIT_NM"] for r in itm if r.get("OBJ_ID") == "ITEM" and r.get("UNIT_NM")}
    values = {c: [r["ITM_NM"] for r in itm if r.get("OBJ_NM") == c][:50] for c in classes}
    notes = [clean(r.get("CMMT_DC")) for r in _rows(meta.get("CMMT")) if isinstance(r, dict) and clean(r.get("CMMT_DC"))]

    stats_name = ex.get("statsNm", "")
    purpose = re.split(r"(?<=[.다])\s", re.sub(r"^ㅇ\s*", "", ex.get("writingPurps", "")))[0]
    description = " ".join(p for p in (f"「{stats_name}」" if stats_name else "", table["TBL_NM"] + ".", purpose) if p)
    retention = " ".join(p for p in (ex.get("basisLaw", ""), f"(승인 {ex['confmNo']})" if ex.get("confmNo") else "") if p)
    method = " · ".join(p for p in (ex.get("statsKind", ""), ex.get("dataCollectMth", "")) if p)
    confm = ex.get("confmDt", "")
    structure = {"classes": classes, "items": items, "prd": prd}
    return StandardRecord(
        dataset={
            "dataset_id": f"urn:kosis:{org_id}:{tbl_id}",
            "source": "KOSIS",
            "title": table["TBL_NM"],
            "description": description,
            "publisher": org.get("ORG_NM") or "",
            "publisher_code": org_id,
            "keywords": list(dict.fromkeys(classes + items))[:10],
            "theme": path,
            "spatial": ex.get("examinObjArea") or None,
            "temporal_start": start,
            "temporal_end": end,
            "temporal_resolution": RESOLUTION.get(prd.get("PRD_SE", "")),
            "accrual_periodicity": periodicity_code(ex.get("pubPeriod") or ex.get("statsPeriod", "")),
            "license": "KOSIS-TERMS",
            "rights": KOSIS_RIGHTS,
            "landing_page": f"https://kosis.kr/statHtml/statHtml.do?orgId={org_id}&tblId={tbl_id}",
            "external_ids": [f"kosis:{tbl_id}"] + ([f"kosis-stat:{table['STAT_ID']}"] if table.get("STAT_ID") else []),
            "created_at": date(int(confm[:4]), int(confm[4:6]), int(confm[6:8])) if re.match(r"^\d{8}$", confm) else None,
            "updated_at": _date(table.get("SEND_DE")),
            "item_count": 1,
        },
        attr={
            "list_type": "STAT",
            "stat_name": stats_name or None,
            "spatial_unit": spatial_unit(ex.get("pubExtent", ""), *classes),
            "dept_name": source.get("DEPT_NM") or None,
            "dept_phone": source.get("DEPT_PHONE") or None,
            "retention_basis": retention or None,
            "collection_method": method or None,
            "data_limitation": ex.get("dataUserNote") or None,
            "other_notes": "\n".join(notes) or None,
            "standard_data_yn": False,
            "cost_charged": "무료",
        },
        file={
            # 같은 기관·같은 이름의 통계표가 판을 달리해 여럿 있다(…_2018 · …_NEW) — 통계표 ID 로 구분
            "file_name": f"{org.get('ORG_NM', org_id)}_{table['TBL_NM']}_{tbl_id}",
            "file_format": "KOSIS",
            "record_count": None,
            "column_names": classes + items,
            "column_profile": {"classes": values, "units": units},
            "coverage_start": start,
            "coverage_end": end,
            "structure_sha256": structure_hash(structure),
        },
    )
