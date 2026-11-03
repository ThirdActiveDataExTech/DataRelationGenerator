"""공공데이터포털 목록개방현황(파일, 36항목) 전수 채움률 — 전체와 목록유형(FILE/API/STD)별.

원본: https://www.data.go.kr/data/15062804/fileData.do (월간 갱신). 분석에 쓴 판은 시드 데이터 레포
`source/datagokr_목록개방현황_20260930.csv.gz` 에 보관한다 (SEED_DIR 은 db/.env).

    uv run --no-project --with pandas python scripts/metadata_standard/listing_fill.py \
        $SEED_DIR/source/datagokr_목록개방현황_20260930.csv.gz docs/metadata_standard/fields.json

fields.json 의 "datagokr.listing_file" 채널을 갱신한다. '-'·'해당없음' 같은 자리표시 값은 빈 값으로 센다.
"""

import json
import sys
from pathlib import Path

import pandas as pd

PLACEHOLDER = {"-", "--", "없음", "해당없음", "해당 없음", "N/A", "null", "NULL", "."}
TYPES = ["FILE", "API", "STD"]


def fill(series: pd.Series) -> float:
    s = series.fillna("").str.strip()
    return round(float(((s != "") & ~s.isin(PLACEHOLDER)).mean()), 3) if len(s) else 0.0


def main() -> None:
    src, out = sys.argv[1], Path(sys.argv[2])
    df = pd.read_csv(src, dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    fields = {}
    for col in df.columns:
        values = df[col].fillna("").str.strip()
        top = values[values != ""].value_counts().head(3)
        fields[col] = {
            "fill": fill(df[col]),
            "by_type": {t: fill(df.loc[df["목록유형"] == t, col]) for t in TYPES},
            "example": str(top.index[0])[:120] if len(top) else "",
            "top": {str(k)[:40]: int(v) for k, v in top.items()},
        }
    result = json.loads(out.read_text()) if out.exists() else {}
    result["datagokr.listing_file"] = {
        "n": len(df),
        "by_type": df["목록유형"].value_counts().to_dict(),
        "source": Path(src).name,
        "fields": fields,
    }
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(len(df), result["datagokr.listing_file"]["by_type"])


if __name__ == "__main__":
    main()
