"""schema.org Movie JSON-LD(영문) → 데이터셋 등록 — 언어가 다른 데이터 간 연결 실증용.

수집 파이프라인이 저장소에 남긴 영화 페이지 JSON-LD 를 원천별로 한 데이터셋(행 = 영화 1편)으로 묶는다.
설명·키워드는 원문 언어(영어) 그대로 두어 다국어 임베딩이 국문 데이터와 직접 비교하게 한다.

    PYTHONPATH=. uv run python scripts/load_movie_jsonld.py movies_jsonld.json
    (파일 형식: {"rt": [JSON-LD, ...], "tmd": [JSON-LD, ...]} — 원천 코드별 목록)
"""

import argparse
import json
from pathlib import Path

from app.src.dataset.dataset_service import DatasetService

SOURCES = {
    "rt": {
        "table_name": "Rotten Tomatoes Movie Metadata",
        "provider": "Rotten Tomatoes",
        "landing_page": "https://www.rottentomatoes.com",
        "description": "Movie metadata collected from Rotten Tomatoes pages as schema.org JSON-LD: title, genre, "
                       "release date, director, cast, content rating and audience score of {n} films.",
        "keywords": ["movie", "film", "genre", "director", "cast", "audience score", "film review"],
    },
    "tmd": {
        "table_name": "TMDb Movie Metadata",
        "provider": "The Movie Database (TMDb)",
        "landing_page": "https://www.themoviedb.org",
        "description": "Movie metadata collected from The Movie Database (TMDb) pages as schema.org JSON-LD: title, "
                       "genre, release date, country of origin, runtime, user rating and plot synopsis of {n} films.",
        "keywords": ["movie", "film", "genre", "plot synopsis", "release date", "country of origin", "user rating"],
    },
}


def _names(value) -> str:
    """Person·Country 목록 → 이름 (앞 5개)."""
    items = value if isinstance(value, list) else [value] if value else []
    return ", ".join(i.get("name", "") for i in items[:5] if isinstance(i, dict))


def movie_record(ld: dict) -> dict:
    """Movie JSON-LD 1건 → 레코드 1행."""
    rating = ld.get("aggregateRating") or {}
    genre = ld.get("genre") or []
    released = (ld.get("releasedEvent") or [{}])[0].get("startDate", "")
    return {
        "title": ld.get("name", ""),
        "genre": ", ".join(genre) if isinstance(genre, list) else str(genre),
        "release_date": released or str(ld.get("dateCreated", ""))[:10],
        "director": _names(ld.get("director")),
        "cast": _names(ld.get("actor")),
        "country": _names(ld.get("countryOfOrigin")),
        "runtime": ld.get("duration", ""),
        "content_rating": ld.get("contentRating", ""),
        "rating": rating.get("ratingValue"),
        "rating_count": rating.get("ratingCount"),
        "synopsis": ld.get("description", ""),
        "url": ld.get("url", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="원천 코드별 JSON-LD 목록 파일")
    args = parser.parse_args()

    service = DatasetService()
    for code, items in json.loads(args.path.read_text(encoding="utf-8")).items():
        meta = SOURCES[code]
        records = [movie_record(ld) for ld in items if ld.get("@type") == "Movie"]  # 수집 요약 행 제외
        description = meta["description"].format(n=len(records))
        profile = service.ingest_table(
            records, meta["table_name"], description=description, table_summary=description,
            keywords=meta["keywords"], party="3rd", provider=meta["provider"], landing_page=meta["landing_page"],
        )
        print(f"{meta['table_name']}: {len(records)}편, 언어 {profile.language}, 분류 {profile.categories}")


if __name__ == "__main__":
    main()
