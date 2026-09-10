"""카탈로그 access_url 실데이터 수집 (액티브 루프 확장)."""

import csv
import io
import json

import logging
import requests

from app.config import settings

logger = logging.getLogger(__name__)


class FetchedData:
    """access_url 수집 결과 — 파싱된 레코드와 원본 보관용 바이트."""

    def __init__(self, records: list[dict], content: bytes, filename: str):
        """Access url 수집 결과 초기화."""
        self.records = records
        self.content = content
        self.filename = filename


def fetch_and_store(access_url: str, table_name: str) -> tuple[list[dict], str]:
    """access_url 실데이터 수집 + 원본 보관(raw_storage).

    수집 비활성/실패 시 ([], "") — 호출부는 메타데이터-only 등록으로 폴백한다.

    Returns:
        (레코드 목록, raw_storage_key)
    """
    if not settings.CATALOG_FETCH_DATA_ENABLED:
        return [], ""

    fetched = fetch_records(access_url)
    if fetched is None:
        return [], ""

    from app.src.dataset.raw_storage import get_raw_storage

    raw_key = get_raw_storage().store_raw_dataset(
        table_name, fetched.filename, fetched.content
    ) or ""
    return fetched.records, raw_key


def fetch_records(access_url: str) -> FetchedData | None:
    """access_url 에서 실데이터를 받아 레코드 목록으로 변환.

    실패(네트워크 오류, HTML 응답, 미지원 포맷, 용량 초과)는 None 반환 —
    호출부는 메타데이터-only 등록으로 폴백한다.

    Args:
        access_url: 카탈로그 엔트리의 데이터 접근 URL (DCAT distribution accessURL)

    Returns:
        FetchedData 또는 None
    """
    if not access_url or not access_url.lower().startswith(("http://", "https://")):
        return None

    try:
        content, content_type = _download(access_url)
    except requests.RequestException as e:
        logger.warning(f"[DataFetch] 다운로드 실패 ({access_url}): {e}")
        return None
    except ValueError as e:
        logger.warning(f"[DataFetch] 스킵 ({access_url}): {e}")
        return None

    records = parse_content(content, content_type, access_url)
    if not records:
        return None

    if len(records) > settings.DATA_FETCH_MAX_RECORDS:
        logger.info(
            f"[DataFetch] 레코드 상한 적용: {len(records)} → {settings.DATA_FETCH_MAX_RECORDS} ({access_url})"
        )
        records = records[: settings.DATA_FETCH_MAX_RECORDS]

    filename = access_url.rstrip("/").rsplit("/", 1)[-1].split("?")[0] or "fetched"
    logger.info(f"[DataFetch] 수집 완료: {len(records)}건 ({access_url})")
    return FetchedData(records=records, content=content, filename=filename)


def _download(url: str) -> tuple[bytes, str]:
    """스트리밍 다운로드 — 용량 상한 초과 시 중단.

    Returns:
        (본문 바이트, Content-Type 헤더 소문자)

    Raises:
        requests.RequestException: 네트워크/HTTP 오류
        ValueError: 용량 초과
    """
    max_bytes = settings.DATA_FETCH_MAX_BYTES
    with requests.get(url, timeout=settings.DATA_FETCH_TIMEOUT_SECONDS, stream=True) as resp:
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "").lower()

        chunks, total = [], 0
        for chunk in resp.iter_content(chunk_size=1024 * 256):
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"용량 상한 초과 (> {max_bytes} bytes)")
            chunks.append(chunk)
        return b"".join(chunks), content_type


def parse_content(content: bytes, content_type: str, url: str = "") -> list[dict]:
    """응답 본문을 레코드 목록으로 파싱. 미지원 포맷은 빈 목록.

    판별 순서: Content-Type → URL 확장자 → 본문 첫 글자 스니핑.
    HTML(랜딩페이지)은 데이터가 아니므로 스킵한다.
    """
    if not content:
        return []
    if "text/html" in content_type:
        return []

    url_lower = url.lower().split("?")[0]

    if "csv" in content_type or url_lower.endswith(".csv"):
        return _parse_csv(content)
    if url_lower.endswith(".jsonl"):
        return _parse_jsonl(content)
    if "json" in content_type or url_lower.endswith(".json"):
        return _parse_json(content)

    # Content-Type 이 불명확한 경우 본문으로 스니핑
    head = content.lstrip()[:1]
    if head == b"<":  # HTML/XML 랜딩페이지
        return []
    if head in (b"{", b"["):
        return _parse_json(content)
    return _parse_csv(content)


def _decode(content: bytes) -> str:
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return content.decode("cp949")  # 공공데이터 CSV 에 흔한 인코딩


def _parse_csv(content: bytes) -> list[dict]:
    try:
        rows = list(csv.DictReader(io.StringIO(_decode(content))))
    except (UnicodeDecodeError, csv.Error) as e:
        logger.warning(f"[DataFetch] CSV 파싱 실패: {e}")
        return []
    # 헤더만 있거나 컬럼이 1개뿐인 비정형 텍스트는 데이터로 보지 않음
    if rows and len(rows[0]) > 1:
        return rows
    return []


def _parse_jsonl(content: bytes) -> list[dict]:
    try:
        return [
            json.loads(line)
            for line in _decode(content).splitlines()
            if line.strip()
        ]
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        logger.warning(f"[DataFetch] JSONL 파싱 실패: {e}")
        return []


def _parse_json(content: bytes) -> list[dict]:
    try:
        data = json.loads(_decode(content))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        logger.warning(f"[DataFetch] JSON 파싱 실패: {e}")
        return []

    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        # {"data": [...]} / {"records": [...]} / {"items": [...]} 등 래핑 응답 언랩
        for value in data.values():
            if isinstance(value, list) and value and all(isinstance(r, dict) for r in value):
                return value
        return [data]
    return []
