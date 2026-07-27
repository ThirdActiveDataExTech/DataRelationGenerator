"""① 컬럼 간 연관성 분석 서비스(column-to-column-correlation-analysis) 클라이언트.

Tier 1.5: 이름이 달라 Tier1 규칙 매칭이 놓치는 컬럼 쌍(예: hr ↔ heart_rate)을
①의 검증된 스키마 매칭 모델(Transformer 임베딩 + XGBoost, REMA)로 탐지한다.

교환 포맷: 서비스가 파일 경로/S3 URI 입력만 받으므로, 프로파일의 샘플 레코드를
CSV로 MinIO에 올리고 서비스의 /correlations/remote 가 S3 URI로 읽게 한다.
"""

import csv
import io
import logging
from typing import Any

import requests

from app.config import settings
from app.schemas.dataset import ColumnMatchInfo, DatasetProfile

logger = logging.getLogger(__name__)


def _strip_table_prefix(column: str) -> str:
    """서비스 응답의 'Table1.컬럼명' 프리픽스 제거."""
    return column.split(".", 1)[1] if "." in column else column


def parse_match_columns(matches: list[dict[str, Any]]) -> list[ColumnMatchInfo]:
    """서비스 matches 응답 → ColumnMatchInfo 목록 (ai_correlation)."""
    return [
        ColumnMatchInfo(
            column_a=_strip_table_prefix(m["source_column"]),
            column_b=_strip_table_prefix(m["target_column"]),
            match_type="ai_correlation",
            correlation=m.get("correlation_coefficient"),
        )
        for m in matches
    ]


def sample_csv_bytes(profile: DatasetProfile) -> bytes:
    """프로파일의 컬럼명 + 샘플 레코드를 CSV 바이트로 직렬화."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=profile.column_names, extrasaction="ignore")
    writer.writeheader()
    for record in profile.sample_records:
        writer.writerow({k: record.get(k, "") for k in profile.column_names})
    return buf.getvalue().encode("utf-8")


def upload_sample_csv(profile: DatasetProfile) -> str | None:
    """샘플 CSV를 MinIO에 올리고 S3 URI 반환. MinIO 비활성/실패 시 None.

    ponytail: 매 실행마다 타임스탬프 키로 새로 업로드 (수 KB짜리 샘플이라 정리 불필요 수준).
    누적이 문제되면 고정 키 overwrite 로 교체.
    """
    if not settings.MINIO_ENABLED or not profile.column_names:
        return None
    from app.clients.minio_client import get_minio_storage

    key = get_minio_storage().store_raw_dataset(
        f"tier15-{profile.table_name}", "sample.csv", sample_csv_bytes(profile)
    )
    return f"s3://{settings.MINIO_BUCKET}/{key}" if key else None


class CorrelationClient:
    """column-to-column 서비스 REST 클라이언트 (x-token 헤더 인증)."""

    def __init__(self, base_url: str | None = None, x_token: str | None = None):
        self.base_url = (base_url or settings.CORRELATION_SERVICE_URL).rstrip("/")
        self.headers = {"x-token": x_token or settings.CORRELATION_SERVICE_X_TOKEN}

    def match_columns(self, l_s3_uri: str, r_s3_uri: str) -> list[ColumnMatchInfo]:
        """두 샘플 CSV(S3 URI)의 컬럼 매칭 실행 (서비스 기본 임계값 0.90 사용)."""
        resp = requests.post(
            f"{self.base_url}/correlations/remote",
            headers=self.headers,
            data={
                "l_s3_uri": l_s3_uri,
                "r_s3_uri": r_s3_uri,
                "endpoint_url": settings.MINIO_ENDPOINT,
                "access_key": settings.MINIO_ACCESS_KEY,
                "secret_key": settings.MINIO_SECRET_KEY,
                "region_name": settings.MINIO_REGION,
            },
            timeout=120,
        )
        resp.raise_for_status()
        result = resp.json().get("result") or {}
        return parse_match_columns(result.get("matches") or [])


def get_correlation_client() -> CorrelationClient:
    return CorrelationClient()
