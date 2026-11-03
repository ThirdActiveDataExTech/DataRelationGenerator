"""비정형 텍스트(문서) 등록 검증 — 본문 발췌가 설명·분류·임베딩 입력이 되고, 형식·언어가 원장에 남는다."""

import io
import zipfile

import pytest

from app.src.dataset import dataset_service as module
from app.src.dataset.dataset_service import DatasetService


def _docx(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", f'<w:document xmlns:w="w"><w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>')
    return buf.getvalue()


class _Repo:
    def __init__(self):
        self.saved = []

    def get(self, table_name):
        return None

    def upsert(self, profile):
        self.saved.append(profile)


@pytest.fixture
def calls(monkeypatch):
    seen = {"embed": [], "llm": []}
    monkeypatch.setattr(module.embedding_service, "generate_embedding", lambda t: seen["embed"].append(t) or [0.1, 0.2])

    def generate(self, variables):
        seen["llm"].append((self.task, variables))
        if self.task == "document_metadata_extraction":
            return {"description": "시도별 어린이집 보육 실태를 조사한 보고서", "keywords": ["어린이집", "보육"]}
        return {"categories": ["교육 > 유아", "사회 > 복지"]}

    monkeypatch.setattr(module.LLMResponse, "generate_llm_response", generate)
    return seen


def test_uploaded_document_becomes_unstructured_text_dataset(calls):
    repo = _Repo()
    profile = DatasetService(dataset_repo=repo, relation_repo=object()).ingest_document(
        _docx("2021년 전국 보육실태조사 결과 어린이집 이용 현황"), "보육실태.docx", "보육실태조사 보고서", party="3rd",
    )

    assert repo.saved == [profile]
    assert profile.source_type == "unstructured_text" and profile.language == "ko"
    assert profile.text_profile["format"] == "DOCX" and "어린이집 이용 현황" in profile.text_profile["excerpt"]
    assert (profile.column_names, profile.record_count) == ([], 0)
    # 설명이 없으면 본문 발췌로 만들고, 임베딩·분류 입력에는 본문 발췌가 들어간다
    assert profile.table_summary == "시도별 어린이집 보육 실태를 조사한 보고서"
    assert "어린이집 이용 현황" in profile.embed_text and calls["embed"][0] == profile.embed_text
    classify = next(v for task, v in calls["llm"] if task == "dataset_category_classification")
    assert "본문 발췌: 2021년 전국 보육실태조사" in classify["summary"]
    assert profile.categories == ["교육 > 유아", "사회 > 복지"] and profile.domain == "교육"


def test_given_description_keywords_and_categories_skip_llm(calls):
    features = {"format": "PDF", "char_count": 42, "excerpt": "Number of hospital beds by region and type"}
    profile = DatasetService(dataset_repo=_Repo(), relation_repo=object()).document_profile(
        "Hospital bed report", features, "Annual report", ["hospital"], categories=["의료 > 공중보건"],
    )
    assert calls["llm"] == []  # 설명·키워드·분류가 모두 주어지면 LLM 을 부르지 않는다
    assert (profile.language, profile.categories, profile.table_summary) == ("en", ["의료 > 공중보건"], "Annual report")


def test_upload_endpoint_routes_documents_to_text_path(monkeypatch):
    from fastapi.testclient import TestClient

    from app.api.routers import dataset as router
    from app.config import settings
    from app.main import app
    from app.schemas.dataset import DatasetProfile

    seen = {}

    def ingest_document(content, filename, table_name, **kwargs):
        seen.update(filename=filename, table_name=table_name, size=len(content))
        return DatasetProfile(
            table_name=table_name, domain="", source_type="unstructured_text", language="ko", record_count=0,
            column_count=0, column_names=[], sample_records=[], text_profile={"format": "DOCX", "char_count": 9, "excerpt": "x"},
        )

    monkeypatch.setattr(router.dataset_service, "ingest_document", ingest_document)
    monkeypatch.setattr(router.dataset_service, "ingest_table", lambda **kw: pytest.fail("문서는 표 경로로 가면 안 된다"))
    monkeypatch.setattr(settings, "RAW_STORAGE_BACKEND", "none")
    response = TestClient(app).post(
        "/dataset/ingest", headers={"x-token": settings.X_TOKEN},
        files={"file": ("보육실태.docx", _docx("보육실태조사"), "application/octet-stream")}, data={"table_name": "보육실태조사 보고서"},
    )
    assert response.status_code == 200, response.text
    assert seen["filename"] == "보육실태.docx" and seen["table_name"] == "보육실태조사 보고서"
    assert response.json()["result"]["source_type"] == "unstructured_text"


def test_judge_sees_document_excerpt_instead_of_empty_samples():
    from app.schemas.dataset import DatasetProfile
    from app.src.dataset.relation_pipeline import _judge_samples

    base = dict(domain="", record_count=0, column_count=0, column_names=[])
    doc = DatasetProfile(table_name="보고서", sample_records=[], text_profile={"format": "PDF", "char_count": 9, "excerpt": "어린이집 이용 현황"}, **base)
    table = DatasetProfile(table_name="표", sample_records=[{"시설명": "가"}], **base)
    assert "어린이집 이용 현황" in _judge_samples(doc)
    assert _judge_samples(table) == '[{"시설명": "가"}]'
