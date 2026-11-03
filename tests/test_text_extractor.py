"""비정형 텍스트 본문 추출 검증 — 형식 판별(내용 기준), 형식별 본문, 언어 판별."""

import io
import zipfile

import pytest

from app.exceptions.service import DocumentTextExtractionError
from app.src.dataset.text_extractor import (
    detect_format,
    detect_language,
    extract_text,
    hwp_records_text,
    text_profile,
)


def _zip(parts: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, body in parts.items():
            z.writestr(name, body)
    return buf.getvalue()


def _pdf(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 10 100 Td ({text}) Tj ET".encode()
    objects = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 200]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>",
        b"<</Length %d>>stream\n" % len(stream) + stream + b"\nendstream",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    out, offsets = io.BytesIO(), []
    out.write(b"%PDF-1.4\n")
    for i, body in enumerate(objects, 1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj" % i + body + b"endobj\n")
    xref = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1))
    out.write(b"".join(b"%010d 00000 n \n" % o for o in offsets))
    out.write(b"trailer<</Size %d/Root 1 0 R>>\nstartxref\n%d\n%%%%EOF" % (len(objects) + 1, xref))
    return out.getvalue()


HWPX = _zip({
    "mimetype": "application/hwp+zip",
    "Contents/section0.xml": '<hs:sec xmlns:hs="s" xmlns:hp="p"><hp:p><hp:run><hp:t>의료기관 병상 현황</hp:t></hp:run></hp:p>'
                             "<hp:p><hp:run><hp:t>시도별 &amp; 종별</hp:t></hp:run></hp:p></hs:sec>",
})
DOCX = _zip({
    "word/document.xml": '<w:document xmlns:w="w"><w:body><w:p><w:r><w:instrText>PAGE</w:instrText></w:r>'
                         "<w:r><w:t>Hospital bed</w:t></w:r><w:r><w:t>s by region</w:t></w:r></w:p></w:body></w:document>",
})
ODT = _zip({
    "content.xml": '<office:document-content xmlns:office="o" xmlns:text="t"><office:body><office:text>'
                   "<text:h>보육 실태</text:h><text:p>어린이집 <text:span>현황</text:span></text:p>"
                   "</office:text></office:body></office:document-content>",
})


def test_format_is_detected_from_content_not_extension():
    assert detect_format(HWPX, "hwp") == "HWPX"  # 확장자가 HWP 로 표기된 HWPX
    assert detect_format(DOCX, "") == "DOCX"
    assert detect_format(ODT, "") == "ODT"
    assert detect_format(_pdf("x"), "hwp") == "PDF"
    assert detect_format(b"\xd0\xcf\x11\xe0" + b"\0" * 8, "") == "HWP"
    assert detect_format(b"\r\n\r\n<script>x</script><div>y</div>", "pdf") == "HTML"  # PDF 링크가 게시판 화면
    assert detect_format("요양기관 현황".encode(), "hwp") == "TXT"


def test_zip_documents_keep_paragraphs_and_drop_field_codes():
    assert extract_text(HWPX) == ("HWPX", "의료기관 병상 현황\n시도별 & 종별")
    assert extract_text(DOCX) == ("DOCX", "Hospital beds by region")
    assert extract_text(ODT) == ("ODT", "보육 실태\n어린이집 현황")


def test_pdf_text():
    assert extract_text(_pdf("Hospital beds")) == ("PDF", "Hospital beds")


def test_html_drops_scripts_and_menus():
    page = ("<html><head><title>t</title></head><body><nav>메뉴</nav><script>var a=1</script><div>로그인</div>"
            "<ul><li><a href='/p1'>스토킹·교제폭력 방지 및 피해자 지원 정책 안내</a></li></ul>"
            "<p>시도별 의료기관 병상 정보를 매년 공개합니다.</p></body></html>")
    assert extract_text(page.encode()) == ("HTML", "시도별 의료기관 병상 정보를 매년 공개합니다.")


def test_txt_falls_back_to_cp949():
    assert extract_text("요양기관 현황".encode("cp949"), "txt") == ("TXT", "요양기관 현황")


def test_empty_or_unsupported_documents_raise():
    with pytest.raises(DocumentTextExtractionError):
        extract_text(b"<html><body><script>x</script></body></html>")
    with pytest.raises(DocumentTextExtractionError):
        extract_text(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR", "jpg")


def _hwp_record(tag: int, payload: bytes) -> bytes:
    return (tag | (len(payload) << 20)).to_bytes(4, "little") + payload


def _wchars(*codes: int) -> bytes:
    return b"".join(c.to_bytes(2, "little") for c in codes)


def test_hwp_records_keep_para_text_and_skip_controls():
    text = "병상".encode("utf-16-le") + _wchars(9, *[0] * 6, 9) + "정보".encode("utf-16-le")  # 탭(8칸 인라인 컨트롤)
    table = _wchars(11, *[0] * 6, 11)  # 표 같은 확장 컨트롤(8칸) — 글자로 나오면 안 된다
    raw = _hwp_record(66, b"\0" * 22) + _hwp_record(67, text + table + _wchars(13))
    assert hwp_records_text(raw) == "병상 정보\n"


def test_language_detection():
    assert detect_language("시도별 의료기관 병상 정보 (Hospital bed, ICU)") == "ko"
    assert detect_language("Movie metadata with genre, rating and plot synopsis") == "en"
    assert detect_language("Bảo vệ sức khỏe bản thân, khoa nội, khoa nhi") == "vi"
    assert detect_language("2024 1234") == "und"
    assert detect_language("https://www.liveinkorea.kr/web/lay1/bbs/S1T13C42/A/13/list.do") == "und"  # URL 만 있는 본문


def test_profile_keeps_excerpt_and_drops_control_chars():
    fmt, text = extract_text("병상\x00 정보\x07\n".encode() + "가".encode() * 2000, "txt")
    assert text.startswith("병상 정보\n")
    assert text_profile(fmt, text) == {"format": "TXT", "char_count": len(text), "excerpt": text[:1500]}
