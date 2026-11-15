"""비정형 텍스트 데이터(문서 파일) 본문 추출 및 형식·언어 판별.

형식은 확장자가 아니라 파일 앞부분(매직 바이트)과 압축 내부 구성으로 정한다 — 공공데이터의
확장자 표기는 실제 파일과 다를 때가 있다(HWP 로 표기된 HWPX, PDF 링크가 HTML 게시판인 경우 등).
"""

import html
import io
import re
import zipfile
import zlib
from html.parser import HTMLParser

import olefile
import py3langid
from pypdf import PdfReader
from pypdf.errors import DependencyError, PyPdfError

from app.exceptions.service import DocumentTextExtractionError

# 비정형 텍스트로 다루는 형식 (원장 file_format 값은 대문자)
DOCUMENT_FORMATS = frozenset({"PDF", "HWP", "HWPX", "DOCX", "ODT", "HTML", "TXT"})

_MAX_PDF_PAGES = 30  # 앞부분만으로 분류·임베딩에 충분하다. ponytail: 긴 보고서 뒷부분은 버린다
_MAX_CHARS = 20_000
_EXCERPT_CHARS = 1_500
_HTML_MIN_LINE = 20  # 이보다 짧은 화면 줄은 메뉴·버튼으로 본다
_LANGUAGE_SAMPLE_CHARS = 2_000
_MIN_LANGUAGE_LETTERS = 5  # 이보다 글자가 적으면 언어를 정하지 않는다 (URL·숫자만 있는 본문)
_URL = re.compile(r"https?://\S+|www\.\S+")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")  # 줄바꿈·탭 외 제어 문자 (원장 JSON 에 못 넣는 NUL 포함)

# 압축(zip) 문서 형식별 본문 파트와 문단 끝 태그
_ZIP_PARTS = {
    "HWPX": (re.compile(r"Contents/section\d+\.xml"), re.compile(r"</hp:p>")),
    "DOCX": (re.compile(r"word/document\.xml"), re.compile(r"</w:p>")),
    "ODT": (re.compile(r"content\.xml"), re.compile(r"</text:(?:p|h)>")),
}
_FIELD_CODE = re.compile(r"<w:instrText[^>]*>.*?</w:instrText>", re.S)  # DOCX 필드 명령(PAGE·TOC 등)
_TAG = re.compile(r"<[^>]+>")

_HWPTAG_PARA_TEXT = 67
_HWP_CHAR_CONTROLS = {0, 10, 13, *range(24, 32)}  # 1칸짜리 문자 컨트롤. 나머지 0~31 은 8칸짜리


def detect_format(data: bytes, hint: str = "") -> str:
    """파일 내용으로 형식을 판별한다. 이진 파일인데 문서 형식이 아니면 확장자 힌트(대문자)를 그대로 쓴다."""
    if data.startswith(b"%PDF"):
        return "PDF"
    if data.startswith(b"\xd0\xcf\x11\xe0"):  # OLE 복합 문서 — 한글 5.x
        return "HWP"
    if data.startswith(b"PK"):
        try:
            names = zipfile.ZipFile(io.BytesIO(data)).namelist()
        except zipfile.BadZipFile:
            return hint.upper()
        return next(
            (fmt for fmt, (part, _) in _ZIP_PARTS.items() if any(part.fullmatch(n) for n in names)), hint.upper()
        )
    if b"\x00" in data[:1024]:  # 텍스트가 아닌 이진 파일 (이미지 등) — 확장자 힌트로만 판단
        return hint.upper()
    # 글자 파일은 확장자와 무관하게 내용으로 — PDF 링크가 앞에 빈 줄이 붙은 게시판 화면인 경우가 있다
    return "HTML" if data[:2048].lstrip().startswith(b"<") else "TXT"


def extract_text(data: bytes, hint: str = "") -> tuple[str, str]:
    """문서 본문을 읽는다.

    Returns:
        (형식, 본문) — 본문은 공백을 정리하고 앞 20,000자까지

    Raises:
        DocumentTextExtractionError: 손상·암호화·미지원 형식이거나 본문이 비어 있음
    """
    fmt = detect_format(data, hint)
    try:
        if fmt == "PDF":
            text = _pdf_text(data)
        elif fmt == "HWP":
            text = _hwp_text(data)
        elif fmt in _ZIP_PARTS:
            text = _zip_xml_text(data, fmt)
        elif fmt == "HTML":
            text = _html_text(_decode(data))
        elif fmt == "TXT":
            text = _decode(data)
        else:
            raise DocumentTextExtractionError(f"미지원 형식: {fmt}")
    except (PyPdfError, DependencyError, OSError, zipfile.BadZipFile, zlib.error, UnicodeDecodeError) as e:
        raise DocumentTextExtractionError(f"{fmt} 본문 읽기 실패: {e}") from e
    text = _normalize(text)[:_MAX_CHARS]
    if not text:
        raise DocumentTextExtractionError(f"{fmt} 본문이 비어 있음 (이미지·암호화 문서 등)")
    return fmt, text


def detect_language(text: str) -> str:
    """언어 코드(ISO 639-1 — ko·en·vi·km 등 97개 언어 식별 모델). URL 을 빼고 글자가 거의 없으면 und."""
    sample = _URL.sub(" ", text[:_LANGUAGE_SAMPLE_CHARS])
    return py3langid.classify(sample)[0] if sum(c.isalpha() for c in sample) >= _MIN_LANGUAGE_LETTERS else "und"


def text_profile(fmt: str, text: str) -> dict:
    """원장 column_profile.text_profile 에 둘 본문 특징 (형식·글자 수·발췌). 언어는 데이터셋 항목(language)에 둔다."""
    return {"format": fmt, "char_count": len(text), "excerpt": text[:_EXCERPT_CHARS]}


def _normalize(text: str) -> str:
    text = _CONTROL_CHARS.sub("", text)
    lines = (re.sub(r"[ \t 　]+", " ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _pdf_text(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        reader.decrypt("")  # 열람 암호 없는 사용 제한 PDF 는 빈 암호로 풀린다
    pages = []
    for page in reader.pages[:_MAX_PDF_PAGES]:
        try:
            pages.append(page.extract_text() or "")
        except PyPdfError:  # 해석 한도를 넘는 쪽(도면·그림 위주)은 건너뛰고 나머지 쪽을 쓴다
            continue
    return "\n".join(pages)


def _zip_xml_text(data: bytes, fmt: str) -> str:
    part, para_end = _ZIP_PARTS[fmt]
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = sorted(
            (n for n in z.namelist() if part.fullmatch(n)), key=lambda n: int(re.sub(r"\D", "", n) or 0)
        )
        xml = "\n".join(z.read(n).decode("utf-8") for n in names)
    xml = para_end.sub("\n", _FIELD_CODE.sub("", xml))
    return html.unescape(_TAG.sub("", xml))


def _hwp_text(data: bytes) -> str:
    """한글 5.x(OLE) 본문 — BodyText/Section* 스트림의 문단 텍스트 레코드를 읽는다."""
    with olefile.OleFileIO(io.BytesIO(data)) as ole:
        flags = int.from_bytes(ole.openstream("FileHeader").read()[36:40], "little")
        if flags & 0b10:  # 암호 설정 문서
            raise DocumentTextExtractionError("암호가 설정된 HWP")
        sections = sorted(
            (e for e in ole.listdir() if len(e) == 2 and e[0] == "BodyText" and e[1].startswith("Section")),
            key=lambda e: int(e[1][len("Section"):]),
        )
        if not sections:  # 배포용 문서는 본문이 ViewText 에 암호화돼 있다
            raise DocumentTextExtractionError("본문 스트림 없음 (배포용 문서)")
        texts = []
        for entry in sections:
            raw = ole.openstream(entry).read()
            texts.append(hwp_records_text(zlib.decompress(raw, -15) if flags & 1 else raw))
    return "\n".join(texts)


def hwp_records_text(raw: bytes) -> str:
    """HWP 본문 레코드 열에서 문단 텍스트(HWPTAG_PARA_TEXT)만 이어 붙인다."""
    paragraphs, i = [], 0
    while i + 4 <= len(raw):
        header = int.from_bytes(raw[i:i + 4], "little")
        tag, size = header & 0x3FF, header >> 20
        i += 4
        if size == 0xFFF:  # 4095 이상은 길이를 다음 4바이트에 둔다
            size = int.from_bytes(raw[i:i + 4], "little")
            i += 4
        if tag == _HWPTAG_PARA_TEXT:
            paragraphs.append(_hwp_para_text(raw[i:i + size]))
        i += size
    return "\n".join(paragraphs)


def _hwp_para_text(buf: bytes) -> str:
    chars, j, n = [], 0, len(buf) // 2
    while j < n:
        code = int.from_bytes(buf[2 * j:2 * j + 2], "little")
        if code >= 32:
            chars.append(chr(code))
        elif code in (10, 13):
            chars.append("\n")
        elif code in (9, 30, 31):  # 탭·묶음 빈칸·고정폭 빈칸
            chars.append(" ")
        j += 1 if code >= 32 or code in _HWP_CHAR_CONTROLS else 8
    return "".join(chars)


class _HTMLText(HTMLParser):
    # 링크(a) 글자도 뺀다 — 게시판 화면의 메뉴·바로가기는 링크이고 게시글 본문은 일반 글자다
    _SKIP = {"script", "style", "noscript", "head", "nav", "header", "footer", "a", "select", "button"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._depth:
            self._depth -= 1
        elif tag in {"p", "div", "li", "tr", "br", "h1", "h2", "h3", "h4", "td"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._depth:
            self.parts.append(data)


def _html_text(markup: str) -> str:
    """HTML 화면의 본문 글자 — 스크립트·머리말·메뉴·링크 글자를 빼고, 버튼·제목표시처럼 짧은 줄도 버린다.

    ponytail: 링크 글자와 줄 길이로 본문을 가린다(메뉴는 링크, 문장은 길다). 화면 구조별 본문 영역 추정은 하지 않는다
    """
    parser = _HTMLText()
    parser.feed(markup)
    lines = (line.strip() for line in "".join(parser.parts).splitlines())
    return "\n".join(line for line in lines if len(line) >= _HTML_MIN_LINE)
