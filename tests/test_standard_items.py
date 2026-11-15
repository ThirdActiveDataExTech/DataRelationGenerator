"""표준 항목 변환 — 1단계 조사에서 본 실제 값 표기 기준."""

from datetime import date

from app.src.dataset.standard_items import (
    datagokr_record,
    display_name,
    file_date_suffix,
    hcdl_record,
    hira_record,
    kosis_period,
    kosis_record,
    license_code,
    parse_period,
    periodicity_code,
    spatial_unit,
)


def test_parse_period_free_text():
    assert parse_period("2021년 1월 ~ 2025년 12월") == (date(2021, 1, 1), date(2025, 12, 31))
    assert parse_period("1952년~2025년") == (date(1952, 1, 1), date(2025, 12, 31))
    assert parse_period("~ 2025.03.11.") == (None, date(2025, 3, 11))
    assert parse_period("2024년 9월") == (date(2024, 9, 1), date(2024, 9, 30))
    assert parse_period("해당없음") == (None, None)


def test_kosis_period_forms():
    assert kosis_period("년", "2011") == (date(2011, 1, 1), date(2011, 12, 31))
    assert kosis_period("월", "2012.07") == (date(2012, 7, 1), date(2012, 7, 31))
    assert kosis_period("월", "201207") == (date(2012, 7, 1), date(2012, 7, 31))
    assert kosis_period("분기", "1999 3/4") == (date(1999, 7, 1), date(1999, 9, 30))
    assert kosis_period("반기", "2020 2/2") == (date(2020, 7, 1), date(2020, 12, 31))


def test_codes():
    assert file_date_suffix("건강보험심사평가원_요양기관 개설 현황_20251231") == date(2025, 12, 31)
    assert file_date_suffix("목록명") is None
    assert periodicity_code("연간") == periodicity_code("1년") == "ANNUAL"
    assert periodicity_code("수시 (1회성 데이터)") == "IRREGULAR"
    assert license_code("이용허락범위 제한 없음") == "OPEN"
    assert license_code("제3자 권리 포함 : 저작권 표시/공공저작물 : 출처표시 (제 1유형)") == "KOGL-1"
    # 심평원 표기 — 유형 번호 없이 조건만 적는다
    assert license_code("저작자 표시-출처표시(공공누리) 출처, 만든이 등을 표시한다면 자유롭게 사용 가능합니다.") == "KOGL-1"
    assert license_code("공공누리 출처표시-상업적이용금지-변경금지") == "KOGL-4"
    assert periodicity_code("연 1회") == "ANNUAL" and periodicity_code("월 1회") == "MONTHLY"
    assert periodicity_code("사유 발생시") == "IRREGULAR" and periodicity_code("1회한") == "ONCE"
    assert spatial_unit("시도", "행정구역(시군구)별") == "SIGUNGU"
    assert display_name("건강보험심사평가원_요양기관 개설 현황", "건강보험심사평가원") == "요양기관 개설 현황"


def test_datagokr_record_file_row():
    rec = datagokr_record({
        "목록키": "15051057", "목록유형": "FILE", "목록명": "건강보험심사평가원_요양기관 개설 현황",
        "파일데이터명": "건강보험심사평가원_요양기관 개설 현황_20251231", "분류체계": "보건의료 - 보건의료",
        "제공기관코드": "B551182", "제공기관": "건강보험심사평가원", "업데이트 주기": "연간", "전체행": "1",
        "확장자(데이터포맷)": "pdf", "키워드": "요양기관,개설", "등록일": "2020-01-02", "수정일": "2026-01-05",
        "이용허락범위": "이용허락범위 제한 없음", "시간범위": "-", "표준데이터여부": "N", "다운로드_활용신청건수": "12",
        "목록 URL": "https://www.data.go.kr/data/15051057/fileData.do",
    })
    assert rec.dataset["dataset_id"] == "urn:datagokr:15051057"
    assert rec.dataset["external_ids"] == ["datagokr:15051057"]
    assert rec.dataset["theme"] == ["보건의료", "보건의료"]
    assert rec.dataset["accrual_periodicity"] == "ANNUAL" and rec.dataset["license"] == "OPEN"
    assert rec.dataset["temporal_start"] is None
    assert rec.file["record_count"] is None  # pdf 의 '1' 은 자리표시
    assert rec.file["coverage_end"] == date(2025, 12, 31)
    assert rec.attr["standard_data_yn"] is False and rec.attr["list_type"] == "FILE"


def test_kosis_record():
    table = {"TBL_ID": "DT_1B040B3", "TBL_NM": "행정구역(시군구)별 주민등록세대수", "ORG_ID": "101",
             "STAT_ID": "2008001", "SEND_DE": "2026-09-02"}
    meta = {
        "ORG": [{"ORG_NM": "국가데이터처"}],
        "PRD": [{"PRD_SE": "년", "STRT_PRD_DE": "1992", "END_PRD_DE": "2025"},
                {"PRD_SE": "월", "STRT_PRD_DE": "2011.01", "END_PRD_DE": "2026.08"}],
        "ITM": [{"OBJ_ID": "ITEM", "OBJ_NM": "항목", "ITM_NM": "세대수", "UNIT_NM": "세대"},
                {"OBJ_ID": "A", "OBJ_NM": "행정구역(시군구)별", "ITM_NM": "서울특별시"}],
        "CMMT": [{"CMMT_NM": "통계표", "CMMT_DC": "2024.1.1일자로 신설"}],
        "SOURCE": [{"DEPT_NM": "주민과", "DEPT_PHONE": "044-205-3158"}],
    }
    expl = [{"statsNm": "주민등록인구현황"}, {"writingPurps": "ㅇ 주민등록 인구 현황을 제공한다. 둘째 문장."},
            {"pubPeriod": "월"}, {"pubExtent": "시군구"}, {"examinObjArea": "전국"}, {"confmDt": "20080108"}]
    rec = kosis_record(table, ["인구", "주민등록인구현황"], meta, expl)
    assert rec.dataset["dataset_id"] == "urn:kosis:101:DT_1B040B3"
    assert rec.dataset["external_ids"] == ["kosis:DT_1B040B3", "kosis-stat:2008001"]
    assert rec.dataset["temporal_resolution"] == "P1M"  # 월·년 중 촘촘한 쪽
    assert rec.dataset["temporal_start"] == date(2011, 1, 1) and rec.dataset["temporal_end"] == date(2026, 8, 31)
    assert rec.dataset["accrual_periodicity"] == "MONTHLY" and rec.dataset["created_at"] == date(2008, 1, 8)
    assert rec.dataset["description"] == "「주민등록인구현황」 행정구역(시군구)별 주민등록세대수. 주민등록 인구 현황을 제공한다."
    assert rec.attr["spatial_unit"] == "SIGUNGU" and rec.attr["stat_name"] == "주민등록인구현황"
    assert rec.file["file_name"] == "국가데이터처_행정구역(시군구)별 주민등록세대수_DT_1B040B3"
    assert rec.file["column_names"] == ["행정구역(시군구)별", "세대수"]
    assert rec.file["column_profile"]["units"] == {"세대수": "세대"}


def test_hira_record_cross_posted_to_datagokr():
    """상세 화면 값 그대로 (sno 13601, 2026-11-09 조회)."""
    rec = hira_record({
        "sno": "13601", "분류": "요양기관 URL복사", "제목": "분만가능 의료기관 목록",
        "자료 상세설명": "※ 본 자료는 청구내역 데이터를 기반으로 산출된 목록으로, 실제 분만 가능 여부는 방문전 확인",
        "이용허락범위": "저작자 표시-출처표시(공공누리) 출처, 만든이 등을 표시한다면 자유롭게 사용 가능합니다.",
        "기타 유의사항": "해당 데이터는 공공데이터 포털을 통해 다운받으실 수 있습니다. https://www.data.go.kr/data/15160285/fileData.do",
        "비용부과유무": "무료", "제공형태": "다운로드", "자료 갱신주기": "없음", "부서명": "빅데이터실",
        "담당자 연락처": "033-739-1008", "작성일": "2026-07-13", "최종수정일": "2026-07-13", "조회수": "1,088",
    })
    assert rec.dataset["dataset_id"] == "urn:hira:13601" and rec.dataset["source"] == "HIRA"
    assert rec.dataset["external_ids"] == ["hira:13601", "datagokr:15160285"]
    assert rec.dataset["theme"] == ["보건의료", "요양기관"]
    assert rec.dataset["landing_page"] == "https://opendata.hira.or.kr/op/opc/selectOpenData.do?sno=13601"
    assert rec.dataset["license"] == "KOGL-1" and rec.dataset["accrual_periodicity"] is None  # '없음'
    assert rec.dataset["created_at"] == date(2026, 7, 13)
    assert rec.attr["list_type"] == "FILE" and rec.attr["dept_phone"] == "033-739-1008"
    assert rec.file["file_name"] == "건강보험심사평가원_분만가능 의료기관 목록"
    assert display_name(rec.file["file_name"], rec.dataset["publisher"]) == "분만가능 의료기관 목록"
    assert rec.metrics == {"view_count": "1088"}


def test_hira_record_openapi_without_counterpart():
    rec = hira_record({"sno": "13400", "분류": "요양기관", "제목": "(OpenAPI)자동차보험진료(청구)기관조회서비스",
                       "제공형태": "OpenAPI", "자료 갱신주기": "연 1회"})
    assert rec.dataset["external_ids"] == ["hira:13400"]
    assert rec.attr["list_type"] == "API" and rec.file["file_format"] == "API"
    assert rec.dataset["accrual_periodicity"] == "ANNUAL"


def test_hcdl_record_with_column_definitions():
    """카탈로그·컬럼 정의 API 응답 그대로 (건강보험심사평가원 명세서 일반내역, 2026-11-09 조회)."""
    item = {"prvdInstSpcd": "H0002", "instNm": "건강보험심사평가원", "clcd": "TWJHC200", "clsNm": "명세서 일반내역",
            "dltnYn": "N", "nmbrPublYn": "Y"}
    columns = [
        {"vrid": "CL_CD", "vrblNm": "요양기관종별구분코드", "prvdInstDataClcd": "문자", "lnth": "2", "vrblCont": "-",
         "cdSetYn": "Y"},
        {"vrid": "DGSBJT_CD", "vrblNm": "진료과목코드", "prvdInstDataClcd": "문자", "lnth": "2", "vrblCont": "-",
         "cdSetYn": "Y"},
        {"vrid": "AC1_yr", "vrblNm": "", "prvdInstDataClcd": "숫자", "lnth": "8", "vrblCont": "□□번", "cdSetYn": "N"},
    ]
    rec = hcdl_record(item, columns)
    assert rec.dataset["dataset_id"] == "urn:hcdl:H0002:TWJHC200" and rec.dataset["source"] == "HCDL"
    assert rec.dataset["landing_page"] == "https://hcdl.mohw.go.kr/catalog/detail/Y/TWJHC200/H0002"
    assert rec.dataset["external_ids"] == ["hcdl:H0002:TWJHC200"]
    assert rec.dataset["description"] == "건강보험심사평가원의 명세서 일반내역. 주요 항목: 요양기관종별구분코드, 진료과목코드, AC1_yr"
    # 한글 컬럼명이 없으면 영문 변수명
    assert rec.file["column_names"] == ["요양기관종별구분코드", "진료과목코드", "AC1_yr"]
    schema = rec.file["column_profile"]["schema"]
    assert schema[0] == {"name": "요양기관종별구분코드", "id": "CL_CD", "type": "문자", "length": "2",
                         "description": None, "code_set": True}
    assert schema[2]["description"] == "□□번"
    assert rec.file["file_name"] == "건강보험심사평가원_명세서 일반내역_TWJHC200"
