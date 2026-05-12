"""Neptune + OpenSearch 기반 LLM Key-Value 추출 서비스."""

import json
import logging
from typing import Dict, Any

from app.src.keyvalue_graph.data_loader import DataRecord
from app.src.keyvalue_graph.field_mappings import FIELD_NAME_MAPPING
from app.src.keyvalue_graph.neptune_graph_builder import NeptuneGraphBuilder
from app.src.keyvalue_graph.text_utils import truncate_single_record
from app.src.meta_extract.llm_response import LLMResponse

logger = logging.getLogger(__name__)


class NeptuneLLMLoader:
    """LLM으로 Key-Value 추출 후 Neptune + OpenSearch에 저장."""

    def __init__(
        self,
        neptune_builder: NeptuneGraphBuilder = None,
        field_mapping: Dict[str, str] = None
    ):
        """초기화.

        Args:
            neptune_builder: NeptuneGraphBuilder 인스턴스
            field_mapping: 필드명 매핑 (원본 → 한글)
        """
        self.neptune_builder = neptune_builder or NeptuneGraphBuilder()
        self.field_mapping = field_mapping or FIELD_NAME_MAPPING

    def extract_and_build_graph(
        self,
        data_text: str,
        centrality_name: str = "데이터현황",
        source_file: str = "llm_extract",
        record_index: int = 0,
        extractable_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """데이터에서 LLM으로 Key-Value 추출 후 Neptune + OpenSearch 저장.

        Args:
            data_text: 원본 데이터 (JSON 문자열)
            centrality_name: 중심 노드 이름
            source_file: 소스 파일명
            record_index: 레코드 인덱스
            extractable_metadata: 추출 가능한 메타데이터 정의

        Returns:
            그래프 생성 결과
        """
        try:
            # 1. JSON 파싱 시도 (실패 시 자연어 텍스트로 처리)
            try:
                raw_data = json.loads(data_text)
                is_json_input = True
            except json.JSONDecodeError:
                raw_data = {"_raw_text": data_text}
                is_json_input = False
                logger.info("JSON 형식 아님, 자연어 텍스트로 처리")

            # JSON인 경우에만 truncate 적용
            if is_json_input:
                truncated_data = truncate_single_record(raw_data, context="key_value_extract")
                truncated_data_text = json.dumps(truncated_data, ensure_ascii=False)
            else:
                truncated_data = raw_data
                truncated_data_text = data_text

            logger.info(
                f"데이터 전처리 완료: 원본 {len(data_text)}자 → {len(truncated_data_text)}자"
            )

            # 2. LLM으로 Key-Value 추출
            logger.info("LLM으로 Key-Value 추출 시작")

            # field_mapping이 있으면 힌트 추가
            if self.field_mapping:
                field_hints_lines = []

                for key, korean_name in self.field_mapping.items():
                    if extractable_metadata and key in extractable_metadata:
                        meta_info = extractable_metadata[key]
                        field_hints_lines.append(
                            f"- {key} ({korean_name}): {meta_info.description} "
                            f"[source: {meta_info.source_field}]"
                        )
                    else:
                        field_hints_lines.append(f"- {key} ({korean_name})")

                field_hints = "\n".join(field_hints_lines)
                data_with_hints = (
                    f"# 추출할 필드 목록:\n{field_hints}\n\n"
                    f"# 데이터:\n{truncated_data_text}"
                )
                logger.info(f"필드 힌트 추가: {len(field_hints_lines)}개 필드")
            else:
                data_with_hints = truncated_data_text

            llm_response = LLMResponse("key_value_extract")
            extraction_result = llm_response.generate_llm_response(
                variables={"data_text": data_with_hints}
            )

            if not isinstance(extraction_result, dict):
                raise ValueError(
                    f"LLM 응답이 dict가 아닙니다: {type(extraction_result)}"
                )

            raw_key_values = extraction_result.get("key_values", {})
            logger.info(f"추출된 Key 개수: {len(raw_key_values)}")

            # 3. Key를 한글로 매핑
            korean_key_values = self._map_keys_to_korean(raw_key_values)

            # 4. DataRecord 생성
            # 자연어 입력인 경우 _raw_text를 제거하고 LLM 추출 결과를 raw_data로 사용
            if not is_json_input and "_raw_text" in raw_data:
                # LLM이 추출한 key_values를 기반으로 raw_data 재구성
                reconstructed_raw_data = {}
                for key, values in korean_key_values.items():
                    reconstructed_raw_data[key] = values[0] if len(values) == 1 else values
                raw_data = reconstructed_raw_data

            record = DataRecord(
                source_file=source_file,
                record_index=record_index,
                raw_data=raw_data,
            )
            record.key_values = korean_key_values

            # 5. Neptune + OpenSearch 저장
            result = self.neptune_builder.build_from_record(
                record,
                centrality_name=centrality_name
            )

            if result["success"]:
                logger.info(
                    f"Neptune + OpenSearch 저장 완료: Document={result['document_id']}, "
                    f"Keys={result['keys_count']}, Values={result['values_count']}"
                )
                return {
                    "success": True,
                    "centrality_name": centrality_name,
                    "document_id": result["document_id"],
                    "keys_count": result["keys_count"],
                    "values_count": result["values_count"],
                    "extraction": {
                        "raw_keys": list(raw_key_values.keys()),
                        "korean_keys": list(korean_key_values.keys()),
                    }
                }
            else:
                logger.error(f"저장 실패: {result.get('error')}")
                return {
                    "success": False,
                    "document_id": result["document_id"],
                    "error": result.get("error"),
                    "extraction": {
                        "raw_keys": list(raw_key_values.keys()),
                        "korean_keys": list(korean_key_values.keys()),
                    }
                }

        except Exception as e:
            logger.exception(f"LLM 기반 그래프 생성 실패: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def _map_keys_to_korean(
        self,
        raw_key_values: Dict[str, list[str]]
    ) -> Dict[str, list[str]]:
        """원본 필드명을 한글로 매핑.

        Args:
            raw_key_values: 원본 Key-Value

        Returns:
            한글 매핑된 Key-Value
        """
        korean_key_values = {}

        for raw_key, values in raw_key_values.items():
            korean_key = self.field_mapping.get(raw_key, raw_key)

            if not values:
                continue

            korean_key_values[korean_key] = values

        logger.info(
            f"Key 매핑 완료: {len(raw_key_values)} → {len(korean_key_values)}"
        )
        return korean_key_values
