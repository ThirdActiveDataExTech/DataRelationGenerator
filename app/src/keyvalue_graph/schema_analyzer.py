from collections import Counter
from typing import Any, Dict, List

import logging

from app.schemas.schema_analysis import FieldStatistics, SchemaAnalysisResult
from app.src.meta_extract.llm_response import LLMResponse

logger = logging.getLogger(__name__)


class SchemaAnalyzer:
    """데이터 샘플링 + 통계 분석 기반 스키마 추론"""

    def __init__(self, sample_size: int = 5):
        """초기화

        Args:
            sample_size: 샘플링할 레코드 개수
        """
        self.sample_size = sample_size

    def analyze(self, records: List[Dict[str, Any]]) -> SchemaAnalysisResult:
        """데이터를 분석하여 스키마 추론

        Args:
            records: 데이터 레코드 리스트

        Returns:
            SchemaAnalysisResult
        """
        if not records:
            raise ValueError("레코드가 비어있습니다")

        logger.info(f"스키마 분석 시작: {len(records)}개 레코드")

        # 1. 샘플 데이터 추출
        sample_records = self._sample_records(records)

        # 2. 필드별 통계 수집
        field_statistics = self._collect_statistics(records)

        # 3. LLM에게 분석 요청
        schema = self._llm_inference(sample_records, field_statistics)

        logger.info(
            f"스키마 분석 완료: 도메인={schema.domain}, "
            f"중심노드={schema.centrality_name}, "
            f"필드 {len(schema.fields)}개, "
            f"신뢰도={schema.confidence:.2f}"
        )

        return schema

    def _sample_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """레코드 샘플링

        Args:
            records: 전체 레코드

        Returns:
            샘플 레코드 리스트
        """
        # 균등하게 샘플링 (처음, 중간, 끝)
        if len(records) <= self.sample_size:
            return records

        step = len(records) // self.sample_size
        indices = [i * step for i in range(self.sample_size)]
        samples = [records[i] for i in indices]

        logger.info(f"샘플링: {len(records)}개 중 {len(samples)}개 추출")
        return samples

    def _collect_statistics(
            self, records: List[Dict[str, Any]]
    ) -> Dict[str, FieldStatistics]:
        """필드별 통계 정보 수집

        Args:
            records: 전체 레코드

        Returns:
            필드명 → FieldStatistics 딕셔너리
        """
        # 모든 필드명 추출
        all_fields = set()
        for record in records:
            all_fields.update(record.keys())

        statistics = {}
        total_count = len(records)

        for field in all_fields:
            # 필드별 값 수집
            values = []
            null_count = 0

            for record in records:
                value = record.get(field)
                if value is None or value == "":
                    null_count += 1
                else:
                    values.append(str(value))

            # 통계 계산
            unique_count = len(set(values))
            null_ratio = null_count / total_count if total_count > 0 else 0.0

            # 샘플 값 추출 (최대 5개, 중복 제거 후 빈도순)
            value_counter = Counter(values)
            samples = [val for val, _ in value_counter.most_common(5)]

            # 값 길이 통계
            if values:
                lengths = [len(v) for v in values]
                value_lengths = {
                    "min": min(lengths),
                    "max": max(lengths),
                    "avg": sum(lengths) // len(lengths),
                }
            else:
                value_lengths = {"min": 0, "max": 0, "avg": 0}

            statistics[field] = FieldStatistics(
                total_count=total_count,
                unique_count=unique_count,
                null_count=null_count,
                null_ratio=round(null_ratio, 3),
                samples=samples,
                value_lengths=value_lengths,
            )

        logger.info(f"통계 수집 완료: {len(statistics)}개 필드")
        return statistics

    def _llm_inference(
            self,
            sample_records: List[Dict[str, Any]],
            field_statistics: Dict[str, FieldStatistics],
    ) -> SchemaAnalysisResult:
        """LLM으로 스키마 추론

        Args:
            sample_records: 샘플 레코드
            field_statistics: 필드별 통계

        Returns:
            SchemaAnalysisResult
        """
        # 샘플 데이터를 보기 좋게 포맷팅
        import json

        sample_data_str = json.dumps(sample_records, ensure_ascii=False, indent=2)

        # 통계를 보기 좋게 포맷팅
        statistics_lines = []
        for field, stats in field_statistics.items():
            line = (
                f"{field}:\n"
                f"  - 전체: {stats.total_count}개\n"
                f"  - 고유값: {stats.unique_count}개\n"
                f"  - Null: {stats.null_count}개 ({stats.null_ratio * 100:.1f}%)\n"
                f"  - 샘플: {stats.samples}\n"
                f"  - 길이: min={stats.value_lengths['min']}, "
                f"max={stats.value_lengths['max']}, "
                f"avg={stats.value_lengths['avg']}"
            )
            statistics_lines.append(line)

        statistics_str = "\n\n".join(statistics_lines)

        # LLM 호출
        llm_response = LLMResponse("schema_analysis")
        result = llm_response.generate_llm_response(
            variables={"sample_data": sample_data_str, "statistics": statistics_str}
        )

        # 결과가 딕셔너리인지 확인
        if not isinstance(result, dict):
            raise ValueError(f"LLM 응답이 딕셔너리가 아닙니다: {type(result)}")

        # Pydantic 모델로 변환
        schema = SchemaAnalysisResult(
            domain=result["domain"],
            centrality_name=result["centrality_name"],
            fields=result["fields"],
            confidence=result.get("confidence", 0.0),
            notes=result.get("notes", "")
        )

        return schema
