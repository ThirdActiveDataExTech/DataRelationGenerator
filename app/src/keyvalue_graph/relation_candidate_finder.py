"""관계 후보 Document 발굴 서비스"""

import logging
from typing import Any

from app.repositories.graph_repository import GraphRepository

logger = logging.getLogger(__name__)


class RelationCandidate:
    """관계 후보 Document 정보"""

    def __init__(
        self,
        doc_id: str,
        aligned_attribute_score: float,
        typed_value_match_score: float,
        graph_structure_similarity: float,
        document_semantic_score: float,
        matched_keys: list[str],
        matched_values: dict[str, str],
    ):
        self.doc_id = doc_id
        self.aligned_attribute_score = aligned_attribute_score  # Key 정렬 (의미적 스키마 매칭)
        self.typed_value_match_score = typed_value_match_score  # 타입별 값 비교 (numeric/set/text)
        self.graph_structure_similarity = graph_structure_similarity  # GraphSAGE 구조 유사도
        self.document_semantic_score = document_semantic_score  # 문서 전체 의미 유사도
        self.matched_keys = matched_keys
        self.matched_values = matched_values
        self.final_score = self._calculate_final_score()

    def _calculate_final_score(self) -> float:
        """최종 점수 계산

        가중치 설계 근거 (NormalizedValue 제외):
        1. aligned_attribute_score (40%): Key 의미 정렬 - 다른 스키마 간 속성 매칭
        2. typed_value_match_score (30%): 타입별 값 비교 - numeric/set/text 각각 적합한 방식
        3. graph_structure_similarity (20%): GraphSAGE - 비슷한 연결 패턴
        4. document_semantic_score (10%): 문서 전체 유사도 - 보조 지표
        """
        return (
            0.40 * self.aligned_attribute_score +
            0.30 * self.typed_value_match_score +
            0.20 * self.graph_structure_similarity +
            0.10 * self.document_semantic_score
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "score": round(self.final_score, 3),
            "breakdown": {
                "aligned_attribute": round(self.aligned_attribute_score, 3),
                "typed_value_match": round(self.typed_value_match_score, 3),
                "graph_structure": round(self.graph_structure_similarity, 3),
                "document_semantic": round(self.document_semantic_score, 3),
            },
            "matched_keys": self.matched_keys,
            "matched_values": self.matched_values,
        }


class RelationCandidateFinder:
    """관계 후보 Document 발굴 서비스

    4가지 전략 조합 (NormalizedValue 제외):
    1. Key Alignment: 의미적 스키마 매칭
    2. Typed Value Comparison: 타입별 값 비교 (numeric/set/text)
    3. GraphSAGE: 구조적 유사도
    4. Document Semantic: 문서 전체 유사도
    """

    def __init__(self, graph_repo: GraphRepository | None = None):
        self.graph_repo = graph_repo or GraphRepository()

    def find_candidates(
        self,
        source_doc_id: str,
        important_keys: list[str],
        important_values: dict[str, list[str]],
        top_k: int = 10,
    ) -> list[RelationCandidate]:
        """관계 후보 Document 검색

        Args:
            source_doc_id: 기준 Document ID
            important_keys: 중요 Key 목록 (LLM이 선정)
            important_values: 중요 Key의 실제 값들 {key_name: [values]}
            top_k: 반환할 후보 개수

        Returns:
            RelationCandidate 리스트 (점수 순)
        """
        logger.info(f"관계 후보 검색 시작: {source_doc_id}, 중요 Key={important_keys}")

        # 1. Key Alignment: 의미적 스키마 매칭
        aligned_candidates = self._find_by_key_alignment(source_doc_id, important_keys)

        # 2. Typed Value Comparison: 타입별 값 비교
        typed_value_candidates = self._typed_value_comparison(source_doc_id, important_keys, important_values)

        # 3. 모든 후보 통합 (중복 제거)
        all_candidate_ids = set(
            list(aligned_candidates.keys()) + list(typed_value_candidates.keys())
        )

        # 4. 각 후보에 대해 종합 점수 계산
        candidates = []
        for candidate_id in all_candidate_ids:
            aligned = aligned_candidates.get(candidate_id, {"score": 0.0, "matched_keys": []})
            typed_val = typed_value_candidates.get(candidate_id, {"score": 0.0, "matched_values": {}})

            # GraphSAGE 구조 유사도 (추후 구현)
            graph_sim = self._get_graph_structure_similarity(source_doc_id, candidate_id)

            # 문서 의미 유사도 (보조 지표)
            doc_semantic = self._get_document_semantic_similarity(source_doc_id, candidate_id)

            candidate = RelationCandidate(
                doc_id=candidate_id,
                aligned_attribute_score=aligned["score"],
                typed_value_match_score=typed_val["score"],
                graph_structure_similarity=graph_sim,
                document_semantic_score=doc_semantic,
                matched_keys=aligned["matched_keys"],
                matched_values=typed_val["matched_values"],
            )
            candidates.append(candidate)

        # 5. 최종 점수로 정렬
        candidates.sort(key=lambda x: x.final_score, reverse=True)

        logger.info(f"관계 후보 {len(candidates)}개 발견, 상위 {top_k}개 반환")
        return candidates[:top_k]

    @staticmethod
    def _numeric_proximity(val1: float, val2: float, scale: float) -> float:
        """숫자 값 근접도 계산

        Args:
            val1, val2: 비교할 숫자
            scale: 의미 있는 차이 기준 (예: 병상수=1000, 연도=10)

        Returns:
            0~1 점수 (가까울수록 1)
        """
        return max(0.0, 1.0 - abs(val1 - val2) / scale)

    @staticmethod
    def _set_overlap(set1: set[str], set2: set[str]) -> float:
        """집합 중첩도 (Jaccard similarity)

        Args:
            set1, set2: 비교할 집합

        Returns:
            0~1 점수 (Jaccard index)
        """
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union

    def _text_similarity(self, text1: str, text2: str) -> float:
        """텍스트 의미 유사도 (임베딩 기반)

        Args:
            text1, text2: 비교할 텍스트

        Returns:
            0~1 점수 (코사인 유사도)
        """
        query = """
        WITH $text1 as t1, $text2 as t2
        // Value 노드에서 임베딩 조회 (기존 임베딩 재사용)
        MATCH (v1:Value {name: t1})
        MATCH (v2:Value {name: t2})
        WHERE v1.value_embedding IS NOT NULL AND v2.value_embedding IS NOT NULL
        RETURN vector.similarity.cosine(v1.value_embedding, v2.value_embedding) as similarity
        """

        results = self.graph_repo.execute_query(query, {"text1": text1, "text2": text2})
        if results and results[0]["similarity"] is not None:
            return max(0.0, results[0]["similarity"])
        return 0.0

    def _find_by_key_alignment(
        self, source_doc_id: str, important_keys: list[str]
    ) -> dict[str, dict]:
        """Key Alignment: 의미적 스키마 매칭

        "지역" ↔ "소재지", "병상수" ↔ "병상 수" 같은 스키마 차이 해결

        Returns:
            {
                "doc_id_1": {"score": 0.85, "matched_keys": ["소재지", "병상 수"]},
                ...
            }
        """

        query = """
        // 1. 기준 Document의 Key들 조회
        MATCH (d:Document {id: $source_doc_id})-[:EXTRACTED_FROM]-(v:Value)-[:HAS_VALUE]-(k:Key)
        WHERE k.name IN $important_keys AND k.name_embedding IS NOT NULL
        WITH collect(DISTINCT k) as source_keys

        // 2. 벡터 유사도로 유사한 Key 찾기 (스키마 정렬)
        UNWIND source_keys as sk
        MATCH (k2:Key)
        WHERE k2.name <> sk.name AND k2.name_embedding IS NOT NULL
        WITH sk, k2,
             vector.similarity.cosine(sk.name_embedding, k2.name_embedding) as key_sim
        WHERE key_sim > 0.7

        // 3. 유사한 Key를 가진 Document 찾기
        MATCH (k2)-[:HAS_VALUE]->(:Value)-[:EXTRACTED_FROM]->(d2:Document)
        WHERE d2.id <> $source_doc_id

        WITH d2, collect(DISTINCT k2.name) as matched_keys, avg(key_sim) as avg_key_sim
        RETURN d2.id as doc_id, avg_key_sim as score, matched_keys
        ORDER BY score DESC
        LIMIT 50
        """

        results = self.graph_repo.execute_query(
            query, {"source_doc_id": source_doc_id, "important_keys": important_keys}
        )

        return {
            r["doc_id"]: {"score": r["score"], "matched_keys": r["matched_keys"]}
            for r in results
        }

    def _find_by_value_similarity(
        self, source_doc_id: str, important_values: dict[str, list[str]]
    ) -> dict[str, dict]:
        """Value 벡터 유사도로 후보 찾기

        LLM이 "대형"을 추출했지만 GraphDB에 "500병상 이상"으로 저장된 경우,
        value_embedding 벡터 유사도로 찾을 수 있음

        Returns:
            {
                "doc_id_1": {"score": 0.82, "matched_values": {"병원규모": "500병상"}},
                ...
            }
        """
        # important_values: {"병원규모": ["대형"], "지역": ["서울"]}
        all_values = [v for values in important_values.values() for v in values]

        query = """
        // 1. 기준 Document의 Value 조회 (중요 Key만)
        MATCH (d:Document {id: $source_doc_id})-[:EXTRACTED_FROM]-(v:Value)
        WHERE v.name IN $all_values AND v.value_embedding IS NOT NULL

        WITH collect(v) as source_values

        // 2. 벡터 유사도로 유사한 Value 찾기
        UNWIND source_values as sv
        MATCH (v2:Value)
        WHERE v2.name <> sv.name AND v2.value_embedding IS NOT NULL
        WITH sv, v2,
             vector.similarity.cosine(sv.value_embedding, v2.value_embedding) as value_sim
        WHERE value_sim > 0.75

        // 3. 유사한 Value를 가진 Document 찾기
        MATCH (v2)-[:EXTRACTED_FROM]->(d2:Document)
        WHERE d2.id <> $source_doc_id

        WITH d2, collect({key: v2.key_name, value: v2.name}) as matched_values, avg(value_sim) as avg_value_sim
        RETURN d2.id as doc_id, avg_value_sim as score, matched_values
        ORDER BY score DESC
        LIMIT 50
        """

        results = self.graph_repo.execute_query(
            query, {"source_doc_id": source_doc_id, "all_values": all_values}
        )

        candidates = {}
        for r in results:
            matched_dict = {mv["key"]: mv["value"] for mv in r["matched_values"]}
            candidates[r["doc_id"]] = {"score": r["score"], "matched_values": matched_dict}

        return candidates

    def _typed_value_comparison(
        self,
        source_doc_id: str,
        important_keys: list[str],
        important_values: dict[str, list[str]],
    ) -> dict[str, dict]:
        """Typed Value Comparison: 타입별 값 비교

        Key 타입에 따라 적합한 비교 방식 적용:
        - numeric: proximity score (예: 병상수 1500 vs 1400)
        - categorical (set): Jaccard overlap (예: 진료과목)
        - text: embedding similarity (예: 자유 서술형)

        Returns:
            {
                "doc_id_1": {"score": 0.75, "matched_values": {"병상수": "1400", "진료과목": "내과"}},
                ...
            }
        """
        candidates = {}

        # 1. 기준 Document의 Key-Value 조회
        query_source = """
        MATCH (d:Document {id: $source_doc_id})-[:EXTRACTED_FROM]-(v:Value)-[:HAS_VALUE]-(k:Key)
        WHERE k.name IN $important_keys
        RETURN k.name as key_name, v.name as value_name, v.value_type as value_type
        """
        source_results = self.graph_repo.execute_query(
            query_source, {"source_doc_id": source_doc_id, "important_keys": important_keys}
        )

        if not source_results:
            return candidates

        # Key별로 그룹화
        source_kv_map = {}
        for r in source_results:
            key = r["key_name"]
            if key not in source_kv_map:
                source_kv_map[key] = {"values": [], "type": r["value_type"]}
            source_kv_map[key]["values"].append(r["value_name"])

        # 2. 각 Key별로 유사한 값을 가진 Document 찾기
        for key_name, kv_info in source_kv_map.items():
            value_type = kv_info["type"]
            source_values = kv_info["values"]

            if value_type == "numeric":
                # Numeric: 범위 기반 검색
                self._find_by_numeric_proximity(
                    key_name, source_values, candidates
                )
            elif value_type == "set" or len(source_values) > 1:
                # Set/Categorical: Jaccard overlap
                self._find_by_set_overlap(
                    source_doc_id, key_name, source_values, candidates
                )
            else:
                # Text: 임베딩 유사도
                self._find_by_text_similarity(
                    source_doc_id, key_name, source_values[0], candidates
                )

        # 3. 점수 정규화 (여러 Key 종합)
        for doc_id in candidates:
            candidates[doc_id]["score"] = candidates[doc_id]["score"] / max(len(source_kv_map), 1)

        return candidates

    def _find_by_numeric_proximity(
        self, key_name: str, source_values: list[str], candidates: dict
    ):
        """숫자형 값 근접도로 후보 찾기"""
        try:
            source_num = float(source_values[0])
        except (ValueError, IndexError):
            return

        # 숫자 scale 추정 (동적 설정)
        scale = max(source_num * 0.5, 100)  # 값의 50% 또는 최소 100

        query = """
        MATCH (k:Key {name: $key_name})-[:HAS_VALUE]->(v:Value)-[:EXTRACTED_FROM]->(d:Document)
        WHERE v.value_type = 'numeric'
        RETURN d.id as doc_id, v.name as value_name
        """
        results = self.graph_repo.execute_query(query, {"key_name": key_name})

        for r in results:
            try:
                target_num = float(r["value_name"])
                proximity = self._numeric_proximity(source_num, target_num, scale)

                if proximity > 0.5:  # 최소 유사도 threshold
                    doc_id = r["doc_id"]
                    if doc_id not in candidates:
                        candidates[doc_id] = {"score": 0.0, "matched_values": {}}
                    candidates[doc_id]["score"] += proximity
                    candidates[doc_id]["matched_values"][key_name] = r["value_name"]
            except ValueError:
                continue

    def _find_by_set_overlap(
        self, source_doc_id: str, key_name: str, source_values: list[str], candidates: dict
    ):
        """집합형 값 Jaccard overlap으로 후보 찾기"""
        source_set = set(source_values)

        query = """
        MATCH (k:Key {name: $key_name})-[:HAS_VALUE]->(v:Value)-[:EXTRACTED_FROM]->(d:Document)
        WHERE d.id <> $source_doc_id
        WITH d, collect(DISTINCT v.name) as values
        RETURN d.id as doc_id, values
        """
        results = self.graph_repo.execute_query(
            query, {"key_name": key_name, "source_doc_id": source_doc_id}
        )

        for r in results:
            target_set = set(r["values"])
            overlap = self._set_overlap(source_set, target_set)

            if overlap > 0.3:  # 최소 30% 중첩
                doc_id = r["doc_id"]
                if doc_id not in candidates:
                    candidates[doc_id] = {"score": 0.0, "matched_values": {}}
                candidates[doc_id]["score"] += overlap
                candidates[doc_id]["matched_values"][key_name] = ", ".join(r["values"])

    def _find_by_text_similarity(
        self, source_doc_id: str, key_name: str, source_value: str, candidates: dict
    ):
        """텍스트형 값 임베딩 유사도로 후보 찾기"""
        query = """
        MATCH (d1:Document {id: $source_doc_id})-[:EXTRACTED_FROM]-(v1:Value)-[:HAS_VALUE]-(k:Key {name: $key_name})
        WHERE v1.name = $source_value AND v1.value_embedding IS NOT NULL

        MATCH (k)-[:HAS_VALUE]->(v2:Value)-[:EXTRACTED_FROM]->(d2:Document)
        WHERE d2.id <> $source_doc_id
          AND v2.value_embedding IS NOT NULL
          AND v2.name <> v1.name

        WITH d2, v2, vector.similarity.cosine(v1.value_embedding, v2.value_embedding) as similarity
        WHERE similarity > 0.7

        RETURN d2.id as doc_id, v2.name as value_name, similarity
        """

        try:
            results = self.graph_repo.execute_query(
                query,
                {"source_doc_id": source_doc_id, "key_name": key_name, "source_value": source_value},
            )

            for r in results:
                doc_id = r["doc_id"]
                if doc_id not in candidates:
                    candidates[doc_id] = {"score": 0.0, "matched_values": {}}
                candidates[doc_id]["score"] += r["similarity"]
                candidates[doc_id]["matched_values"][key_name] = r["value_name"]
        except Exception as e:
            # 벡터 임베딩이 없거나 유효하지 않으면 조용히 건너뛰기
            logger.warning(f"Text similarity 검색 실패 (key={key_name}): {e}")
            pass

    def _get_graph_structure_similarity(self, _source_doc_id: str, _target_doc_id: str) -> float:
        """GraphSAGE 기반 구조적 유사도 계산

        TODO: GraphSAGE 임베딩 구현 후 활성화
        현재는 placeholder로 0.0 반환

        Args:
            _source_doc_id: 기준 Document ID (미사용)
            _target_doc_id: 비교 대상 Document ID (미사용)

        Returns:
            구조적 유사도 (0~1)
        """
        return 0.0

    def _get_document_semantic_similarity(self, source_doc_id: str, target_doc_id: str) -> float:
        """문서 전체 의미 유사도 계산 (보조 지표)

        Document 노드의 raw_text 또는 전체 KV 임베딩 비교

        Args:
            source_doc_id: 기준 Document ID
            target_doc_id: 비교 대상 Document ID

        Returns:
            문서 의미 유사도 (0~1)
        """
        # Document에 document_embedding 속성이 있다면 사용
        query = """
        MATCH (d1:Document {id: $source_doc_id})
        MATCH (d2:Document {id: $target_doc_id})
        WHERE d1.document_embedding IS NOT NULL AND d2.document_embedding IS NOT NULL
        RETURN vector.similarity.cosine(d1.document_embedding, d2.document_embedding) as similarity
        """
        results = self.graph_repo.execute_query(
            query, {"source_doc_id": source_doc_id, "target_doc_id": target_doc_id}
        )

        if results and results[0]["similarity"] is not None:
            return max(0.0, results[0]["similarity"])

        return 0.0

    def _find_by_normalized_match(
        self, source_doc_id: str, important_keys: list[str]
    ) -> dict[str, dict]:
        """NormalizedValue 정확 매칭으로 후보 찾기 (가장 높은 신뢰도)

        정규화된 값이 정확히 일치 = 같은 개념 (Entity Resolution)

        Returns:
            {
                "doc_id_1": {"score": 1.0, "matched": {"지역": "서울", "규모": "대형"}},
                ...
            }
        """
        query = """
        // 1. 기준 Document의 NormalizedValue 조회
        MATCH (d:Document {id: $source_doc_id})-[:HAS_NORMALIZED_VALUE]->(nv:NormalizedValue)
        WHERE nv.key_name IN $important_keys

        WITH collect({key: nv.key_name, value: nv.name}) as source_normalized

        // 2. 같은 NormalizedValue를 공유하는 Document 찾기
        UNWIND source_normalized as snv
        MATCH (d2:Document)-[:HAS_NORMALIZED_VALUE]->(nv2:NormalizedValue {key_name: snv.key, name: snv.value})
        WHERE d2.id <> $source_doc_id

        WITH d2, collect({key: snv.key, value: snv.value}) as matched, count(DISTINCT snv.key) as match_count
        WITH d2, matched, toFloat(match_count) / size($important_keys) as score

        RETURN d2.id as doc_id, score, matched
        ORDER BY score DESC
        LIMIT 50
        """

        results = self.graph_repo.execute_query(
            query, {"source_doc_id": source_doc_id, "important_keys": important_keys}
        )

        candidates = {}
        for r in results:
            matched_dict = {m["key"]: m["value"] for m in r["matched"]}
            candidates[r["doc_id"]] = {"score": r["score"], "matched": matched_dict}

        return candidates

