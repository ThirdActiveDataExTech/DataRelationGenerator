
from typing import List, Dict, Any
import logging
from sentence_transformers import CrossEncoder


class BGEReranker:
    """BGE Reranker v2-m3 기반 경량 재순위화

    LLM 검증 전에 빠르게 관련성 점수를 계산하여 후보를 필터링합니다.
    - 모델: BAAI/bge-reranker-v2-m3 (다국어 지원, 한국어 포함)
    - 속성: CrossEncoder 기반 (query-document pair 직접 점수화)
    - 성능: CPU에서 100ms/문서
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """BGE Reranker 초기화

        Args:
            model_name: HuggingFace 모델명 (기본: bge-reranker-v2-m3)
        """
        logging.info(f"BGE Reranker 로딩 중: {model_name}")
        self.model = CrossEncoder(model_name, max_length=512)
        logging.info("BGE Reranker 로딩 완료")

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int,
        text_key: str = "text"
    ) -> List[Dict[str, Any]]:
        """문서 재순위화

        Args:
            query: 검색 쿼리
            documents: 문서 목록 (각 dict에 text 필드 필요)
            top_k: 반환할 상위 문서 개수 (실제로는 top_k * 2 반환)
            text_key: 문서 텍스트 필드명 (기본: "text")

        Returns:
            재순위화된 문서 목록 (rerank_score 필드 추가)
        """
        if not documents:
            return []

        # Query-document pairs 생성 (텍스트 길이 제한)
        pairs = []
        for doc in documents:
            doc_text = doc.get(text_key, "")
            # 너무 긴 텍스트는 잘라냄 (512 토큰 제한)
            if len(doc_text) > 2000:
                doc_text = doc_text[:2000]
            pairs.append((query, doc_text))

        # 재순위화 점수 계산
        scores = self.model.predict(pairs)

        # 점수 추가
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        # 점수 순 정렬
        sorted_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)

        # 상위 top_k * 2개 반환 (LLM 검증용)
        selected_count = min(top_k * 2, len(sorted_docs))
        result = sorted_docs[:selected_count]

        return result
