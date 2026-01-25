from typing import List

from sentence_transformers import SentenceTransformer

from app.config import settings
from app.exceptions.base import ApplicationError


class EmbeddingService:
    """Embedding 생성 로직"""

    def __init__(self):
        """초기화 - SentenceTransformer 모델 로드"""
        self._model_name: str = settings.EMBEDDING_MODEL_NAME
        self._model: SentenceTransformer = SentenceTransformer(self._model_name)

    def generate_embedding(self, text: str) -> List[float]:
        """텍스트를 embedding 벡터로 변환

        Args:
            text: 변환할 텍스트

        Returns:
            embedding 벡터 (float list, 차원은 모델에 따라 결정됨)

        Raises:
            ApplicationError: 텍스트가 비어있거나 embedding 생성 실패
        """
        if not text or not text.strip():
            raise ApplicationError(
                code=400,
                message="입력 텍스트가 비어있습니다.",
                result={"error_type": "empty_text"}
            )
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()


embedding_service = EmbeddingService()
