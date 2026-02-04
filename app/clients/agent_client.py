from typing import List

import logging
from langchain_openai import ChatOpenAI

from app.config import settings


class LLMClient:
    """LLM 클라이언트 - 폴백 모델 지원"""

    def __init__(self):
        """LLM 클라이언트 초기화"""
        self.primary_model = ChatOpenAI(
            model=settings.MAIN_MODEL_NAME,
            api_key=settings.CLAUDE_API_KEY,
            base_url=settings.LITE_LLM_BASE_URL,
            temperature=0.0
        )

        # 폴백 모델들 설정 (환경 변수에서 읽어올 수 있음)
        self.fallback_models = self._create_fallback_models()

        logging.info(f"LLM Client initialized with primary model: {settings.MAIN_MODEL_NAME}")
        if self.fallback_models:
            logging.info(f"Fallback models: {[model.model_name for model in self.fallback_models]}")

    @staticmethod
    def _create_fallback_models() -> List[ChatOpenAI]:
        """폴백 모델들 생성"""
        fallback_models = []

        # 환경 변수에서 폴백 모델 리스트를 읽어올 수 있음
        # 현재는 하드코딩된 예시
        fallback_model_names = getattr(settings, 'FALLBACK_MODEL_LIST', [])

        for model_name in fallback_model_names:
            try:
                model = ChatOpenAI(
                    model=model_name,
                    api_key=settings.CLAUDE_API_KEY,
                    base_url=settings.LITE_LLM_BASE_URL,
                    temperature=0.0
                )
                fallback_models.append(model)
            except Exception as e:
                logging.warning(f"Failed to create fallback model {model_name}: {e}")

        return fallback_models


# 싱글톤 인스턴스
llm_client = LLMClient()
