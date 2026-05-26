"""Ollama LLM 감정 추론 모듈 — JSON mode로 각 문장의 단어별 감정을 태깅."""

import os

from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")


def tag_emotions(sentences: list[dict]) -> list[dict]:
    """각 문장의 단어별로 감정(joy|sadness|anger|neutral)을 태깅한다.

    Args:
        sentences: volume_level, pitch_level이 포함된 문장/단어 구조

    Returns:
        각 단어에 emotion 필드가 추가된 문장 목록
    """
    # TODO: ollama.Client(host=OLLAMA_HOST) 생성
    # TODO: 문장 텍스트와 볼륨/피치 컨텍스트를 담은 JSON mode 프롬프트 구성
    # TODO: client.chat(model=OLLAMA_MODEL, format="json", messages=[...]) 호출
    # TODO: 응답 JSON 파싱 → 각 단어에 emotion 필드 매핑
    # TODO: 파싱 실패 시 emotion="neutral"로 폴백 처리
    raise NotImplementedError
