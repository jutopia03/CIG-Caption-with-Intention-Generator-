"""Whisper STT 모듈 — 영상에서 음성을 추출하고 단어 단위 타임스탬프를 반환."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "base")


def transcribe(video_path: str | Path) -> list[dict]:
    """Whisper로 영상을 전사하고 단어 단위 타임스탬프 목록을 반환한다.

    Args:
        video_path: 입력 영상 파일 경로

    Returns:
        [{"word": str, "start": float, "end": float}, ...] 형태의 단어 목록
    """
    # TODO: whisper.load_model(WHISPER_MODEL_SIZE)로 모델 로드
    # TODO: model.transcribe(str(video_path), word_timestamps=True) 실행
    # TODO: result["segments"][*]["words"]에서 word-level timestamps 평탄화
    # TODO: [{"word": w["word"], "start": w["start"], "end": w["end"]}, ...] 반환
    raise NotImplementedError
