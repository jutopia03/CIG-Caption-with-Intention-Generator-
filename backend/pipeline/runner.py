"""파이프라인 실행 모듈 — STT → 음향 분석 → 감정 태깅을 결합하여 최종 JSON 생성."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from backend.audio.analyzer import analyze
from backend.llm.emotion_tagger import tag_emotions
from backend.schemas.word_schema import Sentence
from backend.stt.transcriber import transcribe

load_dotenv()

OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "data/outputs"))


def run(video_path: str | Path) -> list[Sentence]:
    """영상 파일을 받아 전체 파이프라인을 실행하고 Sentence 목록을 반환한다.

    Args:
        video_path: 처리할 영상 파일 경로

    Returns:
        Pydantic Sentence 모델 목록 (JSON 직렬화 가능)
    """
    # TODO: transcribe(video_path)로 단어 타임스탬프 추출
    # TODO: 영상에서 오디오 추출 (ffmpeg-python) → 임시 wav 파일 저장
    # TODO: analyze(audio_path, word_timestamps)로 volume_level, pitch_level 추가
    # TODO: 단어 목록을 문장 단위로 그루핑 (구두점/무음 구간 기준)
    # TODO: tag_emotions(sentences)로 각 단어에 emotion 태깅
    # TODO: Sentence 모델로 검증 (pydantic model_validate)
    # TODO: OUTPUT_DIR / "{stem}.json"에 JSON 저장
    # TODO: Sentence 목록 반환
    raise NotImplementedError
