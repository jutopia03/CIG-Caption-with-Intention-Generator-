"""Whisper STT 모듈 — 영상에서 음성을 추출하고 단어 단위 타임스탬프를 반환."""

import logging
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "tiny")

logger = logging.getLogger(__name__)


def transcribe(video_path: str | Path) -> list[dict]:
    """영상 파일을 전사하고 단어 단위 타임스탬프 목록을 반환한다.

    Args:
        video_path: 입력 영상 파일 경로

    Returns:
        [{"word": str, "timestamp_start": float, "timestamp_end": float}, ...]

    Raises:
        FileNotFoundError: video_path가 존재하지 않을 때
        RuntimeError: 오디오 추출 실패 또는 STT 결과가 비어있을 때
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")

    logger.info("STT 시작: %s", video_path)

    tmp_audio: Path | None = None
    try:
        tmp_audio = _extract_audio(video_path)
        words = _run_whisper(tmp_audio)
    finally:
        if tmp_audio is not None and tmp_audio.exists():
            tmp_audio.unlink()

    return words


def transcribe_from_audio(audio_path: str | Path) -> list[dict]:
    """이미 추출된 오디오 파일을 전사하고 단어 단위 타임스탬프 목록을 반환한다.

    runner.py처럼 오디오를 한 번 추출하고 STT와 음향 분석 양쪽에 넘겨야 할 때 사용.
    영상에서 직접 전사할 경우 transcribe()를 사용한다.

    Args:
        audio_path: 16kHz mono WAV 오디오 파일 경로

    Returns:
        [{"word": str, "timestamp_start": float, "timestamp_end": float}, ...]

    Raises:
        FileNotFoundError: audio_path가 존재하지 않을 때
        RuntimeError: STT 결과가 비어있을 때
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {audio_path}")

    logger.info("STT(오디오) 시작: %s", audio_path)
    return _run_whisper(audio_path)


def _extract_audio(video_path: Path) -> Path:
    """ffmpeg로 영상에서 16kHz mono WAV 오디오를 임시 파일로 추출한다."""
    import ffmpeg  # 런타임 의존성 — 테스트 시 모킹 가능

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    tmp_path = Path(tmp.name)

    try:
        (
            ffmpeg
            .input(str(video_path))
            .output(
                str(tmp_path),
                ar=16000,
                ac=1,
                format="wav",
                # 기존 임시 파일 덮어쓰기
                **{"y": None},
            )
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        tmp_path.unlink(missing_ok=True)
        detail = e.stderr.decode(errors="replace") if e.stderr else str(e)
        raise RuntimeError(f"오디오 추출 실패: {detail}") from e

    return tmp_path


def _run_whisper(audio_path: Path) -> list[dict]:
    """Whisper 모델을 로드하고 단어 단위 타임스탬프를 파싱한다."""
    import whisper  # 런타임 의존성 — 테스트 시 모킹 가능

    model = whisper.load_model(WHISPER_MODEL_SIZE)
    logger.info("Whisper 모델 로드 완료: %s", WHISPER_MODEL_SIZE)

    result = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        language="en",
    )

    segments = result.get("segments") or []
    words: list[dict] = []
    for segment in segments:
        for w in segment.get("words", []):
            words.append(
                {
                    "word": w["word"].strip(),
                    "timestamp_start": round(float(w["start"]), 4),
                    "timestamp_end": round(float(w["end"]), 4),
                }
            )

    if not words:
        raise RuntimeError("STT 결과 없음")

    logger.info("STT 완료: 총 %d개 단어 추출", len(words))
    return words
