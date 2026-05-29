"""AssemblyAI STT + 화자 분리 모듈 — 단일 API 호출로 단어 단위 타임스탬프와 화자 레이블을 반환."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ASSEMBLYAI_API_KEY: str = os.getenv("ASSEMBLYAI_API_KEY", "")

logger = logging.getLogger(__name__)


def transcribe_with_speaker(audio_path: str | Path) -> list[dict]:
    """AssemblyAI로 STT + 화자 분리를 동시에 수행한다.

    Args:
        audio_path: 16kHz mono WAV 파일 경로

    Returns:
        [{"word": str, "timestamp_start": float, "timestamp_end": float, "speaker": str}, ...]
        speaker format: "Character_A", "Character_B", ...

    Raises:
        FileNotFoundError: audio_path가 존재하지 않을 때
        RuntimeError: API 오류 또는 결과 없을 때
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {audio_path}")

    logger.info("AssemblyAI STT 시작: %s", audio_path)

    transcript = _call_assemblyai(audio_path)
    _check_transcript_status(transcript)
    words = _extract_words(transcript)

    logger.info("AssemblyAI STT 완료: 총 %d개 단어 추출", len(words))
    return words


def _call_assemblyai(audio_path: Path):  # type: ignore[return]
    """AssemblyAI SDK로 전사 요청을 보내고 Transcript 객체를 반환한다."""
    import assemblyai as aai  # 런타임 의존성 — 테스트 시 모킹 가능

    aai.settings.api_key = ASSEMBLYAI_API_KEY

    config = aai.TranscriptionConfig(
        speaker_labels=True,
        language_code="en",
        speech_model=aai.SpeechModel.best,
    )

    return aai.Transcriber().transcribe(str(audio_path), config)


def _check_transcript_status(transcript) -> None:  # type: ignore[return]
    """전사 결과의 상태를 확인하고 에러 시 RuntimeError를 발생시킨다."""
    import assemblyai as aai

    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f"AssemblyAI 전사 실패: {transcript.error}")


def _extract_words(transcript) -> list[dict]:  # type: ignore[return]
    """Transcript 객체에서 단어 단위 타임스탬프와 화자 레이블을 추출한다."""
    raw_words = transcript.words or []

    if not raw_words:
        raise RuntimeError("AssemblyAI STT 결과 없음")

    return [
        {
            "word":            word.text.strip(),
            "timestamp_start": round(word.start / 1000, 4),
            "timestamp_end":   round(word.end / 1000, 4),
            "speaker":         f"Character_{word.speaker}" if word.speaker else "Character_A",
        }
        for word in raw_words
    ]
