"""Speaker diarization module — assigns speaker labels to each word via pyannote.audio."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_PIPELINE_ID = "pyannote/speaker-diarization-3.1"
_LABEL_PREFIX = "Character_"


def diarize(audio_path: str | Path, word_timestamps: list[dict]) -> list[dict]:
    """Runs pyannote speaker diarization and assigns speaker to each word.

    Args:
        audio_path: Path to 16kHz mono WAV file
        word_timestamps: list of dicts with keys word, timestamp_start, timestamp_end
                         (output of transcribe_from_audio())

    Returns:
        Same list with 'speaker' key added to each word dict.
        Speaker labels are strings like 'Character_A', 'Character_B', etc.
        Falls back to 'Character_A' if diarization fails or no segment matches.
    """
    audio_path = Path(audio_path)
    fallback_label = f"{_LABEL_PREFIX}A"

    try:
        segments = _run_diarization(audio_path)
        label_map = _build_label_map(segments)
        return _assign_speakers(word_timestamps, segments, label_map, fallback_label)
    except Exception as exc:  # noqa: BLE001
        logger.warning("화자 분리 실패, 전체 단어에 '%s' 할당: %s", fallback_label, exc)
        return [dict(w, speaker=fallback_label) for w in word_timestamps]


def _run_diarization(audio_path: Path) -> list[tuple[float, float, str]]:
    """Loads pyannote pipeline and returns list of (start, end, raw_label) segments."""
    import torch  # 런타임 의존성
    from pyannote.audio import Pipeline  # 런타임 의존성

    auth_token: str | None = os.getenv("PYANNOTE_AUTH_TOKEN")
    if not auth_token:
        raise EnvironmentError("PYANNOTE_AUTH_TOKEN이 설정되지 않았습니다.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("pyannote 파이프라인 로드 중 (device=%s)", device)

    pipeline = Pipeline.from_pretrained(_PIPELINE_ID, token=auth_token)
    pipeline.to(torch.device(device))

    diarization_result = pipeline(str(audio_path))

    # pyannote 4.x returns DiarizeOutput; 3.x returns Annotation
    if hasattr(diarization_result, "itertracks"):
        segments: list[tuple[float, float, str]] = [
            (turn.start, turn.end, speaker)
            for turn, _, speaker in diarization_result.itertracks(yield_label=True)
        ]
    else:
        segments = [
            (segment.start, segment.end, segment.speaker)
            for segment in diarization_result
        ]
    logger.info("화자 분리 완료: %d개 구간 검출", len(segments))
    return segments


def _build_label_map(segments: list[tuple[float, float, str]]) -> dict[str, str]:
    """Maps raw pyannote SPEAKER_XX labels to Character_A, Character_B, ... in first-appearance order."""
    seen: dict[str, str] = {}
    for _, _, raw_label in segments:
        if raw_label not in seen:
            letter = chr(ord("A") + len(seen))
            seen[raw_label] = f"{_LABEL_PREFIX}{letter}"
    return seen


def _assign_speakers(
    word_timestamps: list[dict],
    segments: list[tuple[float, float, str]],
    label_map: dict[str, str],
    fallback_label: str,
) -> list[dict]:
    """Assigns a speaker label to each word based on greatest timestamp overlap."""
    result: list[dict] = []
    for word in word_timestamps:
        w_start: float = word["timestamp_start"]
        w_end: float = word["timestamp_end"]

        best_label = fallback_label
        best_overlap = 0.0

        for seg_start, seg_end, raw_label in segments:
            overlap = min(w_end, seg_end) - max(w_start, seg_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = label_map.get(raw_label, fallback_label)

        result.append(dict(word, speaker=best_label))

    return result
