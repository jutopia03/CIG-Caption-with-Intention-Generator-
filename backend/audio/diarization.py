"""Speaker diarization module using NVIDIA NeMo Sortformer."""

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

SORTFORMER_MODEL_NAME: str = os.getenv(
    "SORTFORMER_MODEL_NAME",
    "nvidia/diar_sortformer_4spk-v1",
)
SORTFORMER_MODEL_PATH: str = os.getenv("SORTFORMER_MODEL_PATH", "")
SORTFORMER_BATCH_SIZE: int = int(os.getenv("SORTFORMER_BATCH_SIZE", "1"))
SPEAKER_SMOOTH_MAX_SEC: float = float(os.getenv("SPEAKER_SMOOTH_MAX_SEC", "1.2"))
SPEAKER_SMOOTH_MAX_WORDS: int = int(os.getenv("SPEAKER_SMOOTH_MAX_WORDS", "3"))
_LABEL_PREFIX = "Character_"


def diarize(audio_path: str | Path, word_timestamps: list[dict]) -> list[dict]:
    """Run Sortformer diarization and assign speaker labels to each word."""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    fallback_label = f"{_LABEL_PREFIX}A"

    try:
        segments = _run_diarization(audio_path)
        label_map = _build_label_map(segments)
        assigned_words = _assign_speakers(
            word_timestamps,
            segments,
            label_map,
            fallback_label,
        )
        return _smooth_speaker_islands(assigned_words)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Sortformer diarization failed; assigning all words to '%s': %s",
            fallback_label,
            exc,
        )
        return [dict(w, speaker=fallback_label) for w in word_timestamps]


def _run_diarization(audio_path: Path) -> list[tuple[float, float, str]]:
    """Load Sortformer and return a list of (start, end, raw_label) segments."""
    import torch  # runtime dependency
    from nemo.collections.asr.models import SortformerEncLabelModel  # runtime dependency

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Sortformer loading (model=%s, device=%s)", _model_source(), device)

    if SORTFORMER_MODEL_PATH:
        diar_model = SortformerEncLabelModel.restore_from(
            restore_path=SORTFORMER_MODEL_PATH,
            map_location=device,
            strict=False,
        )
    else:
        diar_model = SortformerEncLabelModel.from_pretrained(SORTFORMER_MODEL_NAME)

    diar_model.eval()
    if hasattr(diar_model, "to"):
        diar_model.to(torch.device(device))

    _configure_streaming_sortformer(diar_model)

    predicted = diar_model.diarize(
        audio=[str(audio_path)],
        batch_size=SORTFORMER_BATCH_SIZE,
    )
    first_audio = predicted[0] if predicted else []
    segments = [_parse_segment(segment) for segment in first_audio]
    valid_segments = [segment for segment in segments if segment is not None]

    if not valid_segments:
        raise RuntimeError("Sortformer diarization returned no segments")

    logger.info("Sortformer diarization complete: %d segments", len(valid_segments))
    return valid_segments


def _model_source() -> str:
    """Return the configured Sortformer model source for logging."""
    return SORTFORMER_MODEL_PATH or SORTFORMER_MODEL_NAME


def _configure_streaming_sortformer(diar_model: Any) -> None:
    """Apply optional streaming Sortformer parameters from environment variables."""
    modules = getattr(diar_model, "sortformer_modules", None)
    if modules is None:
        return

    env_to_attr = {
        "SORTFORMER_CHUNK_LEN": "chunk_len",
        "SORTFORMER_RIGHT_CONTEXT": "chunk_right_context",
        "SORTFORMER_FIFO_LEN": "fifo_len",
        "SORTFORMER_SPKCACHE_UPDATE_PERIOD": "spkcache_update_period",
        "SORTFORMER_SPKCACHE_LEN": "spkcache_len",
    }
    for env_name, attr_name in env_to_attr.items():
        raw_value = os.getenv(env_name)
        if raw_value:
            setattr(modules, attr_name, int(raw_value))

    check = getattr(modules, "_check_streaming_parameters", None)
    if callable(check):
        check()


def _parse_segment(segment: Any) -> tuple[float, float, str] | None:
    """Parse Sortformer segment formats into (start, end, speaker)."""
    if isinstance(segment, str):
        parts = segment.replace(",", " ").split()
        if len(parts) < 3:
            return None
        return (float(parts[0]), float(parts[1]), parts[2])

    if isinstance(segment, dict):
        start = segment.get("start", segment.get("begin", segment.get("start_time")))
        end = segment.get("end", segment.get("end_time"))
        speaker = segment.get("speaker", segment.get("label", segment.get("speaker_id")))
        if start is None or end is None or speaker is None:
            return None
        return (float(start), float(end), str(speaker))

    if isinstance(segment, (list, tuple)) and len(segment) >= 3:
        return (float(segment[0]), float(segment[1]), str(segment[2]))

    return None


def _build_label_map(segments: list[tuple[float, float, str]]) -> dict[str, str]:
    """Map raw Sortformer speaker labels to Character_A, Character_B, ... order."""
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
    """Assign a speaker label to each word based on greatest timestamp overlap."""
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


def _smooth_speaker_islands(words: list[dict]) -> list[dict]:
    """Merge short A-B-A speaker label flips back into the surrounding speaker."""
    if len(words) < 3:
        return words

    groups: list[tuple[int, int, str]] = []
    start = 0
    for index, word in enumerate(words[1:], start=1):
        if word.get("speaker") != words[start].get("speaker"):
            groups.append((start, index, words[start].get("speaker", f"{_LABEL_PREFIX}A")))
            start = index
    groups.append((start, len(words), words[start].get("speaker", f"{_LABEL_PREFIX}A")))

    smoothed = [dict(word) for word in words]
    for group_index in range(1, len(groups) - 1):
        prev_start, prev_end, prev_speaker = groups[group_index - 1]
        curr_start, curr_end, curr_speaker = groups[group_index]
        next_start, next_end, next_speaker = groups[group_index + 1]

        if prev_speaker != next_speaker or curr_speaker == prev_speaker:
            continue

        duration = (
            words[curr_end - 1]["timestamp_end"] - words[curr_start]["timestamp_start"]
        )
        word_count = curr_end - curr_start
        if duration <= SPEAKER_SMOOTH_MAX_SEC or word_count <= SPEAKER_SMOOTH_MAX_WORDS:
            for index in range(curr_start, curr_end):
                smoothed[index]["speaker"] = prev_speaker

    changed = sum(
        1
        for before, after in zip(words, smoothed)
        if before.get("speaker") != after.get("speaker")
    )
    if changed:
        logger.info("Speaker smoothing relabeled %d short-flip words", changed)

    return smoothed
