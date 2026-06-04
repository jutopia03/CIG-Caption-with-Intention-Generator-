"""backend.audio.diarization 유닛 테스트."""

from pathlib import Path
from unittest.mock import patch

import numpy as np

from backend.audio.diarization import (
    _assign_speakers,
    _build_label_map,
    _cluster_embeddings,
    _parse_segment,
    _smooth_speaker_islands,
    diarize,
)

_WORDS = [
    {"word": "hello", "timestamp_start": 0.5, "timestamp_end": 1.0},
    {"word": "world", "timestamp_start": 2.5, "timestamp_end": 3.0},
]


def test_assign_speakers_basic() -> None:
    segments = [(0.0, 2.0, "SPEAKER_00"), (2.0, 4.0, "SPEAKER_01")]
    label_map = {"SPEAKER_00": "Character_A", "SPEAKER_01": "Character_B"}

    result = _assign_speakers(_WORDS, segments, label_map, "Character_A")

    assert result[0]["speaker"] == "Character_A"
    assert result[1]["speaker"] == "Character_B"


def test_assign_speakers_fallback() -> None:
    segments = [(5.0, 6.0, "SPEAKER_00")]
    label_map = {"SPEAKER_00": "Character_A"}
    words = [{"word": "hi", "timestamp_start": 0.0, "timestamp_end": 1.0}]

    result = _assign_speakers(words, segments, label_map, "Character_A")

    assert result[0]["speaker"] == "Character_A"


def test_build_label_map_order() -> None:
    segments = [
        (0.0, 1.0, "SPEAKER_01"),
        (1.0, 2.0, "SPEAKER_00"),
        (2.0, 3.0, "SPEAKER_01"),
    ]

    label_map = _build_label_map(segments)

    assert label_map["SPEAKER_01"] == "Character_A"
    assert label_map["SPEAKER_00"] == "Character_B"


def test_parse_sortformer_string_segment() -> None:
    assert _parse_segment("0.50 2.10 speaker_0") == (0.5, 2.1, "speaker_0")


def test_parse_sortformer_tuple_segment() -> None:
    assert _parse_segment((0.5, 2.1, "speaker_0")) == (0.5, 2.1, "speaker_0")


def test_smooth_speaker_islands_merges_short_flip() -> None:
    words = [
        {"word": "a", "timestamp_start": 0.0, "timestamp_end": 0.3, "speaker": "Character_A"},
        {"word": "b", "timestamp_start": 0.3, "timestamp_end": 0.6, "speaker": "Character_B"},
        {"word": "c", "timestamp_start": 0.6, "timestamp_end": 0.9, "speaker": "Character_A"},
    ]

    result = _smooth_speaker_islands(words)

    assert [word["speaker"] for word in result] == [
        "Character_A",
        "Character_A",
        "Character_A",
    ]


def test_cluster_embeddings_reuses_similar_voice_label() -> None:
    embeddings = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.99, 0.01], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
    ]

    result = _cluster_embeddings(embeddings, threshold=0.72)

    assert result == ["Voice_00", "Voice_00", "Voice_01"]


def test_diarize_fallback_on_model_error(tmp_path: Path) -> None:
    dummy_audio = tmp_path / "audio.wav"
    dummy_audio.write_bytes(b"")

    with patch(
        "backend.audio.diarization._run_diarization",
        side_effect=RuntimeError("mock model error"),
    ):
        result = diarize(str(dummy_audio), list(_WORDS))

    assert all(w["speaker"] == "Character_A" for w in result)
    for original, tagged in zip(_WORDS, result):
        assert tagged["word"] == original["word"]
        assert tagged["timestamp_start"] == original["timestamp_start"]
        assert tagged["timestamp_end"] == original["timestamp_end"]


def test_diarize_fallback_on_pipeline_error(tmp_path: Path) -> None:
    dummy_audio = tmp_path / "audio.wav"
    dummy_audio.write_bytes(b"")

    with patch(
        "backend.audio.diarization._run_diarization",
        side_effect=RuntimeError("mock error"),
    ):
        result = diarize(str(dummy_audio), list(_WORDS))

    assert all(w["speaker"] == "Character_A" for w in result)
