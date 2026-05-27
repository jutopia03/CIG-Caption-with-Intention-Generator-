"""backend.audio.analyzer 유닛 테스트."""

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_WORD_TIMESTAMPS = [
    {"word": "You",  "timestamp_start": 0.7,  "timestamp_end": 1.14},
    {"word": "know", "timestamp_start": 1.14, "timestamp_end": 1.28},
    {"word": "what", "timestamp_start": 1.3,  "timestamp_end": 1.6},
]

# 총 32000 샘플(2초), hop_length=512 기준 약 63 프레임
_N_FRAMES = 63


def _make_librosa_mock() -> MagicMock:
    """librosa 동작을 흉내내는 Mock을 반환한다."""
    lib = MagicMock()
    lib.load.return_value = (np.zeros(32000, dtype=np.float32), 16000)
    lib.note_to_hz.side_effect = lambda note: {"C2": 65.41, "C7": 2093.0}[note]

    # pyin: f0, voiced_flag, voiced_prob 반환
    f0 = np.full(_N_FRAMES, 220.0)
    voiced_flag = np.ones(_N_FRAMES, dtype=bool)
    voiced_prob = np.ones(_N_FRAMES)
    lib.pyin.return_value = (f0, voiced_flag, voiced_prob)

    # frames_to_time: 프레임 인덱스를 시간으로 변환 (hop=512, sr=16000)
    lib.frames_to_time.return_value = np.linspace(0.0, 2.0, _N_FRAMES)

    # RMS: shape (1, n) 배열 반환
    lib.feature.rms.return_value = np.array([[0.1, 0.2, 0.15]])

    return lib


def _patched_analyze(tmp_path: Path, words: list[dict] | None = None) -> list[dict]:
    """librosa를 sys.modules에 주입해 analyze()를 실행하는 헬퍼."""
    lib_mock = _make_librosa_mock()
    sys.modules["librosa"] = lib_mock
    try:
        import backend.audio.analyzer as mod
        importlib.reload(mod)

        fake_audio = tmp_path / "audio.wav"
        fake_audio.touch()
        return mod.analyze(str(fake_audio), words if words is not None else _WORD_TIMESTAMPS)
    finally:
        sys.modules.pop("librosa", None)


def test_analyze_returns_list(tmp_path):
    """analyze()의 반환값이 list여야 한다."""
    result = _patched_analyze(tmp_path)
    assert isinstance(result, list)


def test_analyze_adds_levels(tmp_path):
    """각 항목에 volume_level, pitch_level 키가 추가되어야 한다."""
    result = _patched_analyze(tmp_path)
    for item in result:
        assert "volume_level" in item
        assert "pitch_level" in item


def test_analyze_level_range(tmp_path):
    """모든 level 값이 1~5 범위 안에 있어야 한다."""
    result = _patched_analyze(tmp_path)
    for item in result:
        assert 1 <= item["volume_level"] <= 5
        assert 1 <= item["pitch_level"] <= 5


def test_analyze_empty_input(tmp_path):
    """빈 리스트를 넘기면 빈 리스트를 반환해야 한다."""
    result = _patched_analyze(tmp_path, words=[])
    assert result == []


def test_file_not_found():
    """존재하지 않는 파일 경로를 넘기면 FileNotFoundError가 발생해야 한다."""
    lib_mock = _make_librosa_mock()
    sys.modules["librosa"] = lib_mock
    try:
        import backend.audio.analyzer as mod
        importlib.reload(mod)

        with pytest.raises(FileNotFoundError):
            mod.analyze("/nonexistent/path/audio.wav", _WORD_TIMESTAMPS)
    finally:
        sys.modules.pop("librosa", None)


def test_unvoiced_words_get_level_3(tmp_path):
    """voiced 프레임이 전혀 없는 단어는 pitch_level=3이어야 한다."""
    lib_mock = _make_librosa_mock()
    # 전 구간 voiced_flag=False → 모든 단어가 피치 미검출
    lib_mock.pyin.return_value = (
        np.full(_N_FRAMES, np.nan),
        np.zeros(_N_FRAMES, dtype=bool),
        np.zeros(_N_FRAMES),
    )
    sys.modules["librosa"] = lib_mock
    try:
        import backend.audio.analyzer as mod
        importlib.reload(mod)

        fake_audio = tmp_path / "audio.wav"
        fake_audio.touch()
        result = mod.analyze(str(fake_audio), _WORD_TIMESTAMPS)
    finally:
        sys.modules.pop("librosa", None)

    for item in result:
        assert item["pitch_level"] == 3
