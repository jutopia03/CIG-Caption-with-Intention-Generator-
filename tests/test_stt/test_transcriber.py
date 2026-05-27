"""backend.stt.transcriber 유닛 테스트."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# _extract_audio와 _run_whisper를 모킹해 실제 ffmpeg/Whisper 없이 테스트한다.
_FAKE_WORDS = [
    {"word": "Watch", "timestamp_start": 0.0, "timestamp_end": 0.42},
    {"word": "closely", "timestamp_start": 0.5, "timestamp_end": 1.1},
    {"word": "now", "timestamp_start": 1.2, "timestamp_end": 1.55},
]


def _make_mock_path(exists: bool = True) -> MagicMock:
    p = MagicMock(spec=Path)
    p.exists.return_value = exists
    p.__str__ = lambda self: "/fake/video.mp4"
    return p


@patch("backend.stt.transcriber._run_whisper", return_value=_FAKE_WORDS)
@patch("backend.stt.transcriber._extract_audio")
def test_transcribe_returns_list(mock_extract, mock_whisper, tmp_path):
    """transcribe()의 반환값이 list여야 한다."""
    from backend.stt.transcriber import transcribe

    fake_video = tmp_path / "video.mp4"
    fake_video.touch()

    mock_extract.return_value = tmp_path / "audio.wav"
    (tmp_path / "audio.wav").touch()

    result = transcribe(str(fake_video))
    assert isinstance(result, list)


@patch("backend.stt.transcriber._run_whisper", return_value=_FAKE_WORDS)
@patch("backend.stt.transcriber._extract_audio")
def test_transcribe_word_keys(mock_extract, mock_whisper, tmp_path):
    """각 항목에 word / timestamp_start / timestamp_end 키가 있어야 한다."""
    from backend.stt.transcriber import transcribe

    fake_video = tmp_path / "video.mp4"
    fake_video.touch()

    mock_extract.return_value = tmp_path / "audio.wav"
    (tmp_path / "audio.wav").touch()

    result = transcribe(str(fake_video))
    for item in result:
        assert "word" in item
        assert "timestamp_start" in item
        assert "timestamp_end" in item


@patch("backend.stt.transcriber._run_whisper", return_value=_FAKE_WORDS)
@patch("backend.stt.transcriber._extract_audio")
def test_transcribe_timestamps_order(mock_extract, mock_whisper, tmp_path):
    """timestamp_start가 오름차순으로 정렬되어야 한다."""
    from backend.stt.transcriber import transcribe

    fake_video = tmp_path / "video.mp4"
    fake_video.touch()

    mock_extract.return_value = tmp_path / "audio.wav"
    (tmp_path / "audio.wav").touch()

    result = transcribe(str(fake_video))
    starts = [item["timestamp_start"] for item in result]
    assert starts == sorted(starts)


def test_file_not_found():
    """존재하지 않는 파일 경로를 넘기면 FileNotFoundError가 발생해야 한다."""
    from backend.stt.transcriber import transcribe

    with pytest.raises(FileNotFoundError):
        transcribe("/nonexistent/path/video.mp4")
