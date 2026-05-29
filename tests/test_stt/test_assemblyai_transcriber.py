"""backend.stt.assemblyai_transcriber 유닛 테스트."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_word(text: str, start: int, end: int, speaker: str | None) -> MagicMock:
    w = MagicMock()
    w.text = text
    w.start = start
    w.end = end
    w.speaker = speaker
    return w


def _make_transcript(words, error=None, status_error=False):
    import assemblyai as aai

    transcript = MagicMock()
    transcript.words = words
    transcript.error = error
    transcript.status = aai.TranscriptStatus.error if status_error else aai.TranscriptStatus.completed
    return transcript


@pytest.fixture()
def sample_words():
    return [
        _make_word("You",   700,  1140, "A"),
        _make_word("know",  1140, 1280, "A"),
        _make_word("where", 1300, 1600, "B"),
    ]


@patch("backend.stt.assemblyai_transcriber._call_assemblyai")
def test_returns_list(mock_call, tmp_path, sample_words):
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"")
    mock_call.return_value = _make_transcript(sample_words)

    from backend.stt.assemblyai_transcriber import transcribe_with_speaker
    result = transcribe_with_speaker(audio)

    assert isinstance(result, list)
    assert len(result) == 3


@patch("backend.stt.assemblyai_transcriber._call_assemblyai")
def test_word_keys(mock_call, tmp_path, sample_words):
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"")
    mock_call.return_value = _make_transcript(sample_words)

    from backend.stt.assemblyai_transcriber import transcribe_with_speaker
    result = transcribe_with_speaker(audio)

    for word in result:
        assert "word" in word
        assert "timestamp_start" in word
        assert "timestamp_end" in word
        assert "speaker" in word


@patch("backend.stt.assemblyai_transcriber._call_assemblyai")
def test_milliseconds_to_seconds(mock_call, tmp_path, sample_words):
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"")
    mock_call.return_value = _make_transcript(sample_words)

    from backend.stt.assemblyai_transcriber import transcribe_with_speaker
    result = transcribe_with_speaker(audio)

    assert result[0]["timestamp_start"] == round(700 / 1000, 4)
    assert result[0]["timestamp_end"] == round(1140 / 1000, 4)


@patch("backend.stt.assemblyai_transcriber._call_assemblyai")
def test_speaker_label_format(mock_call, tmp_path, sample_words):
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"")
    mock_call.return_value = _make_transcript(sample_words)

    from backend.stt.assemblyai_transcriber import transcribe_with_speaker
    result = transcribe_with_speaker(audio)

    assert result[0]["speaker"] == "Character_A"
    assert result[2]["speaker"] == "Character_B"


@patch("backend.stt.assemblyai_transcriber._call_assemblyai")
def test_none_speaker_fallback(mock_call, tmp_path):
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"")
    words = [_make_word("hi", 0, 500, None)]
    mock_call.return_value = _make_transcript(words)

    from backend.stt.assemblyai_transcriber import transcribe_with_speaker
    result = transcribe_with_speaker(audio)

    assert result[0]["speaker"] == "Character_A"


def test_file_not_found():
    from backend.stt.assemblyai_transcriber import transcribe_with_speaker
    with pytest.raises(FileNotFoundError):
        transcribe_with_speaker("/nonexistent/audio.wav")


@patch("backend.stt.assemblyai_transcriber._call_assemblyai")
def test_api_error_raises_runtime(mock_call, tmp_path):
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"")
    mock_call.return_value = _make_transcript([], error="quota exceeded", status_error=True)

    from backend.stt.assemblyai_transcriber import transcribe_with_speaker
    with pytest.raises(RuntimeError, match="AssemblyAI 전사 실패"):
        transcribe_with_speaker(audio)


@patch("backend.stt.assemblyai_transcriber._call_assemblyai")
def test_empty_words_raises_runtime(mock_call, tmp_path):
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"")
    mock_call.return_value = _make_transcript([])

    from backend.stt.assemblyai_transcriber import transcribe_with_speaker
    with pytest.raises(RuntimeError, match="결과 없음"):
        transcribe_with_speaker(audio)
