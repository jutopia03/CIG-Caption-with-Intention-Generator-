"""backend.pipeline.runner 유닛 테스트."""

import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# 테스트용 픽스처 데이터
# ---------------------------------------------------------------------------

_WORD_TIMESTAMPS = [
    {"word": "You",  "timestamp_start": 0.70, "timestamp_end": 1.14},
    {"word": "know", "timestamp_start": 1.14, "timestamp_end": 1.28},
]

_ANALYZED_WORDS = [
    {"word": "You",  "timestamp_start": 0.70, "timestamp_end": 1.14, "volume_level": 3, "pitch_level": 2},
    {"word": "know", "timestamp_start": 1.14, "timestamp_end": 1.28, "volume_level": 3, "pitch_level": 1},
]

_FINAL_RESULT = [
    {
        "sentence_id": 1,
        "speaker": "Character_A",
        "text": "You know",
        "words": [
            {"word": "You",  "timestamp_start": 0.70, "timestamp_end": 1.14,
             "emotion": "neutral", "volume_level": 3, "pitch_level": 2},
            {"word": "know", "timestamp_start": 1.14, "timestamp_end": 1.28,
             "emotion": "sadness", "volume_level": 3, "pitch_level": 1},
        ],
    }
]


def _fake_tmp_audio(tmp_path: Path) -> MagicMock:
    """실제로 존재하는 파일을 가리키는 Path mock을 반환한다."""
    real = tmp_path / "audio.wav"
    real.touch()
    m = MagicMock(spec=Path)
    m.exists.return_value = True
    m.unlink.return_value = None
    m.__str__ = lambda self: str(real)
    return m


# ---------------------------------------------------------------------------
# 헬퍼: 세 단계를 모두 mock한 채로 run() 호출
# ---------------------------------------------------------------------------

def _run_with_mocks(tmp_path: Path):
    fake_audio = _fake_tmp_audio(tmp_path)

    with (
        patch("backend.pipeline.runner._extract_audio", return_value=fake_audio),
        patch("backend.pipeline.runner.transcribe_with_speaker",    return_value=_WORD_TIMESTAMPS),
        patch("backend.pipeline.runner.analyze",       return_value=_ANALYZED_WORDS),
        patch("backend.pipeline.runner.tag_emotions",  return_value=_FINAL_RESULT),
    ):
        fake_video = tmp_path / "video.mp4"
        fake_video.touch()

        from backend.pipeline.runner import run
        return run(str(fake_video))


# ---------------------------------------------------------------------------
# 테스트
# ---------------------------------------------------------------------------

def test_run_returns_list(tmp_path):
    """run()의 반환값이 list여야 한다."""
    result = _run_with_mocks(tmp_path)
    assert isinstance(result, list)


def test_run_file_not_found():
    """존재하지 않는 경로를 넘기면 FileNotFoundError가 발생해야 한다."""
    from backend.pipeline.runner import run

    with pytest.raises(FileNotFoundError):
        run("/nonexistent/path/video.mp4")


def test_run_pipeline_order(tmp_path):
    """transcribe → analyze → tag_emotions 순서로 호출되어야 한다."""
    call_order: list[str] = []

    def fake_transcribe(*a, **kw):
        call_order.append("transcribe_with_speaker")
        return _WORD_TIMESTAMPS

    def fake_analyze(*a, **kw):
        call_order.append("analyze")
        return _ANALYZED_WORDS

    def fake_tag(*a, **kw):
        call_order.append("tag_emotions")
        return _FINAL_RESULT

    fake_audio = _fake_tmp_audio(tmp_path)
    fake_video = tmp_path / "video.mp4"
    fake_video.touch()

    with (
        patch("backend.pipeline.runner._extract_audio", return_value=fake_audio),
        patch("backend.pipeline.runner.transcribe_with_speaker",    side_effect=fake_transcribe),
        patch("backend.pipeline.runner.analyze",       side_effect=fake_analyze),
        patch("backend.pipeline.runner.tag_emotions",  side_effect=fake_tag),
    ):
        from backend.pipeline.runner import run
        run(str(fake_video))

    assert call_order == ["transcribe_with_speaker", "analyze", "tag_emotions"]


def test_run_tmp_audio_deleted_on_success(tmp_path):
    """정상 완료 후 임시 오디오 파일이 삭제되어야 한다."""
    fake_audio = _fake_tmp_audio(tmp_path)
    fake_video = tmp_path / "video.mp4"
    fake_video.touch()

    with (
        patch("backend.pipeline.runner._extract_audio", return_value=fake_audio),
        patch("backend.pipeline.runner.transcribe_with_speaker",    return_value=_WORD_TIMESTAMPS),
        patch("backend.pipeline.runner.analyze",       return_value=_ANALYZED_WORDS),
        patch("backend.pipeline.runner.tag_emotions",  return_value=_FINAL_RESULT),
    ):
        from backend.pipeline.runner import run
        run(str(fake_video))

    fake_audio.unlink.assert_called_once()


def test_run_tmp_audio_deleted_on_failure(tmp_path):
    """STT 실패 시에도 임시 오디오 파일이 삭제되어야 한다."""
    fake_audio = _fake_tmp_audio(tmp_path)
    fake_video = tmp_path / "video.mp4"
    fake_video.touch()

    with (
        patch("backend.pipeline.runner._extract_audio", return_value=fake_audio),
        patch("backend.pipeline.runner.transcribe_with_speaker", side_effect=RuntimeError("STT 오류")),
    ):
        from backend.pipeline.runner import run

        with pytest.raises(RuntimeError, match="STT"):
            run(str(fake_video))

    fake_audio.unlink.assert_called_once()


def test_save_result(tmp_path):
    """save_result()가 올바른 JSON 파일을 생성해야 한다."""
    from backend.pipeline.runner import save_result

    output_path = str(tmp_path / "sub" / "result.json")
    save_result(_FINAL_RESULT, output_path)

    saved = json.loads(Path(output_path).read_text(encoding="utf-8"))
    assert saved == _FINAL_RESULT
    assert saved[0]["sentence_id"] == 1
    assert saved[0]["speaker"] == "Character_A"
