"""backend.translation.translator 유닛 테스트."""

from unittest.mock import MagicMock, patch

import pytest


# ── 헬퍼 ────────────────────────────────────────────────────────────────────

def _make_mock_client(translated_texts: list[str]) -> MagicMock:
    """translate_text() 결과를 반환하는 DeepL Translator mock."""
    client = MagicMock()
    client.translate_text.return_value = [MagicMock(text=t) for t in translated_texts]
    return client


# ── 픽스처 ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_sentences() -> list[dict]:
    return [
        {
            "sentence_id": 1,
            "speaker": "Character_A",
            "text": "Hello world",
            "words": [
                {"word": "Hello", "timestamp_start": 0.0, "timestamp_end": 0.5,
                 "emotion": "neutral", "volume_level": 3, "pitch_level": 3},
                {"word": "world", "timestamp_start": 0.5, "timestamp_end": 1.0,
                 "emotion": "joy", "volume_level": 2, "pitch_level": 2},
            ],
        },
        {
            "sentence_id": 2,
            "speaker": "Character_B",
            "text": "How are you",
            "words": [
                {"word": "How", "timestamp_start": 1.2, "timestamp_end": 1.5,
                 "emotion": "neutral", "volume_level": 2, "pitch_level": 2},
            ],
        },
    ]


# ── 테스트 ───────────────────────────────────────────────────────────────────

@patch("backend.translation.translator._get_client")
def test_text_translated_field_added(mock_get_client: MagicMock, sample_sentences: list[dict]) -> None:
    """번역 결과 dict에 text_translated 필드가 추가되는지 확인."""
    mock_get_client.return_value = _make_mock_client(["안녕 세상", "잘 지내?"])

    from backend.translation.translator import translate_sentences
    result = translate_sentences(sample_sentences, target_lang="KO")

    assert result[0]["text_translated"] == "안녕 세상"
    assert result[1]["text_translated"] == "잘 지내?"


@patch("backend.translation.translator._get_client")
def test_words_array_unchanged(mock_get_client: MagicMock, sample_sentences: list[dict]) -> None:
    """words 배열이 번역 전후로 변경되지 않는지 확인."""
    original_words = [[dict(w) for w in s["words"]] for s in sample_sentences]
    mock_get_client.return_value = _make_mock_client(["안녕 세상", "잘 지내?"])

    from backend.translation.translator import translate_sentences
    result = translate_sentences(sample_sentences, target_lang="KO")

    for i, entry in enumerate(result):
        assert entry["words"] == original_words[i]


@patch("backend.translation.translator._get_client")
def test_original_schema_preserved(mock_get_client: MagicMock, sample_sentences: list[dict]) -> None:
    """sentence_id, speaker, text, words 원본 필드가 결과에 그대로 유지되는지 확인."""
    mock_get_client.return_value = _make_mock_client(["안녕 세상", "잘 지내?"])

    from backend.translation.translator import translate_sentences
    result = translate_sentences(sample_sentences, target_lang="KO")

    for original, translated in zip(sample_sentences, result):
        assert translated["sentence_id"] == original["sentence_id"]
        assert translated["speaker"]     == original["speaker"]
        assert translated["text"]        == original["text"]
        assert translated["words"]       == original["words"]


@patch("backend.translation.translator._get_client")
def test_original_sentences_not_mutated(mock_get_client: MagicMock, sample_sentences: list[dict]) -> None:
    """translate_sentences()가 원본 dict를 수정하지 않고 새 dict를 반환하는지 확인."""
    original_keys = set(sample_sentences[0].keys())
    mock_get_client.return_value = _make_mock_client(["안녕 세상", "잘 지내?"])

    from backend.translation.translator import translate_sentences
    translate_sentences(sample_sentences, target_lang="KO")

    assert "text_translated" not in sample_sentences[0]
    assert set(sample_sentences[0].keys()) == original_keys


def test_empty_list_returns_empty() -> None:
    """빈 리스트 입력 시 API 호출 없이 빈 리스트를 반환하는지 확인."""
    from backend.translation.translator import translate_sentences
    result = translate_sentences([], target_lang="KO")
    assert result == []


def test_missing_api_key_raises_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """DEEPL_API_KEY 미설정 시 RuntimeError가 발생하는지 확인."""
    monkeypatch.delenv("DEEPL_API_KEY", raising=False)

    from backend.translation.translator import translate_sentences
    with pytest.raises(RuntimeError, match="DEEPL_API_KEY"):
        translate_sentences(
            [{"sentence_id": 1, "speaker": "Character_A", "text": "hi", "words": []}],
            target_lang="KO",
        )
