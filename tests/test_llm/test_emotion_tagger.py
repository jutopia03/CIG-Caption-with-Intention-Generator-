"""backend.llm.emotion_tagger 유닛 테스트."""

import json
from unittest.mock import MagicMock, patch

import pytest

_WORD_DATA = [
    {"word": "You",   "timestamp_start": 0.70, "timestamp_end": 1.14, "volume_level": 3, "pitch_level": 2},
    {"word": "know",  "timestamp_start": 1.14, "timestamp_end": 1.28, "volume_level": 3, "pitch_level": 1},
    {"word": "where", "timestamp_start": 1.30, "timestamp_end": 1.60, "volume_level": 2, "pitch_level": 3},
    {"word": "this",  "timestamp_start": 1.65, "timestamp_end": 1.85, "volume_level": 4, "pitch_level": 4},
    {"word": "ends",  "timestamp_start": 1.90, "timestamp_end": 2.20, "volume_level": 3, "pitch_level": 2},
]

_LLM_RESPONSE = json.dumps([
    {
        "sentence_id": 1,
        "speaker": "Character_A",
        "text": "You know where this ends",
        "words": [
            {"word": "You",   "emotion": "neutral"},
            {"word": "know",  "emotion": "sadness"},
            {"word": "where", "emotion": "neutral"},
            {"word": "this",  "emotion": "anger"},
            {"word": "ends",  "emotion": "sadness"},
        ],
    }
])


def _mock_post(url, json=None, timeout=None):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"message": {"content": _LLM_RESPONSE}}
    return resp


@patch("backend.llm.emotion_tagger._post_to_ollama", return_value=_LLM_RESPONSE)
def test_tag_emotions_returns_list(mock_post):
    """tag_emotions()의 반환값이 list여야 한다."""
    from backend.llm.emotion_tagger import tag_emotions
    result = tag_emotions(_WORD_DATA)
    assert isinstance(result, list)


@patch("backend.llm.emotion_tagger._post_to_ollama", return_value=_LLM_RESPONSE)
def test_tag_emotions_sentence_structure(mock_post):
    """각 문장에 sentence_id, speaker, text, words 키가 있어야 한다."""
    from backend.llm.emotion_tagger import tag_emotions
    result = tag_emotions(_WORD_DATA)
    for sent in result:
        assert "sentence_id" in sent
        assert "speaker" in sent
        assert "text" in sent
        assert "words" in sent


@patch("backend.llm.emotion_tagger._post_to_ollama", return_value=_LLM_RESPONSE)
def test_tag_emotions_word_emotion(mock_post):
    """각 word에 emotion 키가 있어야 한다."""
    from backend.llm.emotion_tagger import tag_emotions
    result = tag_emotions(_WORD_DATA)
    for sent in result:
        for word in sent["words"]:
            assert "emotion" in word


@patch("backend.llm.emotion_tagger._post_to_ollama", return_value=_LLM_RESPONSE)
def test_tag_emotions_emotion_values(mock_post):
    """emotion은 joy|sadness|anger|neutral 중 하나여야 한다."""
    from backend.llm.emotion_tagger import tag_emotions
    valid = {"joy", "sadness", "anger", "neutral"}
    result = tag_emotions(_WORD_DATA)
    for sent in result:
        for word in sent["words"]:
            assert word["emotion"] in valid


@patch("backend.llm.emotion_tagger._post_to_ollama", return_value=_LLM_RESPONSE)
def test_tag_emotions_preserves_levels(mock_post):
    """volume_level, pitch_level은 음향 분석 결과 그대로 유지되어야 한다."""
    from backend.llm.emotion_tagger import tag_emotions
    result = tag_emotions(_WORD_DATA)
    flat_result = [w for sent in result for w in sent["words"]]
    for orig, merged in zip(_WORD_DATA, flat_result):
        assert merged["volume_level"] == orig["volume_level"]
        assert merged["pitch_level"]  == orig["pitch_level"]


@patch("backend.llm.emotion_tagger._post_to_ollama", side_effect=ConnectionError("연결 실패"))
def test_tag_emotions_fallback(mock_post):
    """Ollama 연결 실패 시 emotion=neutral, speaker=Character_A 로 폴백해야 한다."""
    from backend.llm.emotion_tagger import tag_emotions
    result = tag_emotions(_WORD_DATA)
    assert isinstance(result, list)
    assert len(result) > 0
    for sent in result:
        assert sent["speaker"] == "Character_A"
        for word in sent["words"]:
            assert word["emotion"] == "neutral"


@patch("backend.llm.emotion_tagger._post_to_ollama", return_value="```json\n" + _LLM_RESPONSE + "\n```")
def test_tag_emotions_markdown_stripped(mock_post):
    """```json 마크다운 블록이 감싸진 응답도 정상 파싱되어야 한다."""
    from backend.llm.emotion_tagger import tag_emotions
    result = tag_emotions(_WORD_DATA)
    assert isinstance(result, list)
    assert len(result) > 0


def test_clean_json_text_trailing_comma_object():
    """객체 닫기 전 trailing comma가 제거되어야 한다."""
    from backend.llm.emotion_tagger import _clean_json_text
    raw = '[{"word": "You", "emotion": "neutral",}]'
    result = _clean_json_text(raw)
    import json
    parsed = json.loads(result)
    assert parsed[0]["emotion"] == "neutral"


def test_clean_json_text_trailing_comma_array():
    """배열 닫기 전 trailing comma가 제거되어야 한다."""
    from backend.llm.emotion_tagger import _clean_json_text
    raw = '[{"sentence_id": 1, "words": [{"word": "You", "emotion": "joy"},],},]'
    result = _clean_json_text(raw)
    import json
    parsed = json.loads(result)
    assert parsed[0]["words"][0]["emotion"] == "joy"


def test_clean_json_text_markdown_and_trailing_comma():
    """마크다운 + trailing comma 복합 케이스도 처리되어야 한다."""
    from backend.llm.emotion_tagger import _clean_json_text
    raw = '```json\n[{"word": "hi", "emotion": "sadness",}]\n```'
    import json
    parsed = json.loads(_clean_json_text(raw))
    assert parsed[0]["emotion"] == "sadness"
