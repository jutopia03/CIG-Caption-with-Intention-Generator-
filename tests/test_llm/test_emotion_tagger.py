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

# rule 필터 결과:
#   "You"   vol=3 pitch=2 → neutral  (rule)
#   "know"  vol=3 pitch=1 → None     (LLM 필요)
#   "where" vol=2 pitch=3 → neutral  (rule)
#   "this"  vol=4 pitch=4 → anger    (rule)
#   "ends"  vol=3 pitch=2 → neutral  (rule)
# → LLM에는 "know" 한 단어만 전달

_LLM_RESPONSE = json.dumps([
    {
        "sentence_id": 1,
        "text": "You know where this ends",
        "words": [
            {"word": "know", "emotion": "sadness"},
        ],
    }
])


def _mock_post(url, json=None, timeout=None):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"message": {"content": _LLM_RESPONSE}}
    return resp


# ---------------------------------------------------------------------------
# 기존 테스트 (구조/동작 검증)
# ---------------------------------------------------------------------------

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
    """emotion은 확장된 VALID_EMOTIONS 중 하나여야 한다."""
    from backend.llm.emotion_tagger import tag_emotions, VALID_EMOTIONS
    result = tag_emotions(_WORD_DATA)
    for sent in result:
        for word in sent["words"]:
            assert word["emotion"] in VALID_EMOTIONS


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
    """Ollama 연결 실패 시 rule-based 단어는 rule 감정 유지, 미분류 단어는 neutral로 폴백해야 한다."""
    from backend.llm.emotion_tagger import tag_emotions
    result = tag_emotions(_WORD_DATA)
    assert isinstance(result, list)
    assert len(result) > 0
    for sent in result:
        assert sent["speaker"] == "Character_A"

    # "this"(vol=4, pitch=4) → rule-based anger → LLM 실패해도 anger 유지
    # "know"(vol=3, pitch=1) → rule 미분류 → neutral fallback
    word_emotions = {w["word"]: w["emotion"] for sent in result for w in sent["words"]}
    assert word_emotions["this"] == "anger"
    assert word_emotions["know"] == "neutral"


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


# ---------------------------------------------------------------------------
# 새 테스트: rule-based 1차 필터
# ---------------------------------------------------------------------------

def test_rule_based_emotion_anger_loud_high():
    """vol>=4, pitch>=4 → anger (소리지름/분노)."""
    from backend.llm.emotion_tagger import _rule_based_emotion
    assert _rule_based_emotion(4, 4) == "anger"
    assert _rule_based_emotion(5, 5) == "anger"
    assert _rule_based_emotion(4, 5) == "anger"


def test_rule_based_emotion_anger_loud_low():
    """vol>=4, pitch<=2 → anger (위협/명령)."""
    from backend.llm.emotion_tagger import _rule_based_emotion
    assert _rule_based_emotion(4, 1) == "anger"
    assert _rule_based_emotion(5, 2) == "anger"


def test_rule_based_emotion_sadness_quiet_low():
    """vol<=2, pitch<=2 → sadness (슬픔/우울)."""
    from backend.llm.emotion_tagger import _rule_based_emotion
    assert _rule_based_emotion(1, 1) == "sadness"
    assert _rule_based_emotion(2, 2) == "sadness"


def test_rule_based_emotion_neutral_middle():
    """중간 범위 → neutral (차분한 대화)."""
    from backend.llm.emotion_tagger import _rule_based_emotion
    assert _rule_based_emotion(3, 3) == "neutral"
    assert _rule_based_emotion(2, 4) == "neutral"
    assert _rule_based_emotion(3, 2) == "neutral"


def test_rule_based_emotion_none():
    """규칙 미적용 케이스 → None (LLM으로 넘어감)."""
    from backend.llm.emotion_tagger import _rule_based_emotion
    assert _rule_based_emotion(3, 1) is None   # vol=3(중간), pitch=1(낮음) → 어떤 규칙도 미적용
    assert _rule_based_emotion(1, 3) is None   # vol=1(작음), pitch=3(중간) → sadness 규칙 미적용
    assert _rule_based_emotion(4, 3) is None   # vol>=4, pitch=3(중간) → anger 규칙 미적용


# ---------------------------------------------------------------------------
# 새 테스트: LLM 호출 최적화
# ---------------------------------------------------------------------------

def test_rule_filter_skips_llm_for_all_classified():
    """모든 단어가 rule-based로 분류된 문장은 LLM을 호출하지 않아야 한다."""
    all_rule_data = [
        {"word": "Stop", "timestamp_start": 0.0, "timestamp_end": 0.5,
         "volume_level": 5, "pitch_level": 5, "speaker": "Character_A"},
        {"word": "now",  "timestamp_start": 0.5, "timestamp_end": 1.0,
         "volume_level": 5, "pitch_level": 5, "speaker": "Character_A"},
    ]
    from backend.llm.emotion_tagger import tag_emotions
    with patch("backend.llm.emotion_tagger._post_to_ollama") as mock_llm:
        result = tag_emotions(all_rule_data)
        mock_llm.assert_not_called()
    assert result[0]["words"][0]["emotion"] == "anger"
    assert result[0]["words"][1]["emotion"] == "anger"


@patch("backend.llm.emotion_tagger._post_to_ollama", return_value=_LLM_RESPONSE)
def test_rule_filter_applies_correct_emotions(mock_post):
    """rule-based 분류된 단어는 rule 감정, LLM 단어는 LLM 감정이 적용되어야 한다."""
    from backend.llm.emotion_tagger import tag_emotions
    result = tag_emotions(_WORD_DATA)
    flat = {w["word"]: w["emotion"] for sent in result for w in sent["words"]}
    assert flat["You"]   == "neutral"   # rule: vol=3, pitch=2
    assert flat["know"]  == "sadness"   # LLM 결과
    assert flat["where"] == "neutral"   # rule: vol=2, pitch=3
    assert flat["this"]  == "anger"     # rule: vol=4, pitch=4
    assert flat["ends"]  == "neutral"   # rule: vol=3, pitch=2


# ---------------------------------------------------------------------------
# 새 테스트: LLM 프롬프트 품질
# ---------------------------------------------------------------------------

def test_llm_prompt_includes_volume_pitch():
    """LLM 프롬프트에 volume_level, pitch_level 정보가 포함되어야 한다."""
    from backend.llm.emotion_tagger import _build_user_prompt
    sentences = [
        [
            {"word": "why", "timestamp_start": 0.0, "timestamp_end": 0.5,
             "volume_level": 4, "pitch_level": 3, "_rule_emotion": None},
        ]
    ]
    prompt = _build_user_prompt(sentences)
    assert "volume=4" in prompt
    assert "pitch=3" in prompt


def test_llm_prompt_excludes_rule_classified_words():
    """rule-based로 분류된 단어는 LLM 프롬프트에 포함되지 않아야 한다."""
    from backend.llm.emotion_tagger import _build_user_prompt
    sentences = [
        [
            {"word": "Hello", "timestamp_start": 0.0, "timestamp_end": 0.5,
             "volume_level": 3, "pitch_level": 3, "_rule_emotion": "neutral"},
            {"word": "why",   "timestamp_start": 0.5, "timestamp_end": 1.0,
             "volume_level": 4, "pitch_level": 3, "_rule_emotion": None},
        ]
    ]
    prompt = _build_user_prompt(sentences)
    assert '"why"' in prompt
    assert '"Hello"' not in prompt


# ---------------------------------------------------------------------------
# 새 테스트: 감정 카테고리 확장
# ---------------------------------------------------------------------------

def test_valid_emotions_expanded():
    """VALID_EMOTIONS에 확장된 8개 감정 카테고리가 모두 포함되어야 한다."""
    from backend.llm.emotion_tagger import VALID_EMOTIONS
    expected = {"joy", "sadness", "anger", "fear", "surprise", "disgust", "contempt", "neutral"}
    assert expected == VALID_EMOTIONS


def test_new_emotions_accepted_from_llm():
    """LLM이 새 감정(fear, surprise 등)을 반환해도 그대로 사용되어야 한다."""
    llm_resp = json.dumps([
        {
            "sentence_id": 1,
            "text": "Oh no",
            "words": [{"word": "Oh", "emotion": "fear"}, {"word": "no", "emotion": "surprise"}],
        }
    ])
    word_data = [
        {"word": "Oh", "timestamp_start": 0.0, "timestamp_end": 0.3,
         "volume_level": 3, "pitch_level": 1},  # pitch=1 → rule None
        {"word": "no", "timestamp_start": 0.3, "timestamp_end": 0.6,
         "volume_level": 3, "pitch_level": 1},  # pitch=1 → rule None
    ]
    from backend.llm.emotion_tagger import tag_emotions
    with patch("backend.llm.emotion_tagger._post_to_ollama", return_value=llm_resp):
        result = tag_emotions(word_data)
    emotions = [w["emotion"] for sent in result for w in sent["words"]]
    assert "fear" in emotions
    assert "surprise" in emotions
