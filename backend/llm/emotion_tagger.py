"""Ollama LLM 감정 추론 모듈 — rule-based 1차 필터 + LLM 2차 보완."""

import itertools
import json
import logging
import os
import re
from typing import Any

from dotenv import load_dotenv

load_dotenv()

LLM_HOST: str = os.getenv("LLM_HOST", "http://localhost:11434")
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3.2")

SENTENCE_GAP_SEC: float = 0.8   # 이 간격 이상이면 새 문장
SENTENCE_MAX_WORDS: int = 15    # 이 단어 수 이상이면 강제 분리
SENTENCE_BATCH_SIZE: int = 3    # LLM에 한 번에 보낼 최대 문장 수

VALID_EMOTIONS = {
    "joy", "sadness", "anger", "fear", "surprise", "disgust", "contempt", "neutral"
}
FALLBACK_EMOTION = "neutral"

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a per-word emotion tagger for movie dialogue.\n"
    "Your ONLY job is to assign an emotion to each listed word.\n\n"
    "Valid emotions: joy, sadness, anger, fear, surprise, disgust, contempt, neutral\n\n"
    "Emotion guidelines:\n"
    "- joy: happy, excited, relieved, playful, celebratory expressions\n"
    "- sadness: disappointed, regretful, sorrowful, grieving expressions\n"
    "- anger: frustrated, demanding, aggressive, commanding, threatening tone\n"
    "- fear: terrified, anxious, panicked, dreading tone\n"
    "- surprise: shocked, astonished, disbelieving, unexpected reaction\n"
    "- disgust: revolted, repulsed, strong rejection or disdain\n"
    "- contempt: mocking, sarcastic, condescending, dismissive tone\n"
    "- neutral: informational, calm, matter-of-fact statements\n\n"
    "Use the provided volume and pitch cues to inform your decision:\n"
    "- High volume + high pitch may suggest fear, surprise, or joy\n"
    "- High volume + low pitch may suggest contempt or disgust\n"
    "- Low volume overall may suggest sadness or fear\n\n"
    "Return ONLY a JSON array, no markdown, no explanation.\n"
    "Include only the words listed under 'Words to tag'.\n"
    "Format:\n"
    "[\n"
    "  {\n"
    "    \"sentence_id\": 1,\n"
    "    \"text\": \"full sentence text\",\n"
    "    \"words\": [\n"
    "      {\"word\": \"word\", \"emotion\": \"neutral\"},\n"
    "      ...\n"
    "    ]\n"
    "  }\n"
    "]"
)


# ---------------------------------------------------------------------------
# rule-based 1차 필터
# ---------------------------------------------------------------------------

def _rule_based_emotion(volume: int, pitch: int) -> str | None:
    """volume/pitch 조합으로 1차 감정 분류. 명확하지 않으면 None을 반환한다.

    규칙:
    - vol >= 4 AND pitch >= 4 → anger  (크고 높음: 소리지름/분노)
    - vol >= 4 AND pitch <= 2 → anger  (크고 낮음: 위협/명령)
    - vol <= 2 AND pitch <= 2 → sadness (작고 낮음: 슬픔/우울)
    - vol in [2,3] AND pitch in [2,3,4] → neutral (중간 범위: 차분한 대화)
    """
    if volume >= 4 and pitch >= 4:
        return "anger"
    if volume >= 4 and pitch <= 2:
        return "anger"
    if volume <= 2 and pitch <= 2:
        return "sadness"
    if 2 <= volume <= 3 and 2 <= pitch <= 4:
        return "neutral"
    return None


# ---------------------------------------------------------------------------
# 퍼블릭 인터페이스
# ---------------------------------------------------------------------------

def tag_emotions(word_data: list[dict]) -> list[dict]:
    """단어 리스트를 문장으로 묶고 rule-based 1차 필터 + LLM 2차 보완으로 감정 태깅한다.

    Args:
        word_data: transcribe + analyze 결과
                   (word, timestamp_start, timestamp_end, volume_level, pitch_level)

    Returns:
        Sentence 스키마에 맞는 dict 목록
        (sentence_id, speaker, text, words[emotion, volume_level, pitch_level, ...])
    """
    if not word_data:
        return []

    sentences = _split_into_sentences(word_data)
    logger.info("LLM 감정 추론 시작: 총 %d개 문장", len(sentences))

    # 1단계: rule-based 1차 필터 — 각 단어에 _rule_emotion 주석 추가
    for sentence in sentences:
        for word in sentence:
            word["_rule_emotion"] = _rule_based_emotion(word["volume_level"], word["pitch_level"])

    rule_count = sum(1 for s in sentences for w in s if w["_rule_emotion"] is not None)
    total_words = sum(len(s) for s in sentences)
    logger.info("rule-based 1차 필터: %d/%d 단어 분류 완료", rule_count, total_words)

    # 2단계: rule로 미분류 단어가 있는 문장만 LLM 배치에 포함
    llm_indices = [
        i for i, s in enumerate(sentences)
        if any(w["_rule_emotion"] is None for w in s)
    ]
    llm_sentences = [sentences[i] for i in llm_indices]

    llm_by_orig: dict[int, dict] = {}
    for batch_start in range(0, len(llm_sentences), SENTENCE_BATCH_SIZE):
        batch = llm_sentences[batch_start : batch_start + SENTENCE_BATCH_SIZE]
        batch_results = _call_llm_with_retry(batch)
        for j, result in enumerate(batch_results):
            orig_idx = llm_indices[batch_start + j]
            result["sentence_id"] = orig_idx + 1
            llm_by_orig[orig_idx] = result

    all_llm = [llm_by_orig.get(i, {}) for i in range(len(sentences))]

    merged = _merge_results(sentences, all_llm)
    logger.info("LLM 감정 추론 완료")
    return merged


# ---------------------------------------------------------------------------
# 문장 분리
# ---------------------------------------------------------------------------

def _split_into_sentences(word_data: list[dict]) -> list[list[dict]]:
    """timestamp 간격(0.8s), 단어 수(15개), 또는 화자 변경을 기준으로 문장 단위로 분리한다."""
    sentences: list[list[dict]] = []
    current: list[dict] = [word_data[0]]

    for prev, curr in zip(word_data, word_data[1:]):
        gap = curr["timestamp_start"] - prev["timestamp_end"]
        speaker_changed = curr.get("speaker") != prev.get("speaker")
        if gap >= SENTENCE_GAP_SEC or len(current) >= SENTENCE_MAX_WORDS or speaker_changed:
            sentences.append(current)
            current = []
        current.append(curr)

    if current:
        sentences.append(current)

    return sentences


# ---------------------------------------------------------------------------
# LLM 호출
# ---------------------------------------------------------------------------

def _call_llm_with_retry(sentences: list[list[dict]]) -> list[dict]:
    """LLM 호출 → JSON 파싱 → 실패 시 1회 재시도 → 그래도 실패 시 fallback."""
    user_prompt = _build_user_prompt(sentences)

    for attempt in range(2):
        try:
            raw = _post_to_ollama(user_prompt)
            parsed = _parse_json_response(raw)
            logger.info("LLM 응답 파싱 완료: %d개 문장", len(parsed))
            return parsed
        except (ValueError, json.JSONDecodeError) as e:
            if attempt == 0:
                logger.warning("LLM 응답 파싱 실패, 재시도 중: %s", e)
            else:
                logger.warning("LLM 응답 재시도도 실패, fallback 적용: %s", e)
        except ConnectionError:
            logger.warning("Ollama 서버 연결 실패, fallback 적용")
            break

    return _build_fallback(sentences)


def _build_user_prompt(sentences: list[list[dict]]) -> str:
    """LLM 프롬프트를 생성한다. rule-based로 미분류된 단어만 포함하며 vol/pitch 정보를 함께 전달한다."""
    lines = [
        "Tag the emotion of only the listed words in each sentence.",
        "volume_level: 1=very quiet … 5=very loud  |  pitch_level: 1=very low … 5=very high",
        "",
    ]
    for i, words in enumerate(sentences, start=1):
        text = " ".join(w["word"] for w in words)
        t_start = words[0]["timestamp_start"]
        t_end = words[-1]["timestamp_end"]
        lines.append(f"Sentence {i} ({t_start:.1f}s-{t_end:.1f}s): '{text}'")
        lines.append("Words to tag:")
        for w in words:
            if w.get("_rule_emotion") is None:
                lines.append(
                    f'  - "{w["word"]}" (volume={w["volume_level"]}, pitch={w["pitch_level"]})'
                )
        lines.append("")
    lines.append(
        "Return JSON array with sentence_id, text, and words (only tagged words with emotion) for each sentence."
    )
    return "\n".join(lines)


def _post_to_ollama(user_prompt: str) -> str:
    """requests로 Ollama /api/chat 엔드포인트를 호출하고 content 문자열을 반환한다."""
    import requests  # 런타임 의존성 — 테스트 시 모킹 가능

    url = f"{LLM_HOST}/api/chat"
    payload: dict[str, Any] = {
        "model": LLM_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
    }

    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(f"Ollama 서버에 연결할 수 없습니다: {LLM_HOST}") from e
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Ollama 요청 실패: {e}") from e

    body = resp.json()
    content = body.get("message", {}).get("content")
    if content is None:
        raise ValueError(f"Ollama 응답에 message.content 없음: {body}")
    return content


def _clean_json_text(raw: str) -> str:
    """LLM 응답을 json.loads()에 넣기 전에 전처리한다.

    1. ```json ... ``` 마크다운 코드블록 제거
    2. trailing comma 제거 — llama3.2가 {"key": "val",} 형태를 종종 출력함
    3. 첫 '[' ~ 마지막 ']' 슬라이싱
    """
    text = re.sub(r"```(?:json)?\s*([\s\S]*?)```", r"\1", raw).strip()
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("응답에서 JSON 배열을 찾을 수 없음")

    return text[start : end + 1]


def _parse_json_response(raw: str) -> list[dict]:
    """LLM 응답 문자열을 전처리한 뒤 JSON 배열로 파싱한다."""
    return json.loads(_clean_json_text(raw))


# ---------------------------------------------------------------------------
# LLM 결과와 rule 결과 병합
# ---------------------------------------------------------------------------

def _merge_results(
    sentences: list[list[dict]],
    llm_results: list[dict],
) -> list[dict]:
    """rule emotion과 LLM emotion을 word 단위로 병합한다.

    우선순위: _rule_emotion > LLM emotion (순서 매칭) > FALLBACK_EMOTION
    """
    output: list[dict] = []

    for sent_idx, (word_list, llm_sent) in enumerate(
        itertools.zip_longest(sentences, llm_results, fillvalue={}), start=1
    ):
        if word_list is None:
            continue

        # LLM이 반환한 단어 목록 — rule 미분류 단어와 순서 대응
        llm_words: list[dict] = llm_sent.get("words", [])
        llm_word_idx = 0
        merged_words: list[dict] = []

        for audio_word in word_list:
            rule_emotion = audio_word.get("_rule_emotion")
            if rule_emotion is not None:
                emotion = rule_emotion
            else:
                if llm_word_idx < len(llm_words):
                    raw_emotion = llm_words[llm_word_idx].get("emotion", FALLBACK_EMOTION)
                    emotion = raw_emotion if raw_emotion in VALID_EMOTIONS else FALLBACK_EMOTION
                    llm_word_idx += 1
                else:
                    emotion = FALLBACK_EMOTION

            merged_words.append(
                {
                    "word":            audio_word["word"],
                    "timestamp_start": audio_word["timestamp_start"],
                    "timestamp_end":   audio_word["timestamp_end"],
                    "emotion":         emotion,
                    "volume_level":    audio_word["volume_level"],
                    "pitch_level":     audio_word["pitch_level"],
                }
            )

        sentence_speaker: str = word_list[0].get("speaker", "Character_A")
        output.append(
            {
                "sentence_id": llm_sent.get("sentence_id", sent_idx),
                "speaker":     sentence_speaker,
                "text":        llm_sent.get("text", " ".join(w["word"] for w in word_list)),
                "words":       merged_words,
            }
        )

    return output


def _build_fallback(sentences: list[list[dict]]) -> list[dict]:
    """LLM 완전 실패 시 emotion=neutral, speaker는 첫 단어의 speaker 키에서 가져온다."""
    return [
        {
            "sentence_id": i,
            "speaker":     words[0].get("speaker", "Character_A"),
            "text":        " ".join(w["word"] for w in words),
            "words":       [
                {**w, "emotion": FALLBACK_EMOTION}
                for w in words
            ],
        }
        for i, words in enumerate(sentences, start=1)
    ]
