"""Ollama LLM 감정 추론 모듈 — requests로 /api/chat 호출, 단어별 감정 태깅."""

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

VALID_EMOTIONS = {"joy", "sadness", "anger", "neutral"}
FALLBACK_EMOTION = "neutral"

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a per-word emotion tagger for movie dialogue.\n"
    "Your ONLY job is to assign an emotion to each word.\n\n"
    "Valid emotions: joy, sadness, anger, neutral\n\n"
    "Return ONLY a JSON array, no markdown, no explanation.\n"
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


def tag_emotions(word_data: list[dict]) -> list[dict]:
    """단어 리스트를 문장으로 묶고 LLM으로 감정·화자를 태깅한다.

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

    llm_results = _call_llm_with_retry(sentences)

    merged = _merge_results(sentences, llm_results)
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
    lines = ["Tag the emotion of every word in the following sentences:", ""]
    for i, words in enumerate(sentences, start=1):
        text = " ".join(w["word"] for w in words)
        t_start = words[0]["timestamp_start"]
        t_end = words[-1]["timestamp_end"]
        lines.append(f"Sentence {i} ({t_start:.1f}s-{t_end:.1f}s): '{text}'")
    lines.extend([
        "",
        "Return JSON array with sentence_id, text, and words (with emotion per word) for each sentence.",
    ])
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
    # 1. 마크다운 코드블록 제거
    text = re.sub(r"```(?:json)?\s*([\s\S]*?)```", r"\1", raw).strip()

    # 2. trailing comma 제거
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)

    # 3. JSON 배열 범위 추출
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("응답에서 JSON 배열을 찾을 수 없음")

    return text[start : end + 1]


def _parse_json_response(raw: str) -> list[dict]:
    """LLM 응답 문자열을 전처리한 뒤 JSON 배열로 파싱한다."""
    return json.loads(_clean_json_text(raw))


# ---------------------------------------------------------------------------
# LLM 결과와 음향 분석 결과 병합
# ---------------------------------------------------------------------------

def _merge_results(
    sentences: list[list[dict]],
    llm_results: list[dict],
) -> list[dict]:
    """LLM emotion·speaker·text와 음향 분석 volume/pitch를 word 단위로 병합한다."""
    output: list[dict] = []

    # zip_longest: LLM이 일부 문장만 반환해도 나머지를 fallback으로 채움
    for sent_idx, (word_list, llm_sent) in enumerate(
        itertools.zip_longest(sentences, llm_results, fillvalue={}), start=1
    ):
        if word_list is None:
            continue  # sentences 쪽이 더 짧은 경우(이론상 불가)는 건너뜀
        llm_words: list[dict] = llm_sent.get("words", [])
        merged_words: list[dict] = []

        for word_idx, audio_word in enumerate(word_list):
            emotion = FALLBACK_EMOTION
            if word_idx < len(llm_words):
                raw_emotion = llm_words[word_idx].get("emotion", FALLBACK_EMOTION)
                emotion = raw_emotion if raw_emotion in VALID_EMOTIONS else FALLBACK_EMOTION

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
