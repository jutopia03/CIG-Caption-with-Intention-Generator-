"""DeepL API를 사용한 문장 단위 배치 번역 모듈."""

import logging
import os

import deepl
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _get_client() -> deepl.Translator:
    api_key = os.getenv("DEEPL_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPL_API_KEY가 .env에 설정되지 않았습니다.")
    return deepl.Translator(api_key)


def translate_sentences(sentences: list[dict], target_lang: str) -> list[dict]:
    """sentence 목록의 text 필드를 DeepL로 배치 번역하여 text_translated 필드로 추가한다.

    Args:
        sentences: sentence_id, speaker, text, words 필드를 가진 dict 목록
        target_lang: DeepL 대상 언어 코드 (예: "KO", "EN-US", "JA")

    Returns:
        각 dict에 text_translated 필드가 추가된 새 목록 (원본 수정 없음)

    Raises:
        RuntimeError: DeepL API 호출 실패 시
    """
    if not sentences:
        return []

    texts = [s["text"] for s in sentences]
    logger.info("DeepL 배치 번역 시작: %d개 문장 → %s", len(texts), target_lang)

    client = _get_client()

    try:
        results = client.translate_text(texts, target_lang=target_lang)
    except deepl.DeepLException as e:
        raise RuntimeError(f"DeepL 번역 실패: {e}") from e
    except Exception as e:
        raise RuntimeError(f"번역 중 예기치 않은 오류 발생: {e}") from e

    translated: list[dict] = []
    for sentence, result in zip(sentences, results):
        entry = dict(sentence)
        entry["text_translated"] = result.text
        translated.append(entry)

    logger.info("DeepL 배치 번역 완료: %d개 문장", len(translated))
    return translated
