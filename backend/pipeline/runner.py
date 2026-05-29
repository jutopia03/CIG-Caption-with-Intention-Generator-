"""파이프라인 실행 모듈 — AssemblyAI STT+화자분리 → 음향 분석 → 감정 태깅을 결합하여 최종 JSON 생성."""

import json
import logging
import os
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

from backend.audio.analyzer import analyze
from backend.llm.emotion_tagger import tag_emotions
from backend.schemas.word_schema import Sentence
from backend.stt.assemblyai_transcriber import transcribe_with_speaker

load_dotenv()

OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "data/outputs"))

logger = logging.getLogger(__name__)


def run(video_path: str | Path) -> list[dict]:
    """영상 파일을 받아 AssemblyAI STT+화자분리 → 음향 분석 → 감정 태깅을 순서대로 실행한다.

    Args:
        video_path: 처리할 영상 파일 경로

    Returns:
        Sentence 스키마에 맞는 dict 목록

    Raises:
        FileNotFoundError: video_path가 존재하지 않을 때
        RuntimeError: 각 단계 실패 시 (어느 단계인지 메시지에 포함)
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")

    logger.info("파이프라인 시작: %s", video_path)
    pipeline_start = time.time()

    tmp_audio: Path | None = None
    try:
        try:
            tmp_audio = _extract_audio(video_path)
        except Exception as e:
            raise RuntimeError(f"파이프라인 실패 [오디오 추출 단계]: {e}") from e
        logger.info("오디오 추출 완료")

        # 1단계: AssemblyAI STT + 화자 분리
        t0 = time.time()
        try:
            word_timestamps = transcribe_with_speaker(str(tmp_audio))
        except Exception as e:
            raise RuntimeError(f"파이프라인 실패 [AssemblyAI STT+화자분리 단계]: {e}") from e
        logger.info("AssemblyAI STT+화자분리 완료: %.1f초", time.time() - t0)

        # 2단계: 음향 분석
        t0 = time.time()
        try:
            analyzed_words = analyze(str(tmp_audio), word_timestamps)
        except Exception as e:
            raise RuntimeError(f"파이프라인 실패 [음향 분석 단계]: {e}") from e
        logger.info("음향 분석 완료: %.1f초", time.time() - t0)

        # 3단계: LLM 감정 추론
        t0 = time.time()
        try:
            final_result = tag_emotions(analyzed_words)
        except Exception as e:
            raise RuntimeError(f"파이프라인 실패 [LLM 감정 추론 단계]: {e}") from e
        logger.info("LLM 감정 추론 완료: %.1f초", time.time() - t0)

    finally:
        if tmp_audio is not None and tmp_audio.exists():
            tmp_audio.unlink()

    # Pydantic 검증 — 스키마 위반 시 경고 후 raw dict 반환
    validated = _validate(final_result)

    logger.info("파이프라인 완료: 총 %d개 문장, 전체 소요 %.1f초", len(validated), time.time() - pipeline_start)
    return validated


def save_result(result: list[dict], output_path: str) -> None:
    """결과를 JSON 파일로 저장한다.

    Args:
        result: run()의 반환값
        output_path: 저장할 파일 경로 (부모 디렉토리가 없으면 자동 생성)
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("결과 저장 완료: %s", path)


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

def _extract_audio(video_path: Path) -> Path:
    """ffmpeg로 영상에서 16kHz mono WAV를 임시 파일로 추출한다."""
    import ffmpeg  # 런타임 의존성 — 테스트 시 모킹 가능

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    tmp_path = Path(tmp.name)

    try:
        (
            ffmpeg
            .input(str(video_path))
            .output(str(tmp_path), ar=16000, ac=1, format="wav", **{"y": None})
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        tmp_path.unlink(missing_ok=True)
        detail = e.stderr.decode(errors="replace") if e.stderr else str(e)
        raise RuntimeError(f"오디오 추출 실패: {detail}") from e

    return tmp_path


def _validate(result: list[dict]) -> list[dict]:
    """Pydantic으로 최종 결과를 검증하고 직렬화 가능한 dict로 반환한다.

    검증 실패 시 경고를 로깅하고 원본 dict를 그대로 반환한다.
    """
    validated: list[dict] = []
    for item in result:
        try:
            validated.append(Sentence.model_validate(item).model_dump())
        except Exception as e:
            logger.warning("스키마 검증 실패 (sentence_id=%s): %s", item.get("sentence_id"), e)
            validated.append(item)
    return validated


# ---------------------------------------------------------------------------
# CLI 진입점
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if len(sys.argv) < 2:
        print("사용법: python -m backend.pipeline.runner <video_path>")
        sys.exit(1)

    _video_path = sys.argv[1]
    _result = run(_video_path)

    _output_path = f"data/outputs/{Path(_video_path).stem}_result.json"
    save_result(_result, _output_path)
    print(f"결과 저장 완료: {_output_path}")
