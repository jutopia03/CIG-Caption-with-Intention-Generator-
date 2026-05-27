"""Librosa 음향 분석 모듈 — RMS 볼륨과 pyin 피치를 5단계로 정규화."""

import logging
import math
from pathlib import Path

import numpy as np

SR = 16000          # Whisper와 동일한 샘플레이트
HOP_LENGTH = 512    # librosa pyin 기본값 — frame_times 계산에 사용
logger = logging.getLogger(__name__)


def analyze(audio_path: str | Path, word_timestamps: list[dict]) -> list[dict]:
    """각 단어 구간의 RMS 볼륨과 피치를 추출해 1–5 단계로 정규화한다.

    Args:
        audio_path: 분석할 오디오 파일 경로 (16kHz mono WAV 권장)
        word_timestamps: transcribe() 반환값 (word, timestamp_start, timestamp_end)

    Returns:
        volume_level, pitch_level 필드가 추가된 단어 목록

    Raises:
        FileNotFoundError: audio_path가 존재하지 않을 때
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {audio_path}")

    if not word_timestamps:
        return []

    logger.info("음향 분석 시작: 총 %d개 단어", len(word_timestamps))

    import librosa  # 런타임 의존성 — 테스트 시 모킹 가능

    y, _ = librosa.load(str(audio_path), sr=SR, mono=True)

    # 전체 오디오에서 pyin 한 번만 실행 — 단어 구간별 개별 실행보다 유효 프레임 확보량이 높음
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=SR,
        hop_length=HOP_LENGTH,
    )
    frame_times = librosa.frames_to_time(
        np.arange(len(f0)), sr=SR, hop_length=HOP_LENGTH
    )

    raw_rms, raw_pitch = _extract_raw_features(
        y, word_timestamps, f0, voiced_flag, frame_times
    )

    volume_levels = _normalize_to_levels(raw_rms)
    pitch_levels = _normalize_to_levels(raw_pitch)

    rms_min = min(raw_rms)
    rms_max = max(raw_rms)
    logger.info("음향 분석 완료: 볼륨 범위 %.4f~%.4f", rms_min, rms_max)

    result: list[dict] = []
    for word_info, vol, pit in zip(word_timestamps, volume_levels, pitch_levels):
        result.append({**word_info, "volume_level": vol, "pitch_level": pit})

    return result


def _extract_raw_features(
    y: np.ndarray,
    word_timestamps: list[dict],
    f0: np.ndarray,
    voiced_flag: np.ndarray,
    frame_times: np.ndarray,
) -> tuple[list[float], list[float]]:
    """단어별 RMS 볼륨과 pyin 피치 원시값을 추출한다.

    피치는 전체 오디오에서 한 번 실행한 pyin 결과에서 단어 구간 프레임을
    슬라이싱하므로, 단어 구간이 짧아도 유효 프레임이 충분히 확보된다.
    유효 피치 프레임이 없는 단어는 float('nan') 반환 → 정규화 시 level=3 폴백.
    """
    import librosa  # 런타임 의존성 — 테스트 시 모킹 가능

    total_samples = len(y)
    raw_rms: list[float] = []
    raw_pitch: list[float] = []

    for word_info in word_timestamps:
        ts = word_info["timestamp_start"]
        te = word_info["timestamp_end"]
        start_idx = int(ts * SR)
        end_idx = int(te * SR)

        # 구간이 너무 짧거나 범위를 벗어난 경우 기본값
        if end_idx <= start_idx or start_idx >= total_samples:
            raw_rms.append(0.0)
            raw_pitch.append(float('nan'))
            continue

        end_idx = min(end_idx, total_samples)
        segment = y[start_idx:end_idx]

        # RMS 볼륨 (구간별 계산)
        rms_frames = librosa.feature.rms(y=segment)
        raw_rms.append(float(np.mean(rms_frames)))

        # pyin 결과에서 단어 구간 프레임 슬라이싱 (voiced 프레임만)
        mask = (frame_times >= ts) & (frame_times <= te) & voiced_flag
        word_f0 = f0[mask]
        valid_f0 = word_f0[~np.isnan(word_f0)] if len(word_f0) > 0 else np.array([])

        raw_pitch.append(float(np.mean(valid_f0)) if len(valid_f0) > 0 else float('nan'))

    return raw_rms, raw_pitch


def _normalize_to_levels(values: list[float]) -> list[int]:
    """float 값 목록을 전체 기준 min-max 정규화로 1~5 정수로 변환한다.

    0.0 (RMS 무음 구간) 또는 NaN (피치 미검출) 은 3으로 처리한다.
    min==max인 경우(전체가 동일한 값) 전부 3으로 반환한다.
    """
    valid = [v for v in values if v > 0 and not math.isnan(v)]

    if not valid:
        return [3] * len(values)

    v_min = min(valid)
    v_max = max(valid)

    levels: list[int] = []
    for v in values:
        if v <= 0 or math.isnan(v) or v_min == v_max:
            levels.append(3)
        else:
            level = round(1 + (v - v_min) / (v_max - v_min) * 4)
            levels.append(max(1, min(5, level)))

    return levels
