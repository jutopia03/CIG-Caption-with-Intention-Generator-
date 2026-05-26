"""Librosa 음향 분석 모듈 — RMS 볼륨과 pyin 피치를 5단계로 정규화."""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def analyze(audio_path: str | Path, word_timestamps: list[dict]) -> list[dict]:
    """각 단어 구간의 RMS 볼륨과 pitch를 추출하여 1–5 단계로 정규화한다.

    Args:
        audio_path: 분석할 오디오 파일 경로
        word_timestamps: transcribe() 반환값 (word, start, end 포함)

    Returns:
        volume_level, pitch_level 필드가 추가된 단어 목록
    """
    # TODO: librosa.load(audio_path, sr=None)로 오디오 및 샘플레이트 로드
    # TODO: 각 단어 구간(start~end)을 샘플 인덱스로 변환
    # TODO: librosa.feature.rms()로 구간별 평균 RMS 에너지 계산
    # TODO: librosa.pyin()으로 구간별 F0(기본 주파수) 추출
    # TODO: 전체 발화 기준 min-max 정규화 → ceil(value * 5) 로 1-5 변환
    # TODO: word_timestamps 각 항목에 volume_level, pitch_level 필드 추가 후 반환
    raise NotImplementedError
