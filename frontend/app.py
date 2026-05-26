"""Gradio UI 진입점 — 영상 업로드 → 파이프라인 실행 → 가변 자막 렌더링."""

import gradio as gr

from backend.pipeline.runner import run

# emotion → RobotoFlex font-weight 매핑
EMOTION_WEIGHT: dict[str, int] = {
    "joy": 700,
    "anger": 900,
    "sadness": 300,
    "neutral": 400,
}


def process_video(video_path: str) -> str:
    """Gradio 이벤트 핸들러 — 업로드된 영상을 파이프라인에 넘기고 HTML 자막을 반환한다.

    Args:
        video_path: Gradio가 전달하는 임시 파일 경로

    Returns:
        RobotoFlex 가변폰트 기반 HTML 자막 문자열
    """
    # TODO: run(video_path)로 Sentence 목록 획득
    # TODO: emotion → font-weight, volume_level → font-size, pitch_level → font-style 매핑
    # TODO: 각 단어를 <span style="font-variation-settings: ..."> 태그로 감싸 HTML 조립
    # TODO: 문장 단위 <p> 블록으로 구성 후 반환
    raise NotImplementedError


def build_ui() -> gr.Blocks:
    """Gradio Blocks UI를 빌드하고 반환한다."""
    # TODO: gr.Blocks(theme=gr.themes.Soft()) 레이아웃 구성
    # TODO: RobotoFlex Google Fonts <link> 태그를 head에 주입
    # TODO: gr.Video 업로드 컴포넌트 추가
    # TODO: "자막 생성" 버튼 → process_video 연결
    # TODO: gr.HTML 결과 컴포넌트 추가
    raise NotImplementedError


if __name__ == "__main__":
    ui = build_ui()
    ui.launch()
