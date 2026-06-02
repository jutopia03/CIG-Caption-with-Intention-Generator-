"""Gradio UI 진입점 — 영상 업로드 → 파이프라인 실행 → 가변 자막 렌더링."""

import json
import logging
import os
import shutil
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from backend.pipeline.runner import run, save_result

load_dotenv()

OUTPUT_DIR  = Path(os.getenv("OUTPUT_DIR", "data/outputs"))
SAMPLES_DIR = Path("data/samples")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JS — demo.load()로 실행 (Gradio의 <script> 차단 우회)
# ---------------------------------------------------------------------------

_SUBTITLE_JS = """
function() {
  let initialized      = false;
  let lastDataJson     = null;
  let syncInterval     = null;
  let fullscreenHandler = null;

  // ── 팀원2 연계 애니메이션 컨트롤러 ──────────────────────────────────────
  window.cigAnimation = window.cigAnimation || {
    enabled:   true,
    intensity: 'low',
    setEnabled(on) {
      this.enabled = on;
      this._apply();
    },
    setIntensity(level) {
      this.intensity = level;
      this._apply();
    },
    _apply() {
      const ov = document.getElementById('subtitle-overlay');
      if (!ov) return;
      ov.classList.toggle('cig-anim-off', !this.enabled);
      if (this.intensity === 'low') {
        ov.removeAttribute('data-intensity');
      } else {
        ov.setAttribute('data-intensity', this.intensity);
      }
    },
  };
  // ─────────────────────────────────────────────────────────────────────────

  function initSubtitles() {
    if (initialized) return;

    const dataEl  = document.getElementById('cig-data');
    const video   = document.getElementById('cig-video');
    const overlay = document.getElementById('subtitle-overlay');

    if (!dataEl || !video || !overlay) return;

    initialized = true;

    if (syncInterval) { clearInterval(syncInterval); syncInterval = null; }

    const CIG_DATA = JSON.parse(dataEl.dataset.json);

    const SPEAKER_COLORS = ['#FFFFFF','#FFD700','#00CED1','#FF69B4','#98FB98','#DDA0DD'];
    const speakers = [...new Set(CIG_DATA.map(s => s.speaker))];
    const speakerColorMap = {};
    speakers.forEach((sp, i) => { speakerColorMap[sp] = SPEAKER_COLORS[i % 6]; });

    // 팀원3 카테고리 확장 시 이 맵 + CSS 클래스 한 쌍만 추가하면 됨
    const EMOTION_CLASS = {
      joy:      'cig-emotion-joy',
      sadness:  'cig-emotion-sadness',
      anger:    'cig-emotion-anger',
      fear:     'cig-emotion-fear',
      surprise: 'cig-emotion-surprise',
      disgust:  'cig-emotion-disgust',
      contempt: 'cig-emotion-contempt',
      neutral:  '',
    };

    const VOLUME_STYLE_NORMAL = {
      1: {fontSize:'16px', fontWeight:'400'},
      2: {fontSize:'20px', fontWeight:'500'},
      3: {fontSize:'26px', fontWeight:'600'},
      4: {fontSize:'32px', fontWeight:'700'},
      5: {fontSize:'40px', fontWeight:'900'},
    };

    const VOLUME_STYLE_FULLSCREEN = {
      1: {fontSize:'24px', fontWeight:'400'},
      2: {fontSize:'30px', fontWeight:'500'},
      3: {fontSize:'38px', fontWeight:'600'},
      4: {fontSize:'46px', fontWeight:'700'},
      5: {fontSize:'56px', fontWeight:'900'},
    };

    let VOLUME_STYLE  = VOLUME_STYLE_NORMAL;
    let BASE_FONT_SIZE = '16px';

    let lastSentenceId = -1;

    function syncSubtitles() {
      const t = video.currentTime;
      let currentSentence = null;

      for (const sentence of CIG_DATA) {
        const words = sentence.words;
        const start = words[0].timestamp_start;
        const end   = words[words.length - 1].timestamp_end;
        if (t >= start - 0.3 && t <= end + 0.3) { currentSentence = sentence; break; }
      }

      if (!currentSentence) { overlay.innerHTML = ''; lastSentenceId = -1; return; }

      if (currentSentence.sentence_id !== lastSentenceId) {
        lastSentenceId = currentSentence.sentence_id;
        const color = speakerColorMap[currentSentence.speaker] || '#FFFFFF';
        let html = '';
        for (const word of currentSentence.words) {
          html += '<span'
               + ' data-start="'   + word.timestamp_start        + '"'
               + ' data-end="'     + word.timestamp_end          + '"'
               + ' data-volume="'  + word.volume_level           + '"'
               + ' data-emotion="' + (word.emotion || 'neutral') + '"'
               + ' style="display:inline-block; margin:0 4px; color:' + color
               + '; font-size:' + BASE_FONT_SIZE + '; font-weight:400; opacity:0.6;'
               + ' transition:font-size 0.08s ease, opacity 0.08s ease;">'
               + word.word + '</span>';
        }
        overlay.innerHTML = html;
      }

      overlay.querySelectorAll('span').forEach(span => {
        const s   = parseFloat(span.dataset.start);
        const e   = parseFloat(span.dataset.end);
        const vol = parseInt(span.dataset.volume) || 3;
        const emoClass = EMOTION_CLASS[span.dataset.emotion] || '';
        if (t >= s && t <= e) {
          const st = VOLUME_STYLE[vol] || VOLUME_STYLE[3];
          span.style.fontSize   = st.fontSize;
          span.style.fontWeight = st.fontWeight;
          span.style.opacity    = '1.0';
          if (emoClass) span.classList.add(emoClass);
        } else {
          span.style.fontSize = BASE_FONT_SIZE; span.style.fontWeight = '400'; span.style.opacity = '0.6';
          if (emoClass) span.classList.remove(emoClass);
        }
      });
    }

    syncInterval = setInterval(syncSubtitles, 100);

    // 전체화면 토글 헬퍼
    const container = document.getElementById('cig-container');
    function toggleFullscreen() {
      if (!document.fullscreenElement) {
        container.requestFullscreen();
      } else {
        document.exitFullscreen();
      }
    }

    // 더블클릭 + 커스텀 버튼 모두 컨테이너 전체화면으로
    video.addEventListener('dblclick', toggleFullscreen);
    const fsBtn = document.getElementById('fullscreen-btn');
    if (fsBtn) { fsBtn.addEventListener('click', toggleFullscreen); }

    // 전체화면 진입/해제 시 video 높이만 조정 (exit/request 연속 호출 없음)
    if (fullscreenHandler) {
      document.removeEventListener('fullscreenchange', fullscreenHandler);
    }
    fullscreenHandler = () => {
      if (document.fullscreenElement === container) {
        video.style.height    = '100vh';
        video.style.maxHeight = '100vh';
        VOLUME_STYLE   = VOLUME_STYLE_FULLSCREEN;
        BASE_FONT_SIZE = '24px';
      } else {
        video.style.height    = '';
        video.style.maxHeight = '500px';
        VOLUME_STYLE   = VOLUME_STYLE_NORMAL;
        BASE_FONT_SIZE = '16px';
      }
      // 스타일 전환 즉시 반영 — 현재 문장 span 강제 재생성
      lastSentenceId = -1;
    };
    document.addEventListener('fullscreenchange', fullscreenHandler);

    window.cigAnimation._apply();
    console.log('CIG 자막 초기화 완료:', CIG_DATA.length, '개 문장');
  }

  // 300ms 폴링: cig-data의 data-json이 바뀌면 재초기화
  setInterval(() => {
    const dataEl = document.getElementById('cig-data');
    if (dataEl && dataEl.dataset.json !== lastDataJson) {
      lastDataJson = dataEl.dataset.json;
      initialized  = false;
    }
    initSubtitles();
  }, 300);
}
"""


# ---------------------------------------------------------------------------
# HTML 생성 — <script> 없이 data 속성만 사용
# ---------------------------------------------------------------------------

def generate_subtitle_html(result: list[dict], filename: str) -> str:
    """파이프라인 결과와 파일명을 받아 비디오+자막 오버레이 HTML을 반환한다.

    <script> 태그 없이 JSON을 data-json 속성에 심어두고,
    JS는 demo.load()(_SUBTITLE_JS)가 300ms 폴링으로 감지해 실행한다.

    Args:
        result:   runner.run() 반환값 (Sentence 스키마 기준)
        filename: data/samples/ 아래 복사된 영상 파일명 (예: test_clip.mp4)

    Returns:
        video 태그 + 자막 오버레이 + 숨겨진 JSON 데이터 div를 포함한 HTML
    """
    json_data         = json.dumps(result, ensure_ascii=False)
    json_data_escaped = json_data.replace("'", "&#39;")

    return (
        '<link href="https://fonts.googleapis.com/css2?family=Roboto+Flex'
        ':opsz,wght@8..144,100..900&display=swap" rel="stylesheet">'
        '<style>'
        '#cig-container { position:relative; width:100%; background:#000; }'
        '#cig-container:fullscreen #subtitle-overlay,'
        '#cig-container:-webkit-full-screen #subtitle-overlay,'
        '#cig-container:-moz-full-screen #subtitle-overlay {'
        '  position:fixed; bottom:60px; z-index:999999;'
        '}'
        '#subtitle-overlay{'
        '--cig-joy-bright:1.10;--cig-joy-bounce:2px;--cig-joy-dur:0.5s;'
        '--cig-sad-sat:0.60;--cig-sad-dur:0.9s;'
        '--cig-anger-shake:1px;--cig-anger-dur:0.18s;'
        '--cig-fear-amp:0.8px;--cig-fear-dur:0.12s;'
        '--cig-surprise-scale:1.15;--cig-surprise-dur:0.35s;'
        '--cig-disgust-skew:3deg;--cig-disgust-sat:0.55;--cig-disgust-dur:0.5s;'
        '--cig-contempt-rot:3deg;--cig-contempt-dur:0.6s}'
        '#subtitle-overlay[data-intensity="medium"]{'
        '--cig-joy-bright:1.15;--cig-joy-bounce:4px;--cig-joy-dur:0.45s;'
        '--cig-sad-sat:0.45;--cig-sad-dur:0.75s;'
        '--cig-anger-shake:2px;--cig-anger-dur:0.15s;'
        '--cig-fear-amp:1.2px;--cig-fear-dur:0.10s;'
        '--cig-surprise-scale:1.20;--cig-surprise-dur:0.30s;'
        '--cig-disgust-skew:5deg;--cig-disgust-sat:0.45;--cig-disgust-dur:0.45s;'
        '--cig-contempt-rot:4deg;--cig-contempt-dur:0.5s}'
        '#subtitle-overlay[data-intensity="high"]{'
        '--cig-joy-bright:1.25;--cig-joy-bounce:6px;--cig-joy-dur:0.4s;'
        '--cig-sad-sat:0.30;--cig-sad-dur:0.6s;'
        '--cig-anger-shake:3px;--cig-anger-dur:0.12s;'
        '--cig-fear-amp:1.8px;--cig-fear-dur:0.08s;'
        '--cig-surprise-scale:1.28;--cig-surprise-dur:0.25s;'
        '--cig-disgust-skew:7deg;--cig-disgust-sat:0.35;--cig-disgust-dur:0.4s;'
        '--cig-contempt-rot:5deg;--cig-contempt-dur:0.4s}'
        '#subtitle-overlay.cig-anim-off span{'
        'animation:none!important;filter:none!important;transform:none!important}'
        '@keyframes cig-joy-bounce{'
        '0%,100%{transform:translateY(0)}'
        '40%{transform:translateY(calc(-1*var(--cig-joy-bounce)))}'
        '70%{transform:translateY(calc(-0.4*var(--cig-joy-bounce)))}}'
        '@keyframes cig-sad-fade{'
        '0%,100%{filter:saturate(var(--cig-sad-sat)) brightness(1)}'
        '50%{filter:saturate(var(--cig-sad-sat)) brightness(0.82)}}'
        '@keyframes cig-anger-shake{'
        '0%,100%{transform:translateX(0)}'
        '25%{transform:translateX(calc(-1*var(--cig-anger-shake)))}'
        '75%{transform:translateX(var(--cig-anger-shake))}}'
        '.cig-emotion-joy{'
        'animation:cig-joy-bounce var(--cig-joy-dur) ease-in-out infinite;'
        'filter:brightness(var(--cig-joy-bright))}'
        '.cig-emotion-sadness{'
        'filter:saturate(var(--cig-sad-sat));'
        'animation:cig-sad-fade var(--cig-sad-dur) ease-in-out infinite}'
        '.cig-emotion-anger{'
        'animation:cig-anger-shake var(--cig-anger-dur) ease-in-out infinite}'
        '@keyframes cig-fear-tremble{'
        '0%,100%{transform:translate(0,0)}'
        '25%{transform:translate(calc(-1*var(--cig-fear-amp)),calc(0.5*var(--cig-fear-amp)))}'
        '75%{transform:translate(var(--cig-fear-amp),calc(-0.5*var(--cig-fear-amp)))}}'
        '@keyframes cig-surprise-pop{'
        '0%{transform:scale(1)}'
        '30%{transform:scale(var(--cig-surprise-scale))}'
        '60%{transform:scale(0.97)}'
        '100%{transform:scale(1)}}'
        '@keyframes cig-disgust-skew{'
        '0%,100%{transform:skewX(0deg)}'
        '30%{transform:skewX(calc(-1*var(--cig-disgust-skew)))}'
        '70%{transform:skewX(var(--cig-disgust-skew))}}'
        '@keyframes cig-contempt-tilt{'
        '0%,100%{transform:rotate(0deg)}'
        '50%{transform:rotate(var(--cig-contempt-rot))}}'
        '.cig-emotion-fear{'
        'animation:cig-fear-tremble var(--cig-fear-dur) linear infinite}'
        '.cig-emotion-surprise{'
        'animation:cig-surprise-pop var(--cig-surprise-dur) ease-out forwards}'
        '.cig-emotion-disgust{'
        'filter:saturate(var(--cig-disgust-sat));'
        'animation:cig-disgust-skew var(--cig-disgust-dur) ease-in-out infinite}'
        '.cig-emotion-contempt{'
        'animation:cig-contempt-tilt var(--cig-contempt-dur) ease-in-out infinite}'
        '@media(prefers-reduced-motion:reduce){'
        '#subtitle-overlay span{animation:none!important}}'
        '</style>'
        '<div id="cig-container">'
        '<video id="cig-video"'
        ' src="/gradio_api/file=data/samples/' + filename + '"'
        ' controls'
        ' controlsList="nofullscreen"'
        ' style="width:100%; display:block; max-height:500px;"></video>'
        '<div id="subtitle-overlay" style="'
        'position:absolute; bottom:60px; left:0; right:0; text-align:center;'
        'padding:4px 20px; pointer-events:none;'
        "font-family:'Roboto Flex', sans-serif;"
        'text-shadow: 1px 1px 3px #000, -1px -1px 3px #000;'
        '"></div>'
        '<button id="fullscreen-btn" style="'
        'position:absolute; bottom:10px; right:10px;'
        'background:rgba(0,0,0,0.6); border:none; color:white;'
        'padding:6px 10px; border-radius:4px; cursor:pointer;'
        'font-size:16px; z-index:10;">'
        '&#x26F6;</button>'
        "<div id=\"cig-data\" data-json='"
        + json_data_escaped
        + "' style=\"display:none;\"></div>"
        '</div>'
    )


# ---------------------------------------------------------------------------
# Gradio 이벤트 핸들러
# ---------------------------------------------------------------------------

def process_video(video_file: object) -> tuple[str, str]:
    """자막 생성 버튼 클릭 시 호출되는 핸들러.

    Args:
        video_file: gr.File()이 반환하는 객체.
                    Gradio 버전에 따라 str / NamedString / dict 형태.

    Returns:
        (player_html, status_text)
    """
    # Gradio 버전별 반환 타입 정규화
    src_path: str | None = None
    if isinstance(video_file, dict):
        src_path = video_file.get("name") or video_file.get("path")
    elif hasattr(video_file, "name"):       # NamedString (Gradio 4.x)
        src_path = str(video_file.name)
    elif isinstance(video_file, str):
        src_path = video_file

    if not src_path:
        return "", "영상 파일을 먼저 업로드해주세요."

    try:
        # 업로드 파일을 프로젝트 내 data/samples/ 로 복사 (Gradio static 서빙)
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        filename  = os.path.basename(src_path)
        dst_path  = str(SAMPLES_DIR / filename)
        shutil.copy2(src_path, dst_path)
        logger.info("UI: 영상 복사 완료 %s → %s", src_path, dst_path)

        logger.info("UI: 파이프라인 시작 — %s", dst_path)

        result = run(dst_path)

        stem        = Path(dst_path).stem
        output_path = OUTPUT_DIR / f"{stem}_result.json"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        save_result(result, str(output_path))

        html        = generate_subtitle_html(result, filename)
        n_sentences = len(result)
        n_words     = sum(len(s["words"]) for s in result)
        status      = f"완료: {n_sentences}개 문장, {n_words}개 단어 | 저장: {output_path}"
        logger.info("UI: %s", status)
        return html, status

    except FileNotFoundError as e:
        return "", f"오류: {e}"
    except RuntimeError as e:
        return "", f"파이프라인 오류: {e}"
    except OSError as e:
        return "", f"파일 오류: {e}"
    except Exception as e:
        logger.exception("UI: 예상치 못한 오류")
        return "", f"예상치 못한 오류: {e}"


# ---------------------------------------------------------------------------
# UI 빌드
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    """Gradio Blocks UI를 빌드하고 반환한다."""
    with gr.Blocks(title="CIG — Caption with Intention Generator") as demo:
        gr.Markdown("# CIG — Caption with Intention Generator")
        gr.Markdown(
            "영상을 업로드하고 **자막 생성**을 누르면 "
            "볼륨·화자별 배리어프리 가변 자막을 생성합니다."
        )

        video_file = gr.File(
            label="영상 업로드",
            file_types=[".mp4", ".mkv", ".avi"],
        )

        generate_btn = gr.Button("자막 생성", variant="primary")

        status_box = gr.Textbox(
            label="상태",
            interactive=False,
            placeholder="영상을 업로드한 뒤 '자막 생성'을 클릭하세요.",
        )

        # video 태그 + 자막 오버레이를 모두 담는 단일 HTML 컴포넌트
        player_html = gr.HTML()

        generate_btn.click(
            fn=process_video,
            inputs=[video_file],
            outputs=[player_html, status_box],
        )

        # Gradio의 <script> 차단을 우회해 자막 JS를 안전하게 등록
        demo.load(fn=None, js=_SUBTITLE_JS)

    return demo


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    demo = build_ui()
    demo.launch(
        server_port=int(os.getenv("GRADIO_PORT", "7860")),
        allowed_paths=[str(SAMPLES_DIR)],
    )
