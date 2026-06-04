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

  window.cigAnimation = window.cigAnimation || {
    enabled: true,
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

  window.cigSubtitleSettings = window.cigSubtitleSettings || {
    fontScale: 1,
    position: 'bottom',
    opacity: 1,
    animationEnabled: window.cigAnimation.enabled,
    animationIntensity: window.cigAnimation.intensity,
    apply() {
      const overlay = document.getElementById('subtitle-overlay');
      if (overlay) {
        overlay.style.opacity = String(this.opacity);
        overlay.style.top = this.position === 'top' ? '60px' : '';
        overlay.style.bottom = this.position === 'bottom' ? '60px' : '';
      }
      if (window.cigAnimation) {
        window.cigAnimation.setEnabled(this.animationEnabled);
        window.cigAnimation.setIntensity(this.animationIntensity);
      }
    },
  };

  function scaledPx(px, scale) {
    return Math.round(px * scale) + 'px';
  }

  function bindCustomizationPanel() {
    const settings = window.cigSubtitleSettings;
    const scaleInput = document.getElementById('cig-font-scale');
    const scaleValue = document.getElementById('cig-font-scale-value');
    const positionInputs = document.querySelectorAll('input[name="cig-subtitle-position"]');
    const animationToggle = document.getElementById('cig-animation-enabled');
    const animationIntensity = document.getElementById('cig-animation-intensity');
    const opacityInput = document.getElementById('cig-subtitle-opacity');
    const opacityValue = document.getElementById('cig-subtitle-opacity-value');

    if (!scaleInput || scaleInput.dataset.bound === 'true') return;
    scaleInput.dataset.bound = 'true';

    function syncControls() {
      scaleInput.value = String(settings.fontScale);
      scaleValue.textContent = Math.round(settings.fontScale * 100) + '%';
      positionInputs.forEach(input => {
        input.checked = input.value === settings.position;
      });
      animationToggle.checked = settings.animationEnabled;
      animationIntensity.value = settings.animationIntensity;
      opacityInput.value = String(settings.opacity);
      opacityValue.textContent = Math.round(settings.opacity * 100) + '%';
      settings.apply();
    }

    scaleInput.addEventListener('input', () => {
      settings.fontScale = parseFloat(scaleInput.value) || 1;
      scaleValue.textContent = Math.round(settings.fontScale * 100) + '%';
    });
    positionInputs.forEach(input => {
      input.addEventListener('change', () => {
        if (input.checked) {
          settings.position = input.value;
          settings.apply();
        }
      });
    });
    animationToggle.addEventListener('change', () => {
      settings.animationEnabled = animationToggle.checked;
      settings.apply();
    });
    animationIntensity.addEventListener('change', () => {
      settings.animationIntensity = animationIntensity.value;
      settings.apply();
    });
    opacityInput.addEventListener('input', () => {
      settings.opacity = parseFloat(opacityInput.value) || 1;
      opacityValue.textContent = Math.round(settings.opacity * 100) + '%';
      settings.apply();
    });

    syncControls();
  }

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
      1: {fontSize:16, fontWeight:'400'},
      2: {fontSize:20, fontWeight:'500'},
      3: {fontSize:26, fontWeight:'600'},
      4: {fontSize:32, fontWeight:'700'},
      5: {fontSize:40, fontWeight:'900'},
    };

    const VOLUME_STYLE_FULLSCREEN = {
      1: {fontSize:24, fontWeight:'400'},
      2: {fontSize:30, fontWeight:'500'},
      3: {fontSize:38, fontWeight:'600'},
      4: {fontSize:46, fontWeight:'700'},
      5: {fontSize:56, fontWeight:'900'},
    };

    let VOLUME_STYLE  = VOLUME_STYLE_NORMAL;
    let BASE_FONT_SIZE = 16;

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
        if (currentSentence.text_translated) {
          // 번역 모드: 문장 전체를 하나의 span으로 표시
          const words = currentSentence.words;
          const emotionCounts = {};
          for (const w of words) { const em = w.emotion || 'neutral'; emotionCounts[em] = (emotionCounts[em] || 0) + 1; }
          const dominantEmotion = Object.entries(emotionCounts).sort((a, b) => b[1] - a[1])[0][0];
          const emoClass = EMOTION_CLASS[dominantEmotion] || '';
          const avgVol = Math.round(words.reduce((s, w) => s + (w.volume_level || 3), 0) / words.length);
          html += '<span'
               + (emoClass ? ' class="' + emoClass + '"' : '')
               + ' data-start="'   + words[0].timestamp_start              + '"'
               + ' data-end="'     + words[words.length - 1].timestamp_end + '"'
               + ' data-volume="'  + avgVol                                + '"'
               + ' data-emotion="' + dominantEmotion                       + '"'
               + ' style="display:inline-block; margin:0 4px; color:' + color
               + '; font-size:' + scaledPx(BASE_FONT_SIZE, window.cigSubtitleSettings.fontScale)
               + '; font-weight:400; opacity:1.0;'
               + ' transition:font-size 0.08s ease, opacity 0.08s ease;">'
               + currentSentence.text_translated + '</span>';
        } else {
          // 원문 모드: 단어별 개별 span
          for (const word of currentSentence.words) {
            const emoClass = EMOTION_CLASS[word.emotion] || '';
            html += '<span'
                 + (emoClass ? ' class="' + emoClass + '"' : '')
                 + ' data-start="'   + word.timestamp_start        + '"'
                 + ' data-end="'     + word.timestamp_end          + '"'
                 + ' data-volume="'  + word.volume_level           + '"'
                 + ' data-emotion="' + (word.emotion || 'neutral') + '"'
                 + ' style="display:inline-block; margin:0 4px; color:' + color
                 + '; font-size:' + scaledPx(BASE_FONT_SIZE, window.cigSubtitleSettings.fontScale)
                 + '; font-weight:400; opacity:0.6;'
                 + ' transition:font-size 0.08s ease, opacity 0.08s ease;">'
                 + word.word + '</span>';
          }
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
          span.style.fontSize   = scaledPx(st.fontSize, window.cigSubtitleSettings.fontScale);
          span.style.fontWeight = st.fontWeight;
          span.style.opacity    = '1.0';
        } else {
          span.style.fontSize = scaledPx(BASE_FONT_SIZE, window.cigSubtitleSettings.fontScale);
          span.style.fontWeight = '400';
          span.style.opacity = '0.6';
        }
        if (emoClass) {
          if (window.cigAnimation.enabled) {
            span.classList.add(emoClass);
          } else {
            span.classList.remove(emoClass);
          }
        }
      });
    }

    bindCustomizationPanel();
    window.cigSubtitleSettings.apply();
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
      const settingsPanel = document.getElementById('cig-settings-panel');
      if (document.fullscreenElement === container) {
        video.style.height    = '100vh';
        video.style.maxHeight = '100vh';
        if (settingsPanel) { settingsPanel.style.display = 'none'; }
        VOLUME_STYLE   = VOLUME_STYLE_FULLSCREEN;
        BASE_FONT_SIZE = 24;
      } else {
        video.style.height    = '';
        video.style.maxHeight = '500px';
        if (settingsPanel) { settingsPanel.style.display = ''; }
        VOLUME_STYLE   = VOLUME_STYLE_NORMAL;
        BASE_FONT_SIZE = 16;
      }
      // 스타일 전환 즉시 반영 — 현재 문장 span 강제 재생성
      lastSentenceId = -1;
      window.cigSubtitleSettings.apply();
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
        '.cig-player-shell { display:flex; gap:16px; align-items:flex-start; width:100%; }'
        '.cig-player-main { flex:1 1 auto; min-width:0; }'
        '#cig-container { position:relative; width:100%; background:#000; }'
        '#cig-container:fullscreen #subtitle-overlay,'
        '#cig-container:-webkit-full-screen #subtitle-overlay,'
        '#cig-container:-moz-full-screen #subtitle-overlay {'
        '  position:fixed; bottom:60px; z-index:999999;'
        '}'
        '.cig-player-shell:fullscreen .cig-settings-panel,'
        '.cig-player-shell:-webkit-full-screen .cig-settings-panel,'
        '.cig-player-shell:-moz-full-screen .cig-settings-panel{display:none!important;}'
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
        'animation:cig-surprise-pop var(--cig-surprise-dur) ease-out forwards/* 의도적 1회 실행: 놀람은 순간적 반응 */}'
        '.cig-emotion-disgust{'
        'filter:saturate(var(--cig-disgust-sat));'
        'animation:cig-disgust-skew var(--cig-disgust-dur) ease-in-out infinite}'
        '.cig-emotion-contempt{'
        'animation:cig-contempt-tilt var(--cig-contempt-dur) ease-in-out infinite}'
        '.cig-settings-panel{'
        'flex:0 0 260px; border:1px solid #d7dbe3; border-radius:8px;'
        'background:#f8fafc; color:#111827; padding:14px;'
        "font-family:'Roboto Flex', sans-serif; box-sizing:border-box;}"
        '.cig-settings-panel,.cig-settings-panel h3,.cig-settings-panel label,'
        '.cig-settings-panel legend,.cig-settings-panel span,'
        '.cig-settings-panel select,.cig-settings-panel option{color:#111827!important;}'
        '.cig-settings-panel h3{margin:0 0 12px; font-size:16px; font-weight:700;}'
        '.cig-control{margin-bottom:14px;}'
        '.cig-control label,.cig-control legend{'
        'display:block; margin-bottom:6px; font-size:13px; font-weight:650;}'
        '.cig-control-row{display:flex; align-items:center; justify-content:space-between; gap:10px;}'
        '.cig-control input[type="range"]{width:100%;}'
        '.cig-value{font-size:12px; color:#475569; min-width:44px; text-align:right;}'
        '.cig-segment{display:flex; border:1px solid #cbd5e1; border-radius:6px; overflow:hidden;}'
        '.cig-segment label{flex:1; margin:0; padding:7px 8px; text-align:center; cursor:pointer;}'
        '.cig-segment input{position:absolute; opacity:0; pointer-events:none;}'
        '.cig-segment label:has(input:checked){background:#1f2937; color:#fff!important;}'
        '.cig-segment label:has(input:checked) span{color:#fff!important;}'
        '.cig-control select{background:#fff!important;}'
        '.cig-toggle{display:flex!important; align-items:center; justify-content:space-between; gap:12px;}'
        '.cig-toggle input{'
        'appearance:none; -webkit-appearance:none; width:18px; height:18px;'
        'border:2px solid #111827; border-radius:4px; background:#fff;'
        'cursor:pointer; box-sizing:border-box; position:relative;}'
        '.cig-toggle input:checked{background:#f97316; border-color:#f97316;}'
        '.cig-toggle input:checked::after{'
        'content:""; position:absolute; left:4px; top:1px; width:5px; height:9px;'
        'border:solid #fff; border-width:0 2px 2px 0; transform:rotate(45deg);}'
        '.cig-control select{width:100%; padding:7px 8px; border:1px solid #cbd5e1; border-radius:6px;}'
        '@media(prefers-reduced-motion:reduce){'
        '#subtitle-overlay span{animation:none!important}}'
        '@media(max-width: 900px){.cig-player-shell{flex-direction:column}.cig-settings-panel{width:100%; flex-basis:auto}}'
        '</style>'
        '<div class="cig-player-shell">'
        '<div class="cig-player-main">'
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
        '</div>'
        '<aside id="cig-settings-panel" class="cig-settings-panel">'
        '<h3>자막 설정</h3>'
        '<div class="cig-control">'
        '<label for="cig-font-scale">폰트 크기 기준값</label>'
        '<div class="cig-control-row">'
        '<input id="cig-font-scale" type="range" min="0.75" max="1.5" step="0.05" value="1">'
        '<span id="cig-font-scale-value" class="cig-value">100%</span>'
        '</div>'
        '</div>'
        '<fieldset class="cig-control" style="border:0; padding:0;">'
        '<legend>자막 위치</legend>'
        '<div class="cig-segment">'
        '<label><input type="radio" name="cig-subtitle-position" value="top">상단</label>'
        '<label><input type="radio" name="cig-subtitle-position" value="bottom" checked>하단</label>'
        '</div>'
        '</fieldset>'
        '<div class="cig-control">'
        '<label class="cig-toggle" for="cig-animation-enabled">'
        '<span>감정 애니메이션</span>'
        '<input id="cig-animation-enabled" type="checkbox" checked>'
        '</label>'
        '</div>'
        '<div class="cig-control">'
        '<label for="cig-animation-intensity">애니메이션 강도</label>'
        '<select id="cig-animation-intensity">'
        '<option value="low">낮음</option>'
        '<option value="medium">중간</option>'
        '<option value="high">높음</option>'
        '</select>'
        '</div>'
        '<div class="cig-control" style="margin-bottom:0;">'
        '<label for="cig-subtitle-opacity">자막 투명도</label>'
        '<div class="cig-control-row">'
        '<input id="cig-subtitle-opacity" type="range" min="0.35" max="1" step="0.05" value="1">'
        '<span id="cig-subtitle-opacity-value" class="cig-value">100%</span>'
        '</div>'
        '</div>'
        '</aside>'
        '</div>'
    )


# ---------------------------------------------------------------------------
# Gradio 이벤트 핸들러
# ---------------------------------------------------------------------------

_LANG_MAP: dict[str, str] = {"영어": "en", "한국어": "ko"}


def process_video(
    video_file: object,
    input_lang: str,
    output_lang: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str]:
    """자막 생성 버튼 클릭 시 호출되는 핸들러.

    Args:
        video_file: gr.File()이 반환하는 객체.
                    Gradio 버전에 따라 str / NamedString / dict 형태.
        input_lang: 입력 언어 라디오 값 ("영어" | "한국어")
        output_lang: 출력 언어 라디오 값 ("영어" | "한국어")

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

        def update_progress(
            percent: int,
            message: str,
            current: int,
            total: int,
        ) -> None:
            progress(
                percent / 100,
                desc=f"⏳ {message} ({current}/{total} 단계)",
            )

        result = run(
            dst_path,
            input_lang=_LANG_MAP[input_lang],
            output_lang=_LANG_MAP[output_lang],
            progress_callback=update_progress,
        )

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
    with gr.Blocks(
        title="CIG — Caption with Intention Generator",
        css="""
        .cig-spinner {
            display: inline-block;
            width: 12px;
            height: 12px;
            margin-right: 6px;
            border: 2px solid rgba(255, 255, 255, 0.35);
            border-top-color: #ffffff;
            border-radius: 50%;
            animation: cig-spin 0.8s linear infinite;
            vertical-align: -2px;
        }

        @keyframes cig-spin {
            to {
                transform: rotate(360deg);
            }
        }
        """,
    ) as demo:
        gr.Markdown("# CIG — Caption with Intention Generator")
        gr.Markdown(
            "영상을 업로드하고 **자막 생성**을 누르면 "
            "볼륨·화자별 배리어프리 가변 자막을 생성합니다."
        )

        video_file = gr.File(
            label="영상 업로드",
            file_types=[".mp4", ".mkv", ".avi"],
        )

        input_lang_radio = gr.Radio(
            ["영어", "한국어"],
            value="영어",
            label="입력 언어",
        )
        output_lang_radio = gr.Radio(
            ["영어", "한국어"],
            value="영어",
            label="출력 언어",
        )

        generate_btn = gr.Button("자막 생성", variant="primary")

        status_box = gr.Textbox(
            label="상태",
            interactive=False,
            placeholder="영상을 업로드한 뒤 '자막 생성'을 클릭하세요.",
        )

        player_html = gr.HTML()

        generate_btn.click(
            fn=process_video,
            inputs=[video_file, input_lang_radio, output_lang_radio],
            outputs=[player_html, status_box],
            show_progress="minimal",
            show_progress_on=[player_html],
        )

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
