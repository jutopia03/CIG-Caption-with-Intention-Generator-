# CIG — Caption with Intention Generator

## 프로젝트 목적
자막 없는 영상을 업로드하면 Whisper STT + Librosa 음향 분석 + 로컬 LLM 감정 추론을
거쳐 배리어프리 가변 자막을 Gradio 웹 UI로 실시간 렌더링하는 자동화 시스템.

## 기술 스택
- STT: openai-whisper (word_timestamps=True)
- 음향 분석: librosa (rms, pyin)
- LLM: ollama (로컬, JSON mode 강제)
- UI: gradio + gr.HTML + RobotoFlex 가변폰트
- 스키마 검증: pydantic v2

## 핵심 데이터 스키마 (모든 모듈이 이 구조를 따름)
최종 출력은 아래 Nested JSON 배열 형태:
- sentence_id, speaker, text, words[]
- words 각 항목: word, timestamp_start, timestamp_end,
  emotion(joy|sadness|anger|neutral), volume_level(1-5), pitch_level(1-5)

## 디렉토리 구조 (목표)
cig/
├── AGENTS.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── backend/
│   ├── __init__.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── word_schema.py       # Pydantic 모델 정의
│   ├── stt/
│   │   ├── __init__.py
│   │   └── transcriber.py       # Whisper 실행 및 타임스탬프 추출
│   ├── audio/
│   │   ├── __init__.py
│   │   └── analyzer.py          # Librosa RMS/pyin, 5단계 정규화
│   ├── llm/
│   │   ├── __init__.py
│   │   └── emotion_tagger.py    # Ollama 감정 추론, JSON Structured Output
│   └── pipeline/
│       ├── __init__.py
│       └── runner.py            # 세 모듈 결합 → 최종 JSON
├── frontend/
│   ├── __init__.py
│   └── app.py                   # Gradio UI 진입점
├── data/
│   ├── samples/                 # 테스트 영상 (gitignore)
│   └── outputs/                 # 처리 결과 JSON (gitignore)
└── tests/
    ├── __init__.py
    ├── test_stt/
    ├── test_audio/
    └── test_llm/

## 개발 규칙
- 모든 함수 타입 힌트 필수
- 환경변수는 python-dotenv로만 읽기
- 외부 의존성은 pyproject.toml에만 추가
- 테스트는 tests/ 아래 모듈명과 동일하게