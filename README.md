# CIG — Caption with Intention Generator

> 자막 없는 영상을 업로드하면, AI가 음성 인식 · 음향 분석 · 감정 추론을 거쳐
> 배리어프리(Barrier-free) 가변 자막을 자동으로 생성하는 시스템

---

## 목차

- [프로젝트 개요](#프로젝트-개요)
- [시스템 아키텍처](#시스템-아키텍처)
- [기술 스택](#기술-스택)
- [데이터 스키마](#데이터-스키마)
- [로컬 환경 세팅](#로컬-환경-세팅)
- [브랜치 전략](#브랜치-전략)
- [커밋 컨벤션](#커밋-컨벤션)
- [PR 규칙](#pr-규칙)
- [디렉토리 구조](#디렉토리-구조)
- [개발 규칙](#개발-규칙)
- [팀 역할 분담](#팀-역할-분담)

---

## 프로젝트 개요

CIG는 청각 장애인 및 자막이 필요한 사용자를 위해 단순한 텍스트 자막을 넘어,
**말의 감정과 음량을 시각적으로 표현하는 가변 자막**을 자동 생성합니다.

### 핵심 기능

- 영상 업로드 → 단어 단위 STT 및 타임스탬프 추출
- 음향 신호 분석 → 볼륨 / 피치 5단계 정규화
- LLM 문맥 추론 → 화자 자율 추론 + 단어별 감정 분류 (4종)
- 프론트엔드 → RobotoFlex 가변 폰트 + 감정별 CSS 모션 실시간 렌더링

### 데모 기준

초기 프로토타입은 **영어 서양 영화 영상**을 기준으로 개발합니다.
(영어 Whisper 인식률 및 오픈소스 레퍼런스 풍부)

---

## 시스템 아키텍처

```
[영상 업로드]
      │
      ▼
[OpenAI Whisper] ── 텍스트 + 단어별 타임스탬프
      │
      ├──────────────────────┐
      ▼                      ▼
[음향 분석 트랙]         [LLM 감정 추론 트랙]
 Librosa                  Ollama / vLLM
 볼륨(dB) / 피치 계산      화자 자율 추론
 5단계 동적 정규화         감정 4종 분류
      │                      │
      └──────────┬───────────┘
                 ▼
         [Nested JSON 결합]
                 │
                 ▼
     [Gradio 프론트엔드 렌더링]
      RobotoFlex 가변 폰트
      감정별 CSS 모션 애니메이션
```

---

## 기술 스택

| 역할 | 기술 | 비고 |
|------|------|------|
| STT | OpenAI Whisper | `word_timestamps=True` |
| 음향 분석 | Librosa | `rms`, `pyin` 함수 사용 |
| 로컬 LLM | Ollama | JSON Structured Output 강제 |
| 스키마 검증 | Pydantic v2 | `Word`, `Sentence` 모델 |
| UI | Gradio + gr.HTML | RobotoFlex.ttf 가변 폰트 |
| 코드 품질 | black + ruff + pre-commit | 커밋 전 자동 실행 |

---

## 데이터 스키마

**모든 모듈의 최종 출력은 이 구조를 따릅니다.** 임의로 변경하지 마세요.
스키마 정의 위치: `backend/schemas/word_schema.py`

```json
[
  {
    "sentence_id": 1,
    "speaker": "Character_A",
    "text": "Watch out! There is a car coming behind you!",
    "words": [
      {
        "word": "Watch",
        "timestamp_start": 1.23,
        "timestamp_end": 1.55,
        "emotion": "anger",
        "volume_level": 5,
        "pitch_level": 4
      }
    ]
  }
]
```

### 필드 규칙

| 필드 | 타입 | 허용값 |
|------|------|--------|
| `emotion` | string | `joy` \| `sadness` \| `anger` \| `neutral` |
| `volume_level` | int | `1` ~ `5` (영상 전체 기준 상대 정규화) |
| `pitch_level` | int | `1` ~ `5` (영상 전체 기준 상대 정규화) |

---

## 로컬 환경 세팅

### 사전 요구사항

- Python 3.10 이상
- [Ollama](https://ollama.ai) 설치 및 실행 중
- ffmpeg 설치 (`brew install ffmpeg` / `apt install ffmpeg`)

### 설치

```bash
# 1. 레포 클론
git clone https://github.com/yourteam/cig.git
cd cig

# 2. 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. 의존성 설치
pip install -e ".[dev]"

# 4. 환경변수 설정
cp .env.example .env
# .env 파일을 열어 본인 환경에 맞게 수정

# 5. pre-commit 훅 설치
pre-commit install
```

### 환경변수 설명 (`.env`)

```bash
OLLAMA_HOST=http://localhost:11434   # Ollama 서버 주소
OLLAMA_MODEL=llama3                  # 사용할 LLM 모델명
WHISPER_MODEL_SIZE=base              # tiny / base / small / medium / large
DATA_DIR=./data
OUTPUT_DIR=./data/outputs
```

### 실행 확인

```bash
# Gradio UI 실행
python frontend/app.py

# 파이프라인 단독 테스트
python -m backend.pipeline.runner --input data/samples/test.mp4
```

---

## 브랜치 전략

```
main
 └── develop          ← 통합 브랜치 (PR은 여기로)
      ├── feat/stt
      ├── feat/audio
      ├── feat/llm
      ├── feat/ui
      └── fix/...
```

| 브랜치 | 용도 | 직접 push |
|--------|------|-----------|
| `main` | 배포용 최종본 | ❌ 금지 |
| `develop` | 통합 및 리뷰 | ❌ PR만 가능 |
| `feat/*` | 기능 개발 | ✅ |
| `fix/*` | 버그 수정 | ✅ |

---

## 커밋 컨벤션

```
<type>: <subject>

# 예시
feat: Whisper word_timestamps 파싱 구현
fix: Librosa pyin 무음 구간 예외처리 추가
refactor: emotion_tagger JSON 파싱 로직 분리
test: STT transcriber 유닛 테스트 추가
docs: README 환경 세팅 섹션 보완
chore: pre-commit ruff 버전 업데이트
```

| 타입 | 사용 상황 |
|------|-----------|
| `feat` | 새 기능 |
| `fix` | 버그 수정 |
| `refactor` | 동작 변경 없는 코드 개선 |
| `test` | 테스트 추가 / 수정 |
| `docs` | 문서만 변경 |
| `chore` | 빌드, 설정 파일 변경 |

---

## PR 규칙

1. `develop` 브랜치로만 PR 생성
2. PR 제목은 커밋 컨벤션과 동일한 형식
3. 셀프 리뷰 후 팀원 **최소 1명** 승인 필요
4. 머지 전 CI (pre-commit) 통과 필수
5. PR 설명에 아래 항목 포함:

```markdown
## 작업 내용
- 

## 테스트 방법
- 

## 관련 이슈
- 
```

---

## 디렉토리 구조

```
cig/
├── CLAUDE.md                   # Claude Code 컨텍스트 파일
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
│
├── backend/
│   ├── schemas/
│   │   └── word_schema.py      # Pydantic 모델 (Word, Sentence)
│   ├── stt/
│   │   └── transcriber.py      # Whisper 실행, 타임스탬프 추출
│   ├── audio/
│   │   └── analyzer.py         # Librosa 볼륨/피치 분석
│   ├── llm/
│   │   └── emotion_tagger.py   # Ollama 감정 추론
│   └── pipeline/
│       └── runner.py           # 통합 파이프라인
│
├── frontend/
│   └── app.py                  # Gradio UI 진입점
│
├── data/
│   ├── samples/                # 테스트 영상 (.gitignore)
│   └── outputs/                # 처리 결과 JSON (.gitignore)
│
└── tests/
    ├── test_stt/
    ├── test_audio/
    └── test_llm/
```

---

## 개발 규칙

### 필수

- 모든 함수에 **타입 힌트** 작성
- 환경변수는 반드시 **python-dotenv**로만 읽기 (하드코딩 금지)
- 외부 라이브러리 추가 시 **pyproject.toml**에만 추가
- 새 기능 추가 시 `tests/` 아래 대응 테스트 함께 작성

### 금지

- `.env` 파일 커밋 (API 키, 경로 정보 포함)
- `data/samples/`, `data/outputs/` 커밋 (용량 큰 파일)
- 스키마 (`word_schema.py`) 단독 수정 — 변경 시 반드시 팀 전체 논의

### Claude Code 사용 시

- 작업 시작 전 `CLAUDE.md` 내용 확인
- 생성된 코드도 pre-commit 훅 통과 후 커밋
- 스키마 변경이 필요하다고 판단되면 코드 수정 전 팀에 먼저 공유

---

## 팀 역할 분담

| 브랜치 | 담당 모듈 | 주요 작업 |
|--------|-----------|-----------|
| `feat/stt` | `backend/stt/` | Whisper 실행, 타임스탬프 파싱 |
| `feat/audio` | `backend/audio/` | Librosa RMS/pyin, 5단계 정규화 |
| `feat/llm` | `backend/llm/` | Ollama 연동, 감정 JSON 출력 |
| `feat/ui` | `frontend/` | Gradio UI, CSS 애니메이션 |

> 파이프라인 통합 (`backend/pipeline/runner.py`) 은 1A/1B/LLM 트랙이
> 각자 출력 검증 완료 후 공동 작업합니다.
