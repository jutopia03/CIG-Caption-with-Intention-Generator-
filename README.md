# CIG — Caption with Intention Generator

> 자막 없는 영상을 업로드하면 AI가 자동으로 볼륨·화자별 배리어프리 가변 자막을 생성하는 시스템

[Caption with Intention 캠페인](https://www.captionwithintention.org/)의 세 가지 원칙 — **Attribution(화자 구분)**, **Synchronization(단어 동기화)**, **Intonation(억양·볼륨 시각화)** — 을 AI로 자동화한 시스템입니다.

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

전 세계 4억 6,600만 명의 청각장애인이 영상 콘텐츠를 볼 때 기존 자막은 텍스트 정보만 전달합니다. 말의 강약, 억양, 화자 구분 등 청각 정보의 대부분이 손실됩니다.

CIG는 영상 하나를 업로드하면 다음을 자동으로 처리합니다.

- **화자 구분**: 화자마다 고유 색상을 자동으로 부여 (최대 6명)
- **볼륨 시각화**: 볼륨 1~5단계를 font-size 16px~40px로 실시간 매핑
- **단어 동기화**: 단어가 발화되는 순간 해당 단어만 강조 표시
- **다국어 자막**: 영어·한국어 입출력 지원 (영→영, 영→한, 한→한)
- **자막 커스터마이징**: 폰트 크기, 위치, 투명도, 애니메이션 실시간 조절

---

## 시스템 아키텍처

```
[영상 업로드]
      │
      ▼
[ffmpeg] ── WAV 추출
      │
      ├─────────────────────────┐
      ▼                         ▼
[AssemblyAI]              [Librosa]
 STT + 화자분리              볼륨(RMS) / 피치(pyin)
 단어별 타임스탬프              1~5단계 정규화
      │                         │
      └────────────┬────────────┘
                   ▼
          [Pydantic v2 검증]
                   │
                   ▼ (출력=한국어)
             [DeepL 번역]
                   │
                   ▼
           [JSON 저장/반환]
                   │
                   ▼
      [Gradio 프론트엔드 렌더링]
       RobotoFlex 가변 폰트
       화자별 색상 + 볼륨별 크기
```

---

## 기술 스택

| 역할 | 기술 | 비고 |
|------|------|------|
| STT + 화자분리 | AssemblyAI API | universal-3-pro (한국어: universal-2) |
| 음향 분석 | Librosa | RMS 볼륨, pyin 피치 → 1~5단계 정규화 |
| 번역 | DeepL API | 문장 단위, 한↔영 |
| 스키마 검증 | Pydantic v2 | Word, Sentence 모델 |
| UI | Gradio + gr.HTML | RobotoFlex 가변 폰트 |
| 코드 품질 | black + ruff + pre-commit | 커밋 전 자동 실행 |

---

## 데이터 스키마

**모든 모듈의 최종 출력은 이 구조를 따릅니다.** 스키마 정의 위치: `backend/schemas/word_schema.py`

```json
[
  {
    "sentence_id": 1,
    "speaker": "Character_A",
    "text": "You know where to find me.",
    "text_translated": "어디서 날 찾을 수 있는지 알잖아.",
    "words": [
      {
        "word": "You",
        "timestamp_start": 0.96,
        "timestamp_end": 1.08,
        "emotion": "neutral",
        "volume_level": 3,
        "pitch_level": 2
      }
    ]
  }
]
```

> `text_translated` 필드는 출력 언어가 한국어일 때만 포함됩니다.

### 필드 규칙

| 필드 | 타입 | 허용값 |
|------|------|--------|
| `emotion` | string | `joy` \| `sadness` \| `anger` \| `fear` \| `surprise` \| `disgust` \| `contempt` \| `neutral` |
| `volume_level` | int | `1` ~ `5` (영상 전체 기준 상대 정규화) |
| `pitch_level` | int | `1` ~ `5` (영상 전체 기준 상대 정규화) |

---

## 로컬 환경 세팅

### 사전 요구사항

- Python 3.10 이상
- ffmpeg 설치 (`brew install ffmpeg` / `apt install ffmpeg`)
- AssemblyAI API 키 ([발급](https://www.assemblyai.com/app/account))
- DeepL API 키 ([발급](https://www.deepl.com/your-account/keys))

### 설치

```bash
# 1. 레포 클론
git clone https://github.com/jutopia03/CIG-Caption-with-Intention-Generator-.git
cd CIG-Caption-with-Intention-Generator-

# 2. 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. 의존성 설치
pip install -e ".[dev]"

# 4. 환경변수 설정
cp .env.example .env
# .env 파일을 열어 API 키 및 경로 입력

# 5. pre-commit 훅 설치
pre-commit install
```

### 환경변수 설명 (`.env`)

```
ASSEMBLYAI_API_KEY=your_key_here     # AssemblyAI API 키
DEEPL_API_KEY=your_key_here          # DeepL API 키
LLM_HOST=http://localhost:11434      # Ollama 서버 주소 (확장 기능용)
LLM_MODEL=llama3.1:8b                # LLM 모델명 (확장 기능용)
DATA_DIR=data/samples                # 샘플 영상 경로
OUTPUT_DIR=data/outputs              # 결과 JSON 저장 경로
```

### 실행

```bash
# Gradio UI 실행
python frontend/app.py

# 테스트 실행
python -m pytest tests/ -v
```

---

## 브랜치 전략

```
main  ← 최종 배포본 (PR만 가능)
 └── feat/*   ← 기능 개발
 └── fix/*    ← 버그 수정
```

| 브랜치 | 용도 | 직접 push |
|--------|------|-----------|
| `main` | 최종 배포본 | ❌ PR만 가능 |
| `feat/*` | 기능 개발 | ✅ |
| `fix/*` | 버그 수정 | ✅ |

---

## 커밋 컨벤션

```
<type>: <subject>

# 예시
feat: AssemblyAI 화자분리 파라미터 추가
fix: Librosa pyin 무음 구간 예외처리
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
| `test` | 테스트 추가/수정 |
| `docs` | 문서만 변경 |
| `chore` | 빌드, 설정 파일 변경 |

---

## PR 규칙

1. `main` 브랜치로 PR 생성
2. PR 제목은 커밋 컨벤션과 동일한 형식
3. 팀원 **최소 1명** 승인 필요
4. 머지 전 CI (pre-commit) 통과 필수
5. PR 설명 양식:

```
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
CIG-Caption-with-Intention-Generator-/
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
│
├── backend/
│   ├── schemas/
│   │   └── word_schema.py           # Pydantic 모델 (Word, Sentence)
│   ├── stt/
│   │   └── assemblyai_transcriber.py # STT + 화자분리 + 타임스탬프
│   ├── audio/
│   │   └── analyzer.py              # Librosa 볼륨/피치 분석
│   ├── llm/
│   │   └── emotion_tagger.py        # 감정 추론 (확장 기능)
│   ├── translation/
│   │   └── translator.py            # DeepL 번역
│   └── pipeline/
│       └── runner.py                # 통합 파이프라인
│
├── frontend/
│   └── app.py                       # Gradio UI 진입점
│
├── data/
│   ├── samples/                     # 테스트 영상 (.gitignore)
│   └── outputs/                     # 결과 JSON (.gitignore)
│
└── tests/
    ├── test_stt/
    ├── test_audio/
    ├── test_llm/
    ├── test_pipeline/
    └── test_translation/
```

---

## 개발 규칙

### 필수

- 모든 함수에 **타입 힌트** 작성
- 환경변수는 반드시 **python-dotenv**로만 읽기 (하드코딩 금지)
- 외부 라이브러리 추가 시 **pyproject.toml**에만 추가
- 새 기능 추가 시 `tests/` 아래 대응 테스트 함께 작성

### 금지

- `.env` 파일 커밋 (API 키 포함)
- `data/samples/`, `data/outputs/` 커밋 (용량 큰 파일)
- 스키마(`word_schema.py`) 단독 수정 — 변경 시 반드시 팀 전체 논의

---

## 팀 역할 분담

| 담당 | 주요 작업 |
|------|-----------|
| STT / 화자분리 | AssemblyAI 연동, 타임스탬프 파싱 |
| 음향 분석 | Librosa RMS/pyin, 5단계 정규화 |
| 감정 추론 | rule-based 필터 + LLM 2차 보완 |
| 번역 | DeepL 연동, 다국어 자막 출력 |
| UI | Gradio UI, 자막 커스터마이징 패널, 진행률 바 |
| 파이프라인 통합 | runner.py, Pydantic 검증, JSON 저장 |
