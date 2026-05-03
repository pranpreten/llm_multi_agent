# Setup Guide — Conda 가상환경 + 로컬 웹앱 실행

이 문서는 처음 클론한 사람이 5~10분 안에 환경 구축하고 Streamlit 웹앱을 띄우는 방법을 정리합니다.

---

## 사전 요구사항

- **Python 3.10+** (conda 또는 venv)
- **Gemini API 키** (선택): Planner LLM 으로 Gemini 쓸 때
- **Ollama** (선택): SLM hybrid 모드 또는 완전 오프라인 실행 시

---

## 1. 저장소 클론

```bash
git clone https://github.com/pranpreten/llm_multi_agent.git
cd llm_multi_agent
```

## 2. Conda 가상환경 생성 (권장)

```bash
# 새 환경 (Python 3.10)
conda create -n mas python=3.10 -y
conda activate mas
```

> **venv 쓰고 싶다면:**
> ```bash
> python -m venv mas_venv
> # Windows
> mas_venv\Scripts\activate
> # macOS/Linux
> source mas_venv/bin/activate
> ```

## 3. 의존성 설치

```bash
pip install -r requirements.txt
```

설치되는 핵심 패키지:
- `streamlit` — 웹 UI
- `google-generativeai` — Gemini 클라이언트
- `ollama` — 로컬 SLM 클라이언트
- `pandas`, `numpy`, `scipy`, `scikit-learn` — ML 파이프라인
- `python-dotenv` — 환경변수 로딩

## 4. 환경변수 설정 (`.env`)

`.env.example` 을 복사해서 `.env` 만들고 키 입력:

```bash
# macOS/Linux
cp .env.example .env

# Windows (PowerShell)
Copy-Item .env.example .env
```

`.env` 편집:
```
GEMINI_API_KEY=실제_API_키_입력
```

> `.env` 는 `.gitignore` 에 등록되어 있어 commit 안 됨. **API 키 외부 노출 위험 없음.**

> Gemini 안 쓰고 `mock` 또는 `ollama` 만 쓸 거면 `.env` 비워둬도 됩니다.

## 5. Ollama 설치 (SLM Hybrid 모드 사용 시)

논문의 진정한 hybrid 구조 (Planner=Gemini + Decision=SLM) 또는 완전 로컬 실행을 원할 때.

### 설치

**Windows:**
```powershell
winget install Ollama.Ollama
```

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 모델 다운로드

```bash
ollama pull qwen3:4b   # ~2.5GB, 권장 SLM
ollama list            # 설치 확인
```

### 데몬 실행 확인

```bash
curl http://localhost:11434/api/tags
```
JSON 응답 나오면 OK. 안 나오면 시작 메뉴 / 트레이에서 "Ollama" 실행.

## 6. Streamlit 웹앱 실행

```bash
streamlit run webapp.py --server.port 8503
```

브라우저 자동 오픈 — 또는 직접 http://localhost:8503

### 사이드바 설정 가이드

| 항목 | 권장값 | 비고 |
|---|---|---|
| 데이터셋 | 내장 CSV 선택 | `data/` 아래 모두 자동 노출 |
| 문제유형 | `auto` | dtype 기반 자동 추론 |
| 타겟 컬럼 | 데이터셋의 라벨 컬럼 | (예: `Maintenance_Priority`) |
| Planner LLM | `gemini` | 강한 추론 (오케스트레이션) |
| Decision LLM | `same as planner` 또는 `ollama` | `ollama` 선택 시 진짜 hybrid |

## 7. (옵션) CLI 모드 실행

웹앱 안 쓰고 명령줄에서 돌리고 싶을 때.

### Gemini 단일 모드
```bash
python main_llm.py \
  --dataset "data/Smart Manufacturing Maintenance Dataset/smart_maintenance_dataset.csv" \
  --target-column Maintenance_Priority --problem-type classification
```

### Hybrid (Gemini + Ollama qwen3:4b)
```bash
python main_llm.py \
  --planner-llm gemini \
  --decision-llm ollama --decision-model qwen3:4b \
  --dataset "data/Smart Manufacturing Maintenance Dataset/smart_maintenance_dataset.csv" \
  --target-column Maintenance_Priority --problem-type classification
```

### 완전 오프라인 (Mock)
```bash
python main_llm.py --planner-llm mock --decision-llm mock \
  --dataset "data/Smart Manufacturing Maintenance Dataset/smart_maintenance_dataset.csv" \
  --target-column Maintenance_Priority --problem-type classification
```

### 자동 모드 (auto schema discovery)
```bash
python main_llm.py --auto \
  --dataset "data/Smart Manufacturing Maintenance Dataset/smart_maintenance_dataset_labeled.csv"
```

## 8. 결과물 위치

워크플로우 1회 실행 시 생성되는 파일:

| 파일 | 내용 |
|---|---|
| `logs/workflow_report_*.json` | 단계별 도구 실행 + LLM reason + HITL 이벤트 |
| `logs/detailed_results_*.json` | 모델 결과 + feature 분석 |
| `logs/publication_snapshot_*.json` | 발표용 요약 |
| `logs/*_recommendations.csv` | 추천 액션 표 (CSV) |
| `logs/hitl_audit.json` | HITL 승인/거부 누적 audit |

## 9. 트러블슈팅

| 증상 | 해결 |
|---|---|
| `ModuleNotFoundError: streamlit` | conda 환경 활성화 안 됨 → `conda activate mas` |
| `GEMINI_API_KEY not found` | `.env` 파일 존재 + 키 입력 확인 |
| Ollama 호출 실패 | `curl http://localhost:11434/api/tags` 안 되면 데몬 실행 |
| `qwen3:4b not found` | `ollama pull qwen3:4b` 다시 실행 |
| Streamlit 포트 충돌 | 다른 포트: `--server.port 8504` |
| 한글/이모지 깨짐 (Windows) | 명령 앞에 `set PYTHONUTF8=1` (cmd) 또는 `$env:PYTHONUTF8=1` (PowerShell) |
| Gemini quota 초과 (분당 5) | 잠시 대기 후 재시도 또는 Decision LLM 을 ollama 로 |

## 10. 환경 정리 (제거)

```bash
# Conda
conda deactivate
conda env remove -n mas

# Ollama 모델 제거 (디스크 회수)
ollama rm qwen3:4b
```

---

## 참고 문서

- [README.md](README.md) — 시스템 개요 및 아키텍처
- [QUICKSTART.md](QUICKSTART.md) — venv 기반 5분 셋업
- [documentation/](documentation/) — 상세 문서
