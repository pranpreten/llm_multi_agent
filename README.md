# 스마트 제조 멀티 에이전트 시스템 (MAS)

스마트 제조 환경에서 예지 보전과 최적화를 위한 지능형 멀티 에이전트 시스템입니다. LLM 기반 오케스트레이션으로 제조 데이터를 자동 로드 / 전처리 / 분석하고 처방적 정비 권고까지 생성합니다.

## 🚀 주요 기능

- **LLM 기반 오케스트레이션**: Google Gemini 또는 로컬 LLM (Ollama) 으로 워크플로우를 동적 계획
- **적응형 지능**: 모델 자동 선택 + 성능 기반 retry
- **지능형 전처리**: feature 분석, 인코딩, 결측치 처리 자동화
- **다중 문제 유형**: 분류 / 회귀 / 이상 탐지 지원
- **처방적 권고**: 우선순위 기반 정비 액션 플랜 자동 생성
- **Human-in-the-Loop**: 핵심 결정 단계마다 사용자 승인 인터페이스
- **종합 로깅**: 상세한 audit trail 및 성능 메트릭

## ✨ 최근 개선 사항

### 버그 수정 및 개선 (최신 릴리스)

1. **One-Hot Encoding 이슈 수정**: 식별자 컬럼 (예: `Machine_ID`) 이 one-hot 인코딩되지 않고 pass-through feature 로 처리되어 feature 폭발 및 성능 저하 방지
2. **ID 컬럼 처리 개선**: 모든 모델이 학습 시 ID 컬럼을 자동으로 drop 하면서도 권고 / 리포트용으로 보존
3. **Contributing Factors 강화**: 모든 우선순위 등급 (Critical / Medium / Low) 에서 generic 메시지 대신 실제 feature 값 기반의 상세 정보 표시
4. **워크플로우 완료 처리 개선**: 종료 시그널을 정확히 처리하도록 LLM 검증 로직 수정
5. **데이터 흐름 추적 강화**: 전체 파이프라인의 data shape 추적으로 디버깅과 투명성 향상

## 📋 사전 요구사항

- **Python**: 3.8 이상
- **API 키**: Google Gemini API 키 (로컬 LLM 만 쓰면 선택)
- **Ollama**: 로컬 LLM 사용 시 (선택)

## 🛠️ 설치

### 1. 저장소 클론

```bash
git clone <repository-url>
cd llm_multi_agent
```

### 2. 가상환경 생성

```bash
# 가상환경 생성
python3 -m venv mas_venv

# 활성화
# macOS/Linux
source mas_venv/bin/activate
# Windows
mas_venv\Scripts\activate
```

> Conda 사용자는 [SETUP.md](SETUP.md) 참고.

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정 (선택)

Gemini 사용 시 `.env` 생성:

```bash
echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env
```

### 5. Ollama 설치 (선택 — 로컬 LLM 용)

```bash
# Ollama 설치
curl -fsSL https://ollama.ai/install.sh | sh

# 모델 다운로드 (Qwen3 예시)
ollama pull qwen3:4b
```

## 🎯 빠른 시작

### 대화형 모드 (처음 사용자 권장)

```bash
python3 main_llm.py
```

다음 단계를 안내합니다:
- 데이터셋 선택
- Feature 및 타겟 컬럼 선택
- 문제 유형 식별
- 핵심 결정 승인

### 자동 모드

```bash
python3 main_llm.py --auto --dataset "data/Smart Manufacturing Maintenance Dataset/smart_maintenance_dataset.csv"
```

### 로컬 LLM 사용

```bash
python3 main_llm.py --decision-llm ollama --decision-model qwen3:4b
```

### Streamlit 웹앱 (시각화)

```bash
streamlit run webapp.py --server.port 8503
```

브라우저에서 http://localhost:8503 자동 오픈.

## 📖 사용 예시

### 기본 분석

```bash
# 기본 Gemini LLM 으로 실행
python3 main_llm.py

# 자동 schema discovery 모드
python3 main_llm.py --auto --dataset "path/to/dataset.csv"

# 로컬 LLM 으로 실행
python3 main_llm.py --decision-llm ollama --decision-model qwen3:4b

# 모든 데이터셋 일괄 처리
python3 main_llm.py --batch
```

### 고급 설정 (Hybrid 모드)

```bash
# 강한 LLM (Gemini) + 가벼운 SLM (Ollama Qwen) 조합
python3 main_llm.py \
    --planner-llm gemini \
    --decision-llm ollama \
    --decision-model qwen3:4b \
    --auto

# 특정 데이터셋 + 자동 모드
python3 main_llm.py \
    --auto \
    --dataset "data/Intelligent Manufacturing Dataset/manufacturing_6G_dataset.csv"
```

## 🧠 아키텍처

### 에이전트 구성

1. **LLM Planner Agent**: LLM 추론으로 워크플로우 오케스트레이션
2. **Data Loader Agent**: 데이터셋 로드 및 inspection
3. **Preprocessing Agent**: 지능형 feature 분석 및 데이터 준비
4. **Dynamic Analysis Agent**: 자동 모델 선택 + 다중 모델 분석
5. **Optimization Agent**: 처방적 정비 권고 생성

### 핵심 컴포넌트

- **Schema Discovery**: 데이터셋 자동 이해 (컬럼 dtype, role 추론)
- **Tool Decider**: 지능형 전처리 / 모델 선택
- **Adaptive Intelligence**: 성능 기반 모델 자동 전환
- **Intelligent Summarization**: 깔끔한 출력 + 전체 로깅

## 📊 지원 문제 유형

- **분류 (Classification)**: 카테고리 예측 (예: 정비 우선순위 Low/Medium/High)
- **회귀 (Regression)**: 연속값 예측 (예: 고장 확률)
- **이상 탐지 (Anomaly Detection)**: 비정상 패턴 식별 (타겟 컬럼 불필요)

## 🔧 설정

### 명령줄 옵션

| 옵션 | 설명 |
|--------|-------------|
| `--planner-llm` | Planner LLM 백엔드 (gemini, ollama, mock) |
| `--planner-model` | Planner 모델명 |
| `--decision-llm` | Decision LLM 백엔드 (ollama, mock, None) |
| `--decision-model` | Decision 모델명 (예: qwen3:4b) |
| `--dataset` | CSV 데이터셋 경로 |
| `--target-column` | 타겟 컬럼 명시 (auto schema discovery 우회) |
| `--problem-type` | 문제 유형 명시 (classification / regression / anomaly_detection) |
| `--feature-columns` | feature 컬럼 명시 (콤마 구분) |
| `--auto` | 자동 모드 + schema discovery |
| `--batch` | data/ 폴더의 모든 데이터셋 처리 |
| `--interface` | HITL 인터페이스 (cli 또는 web) |

## 📁 프로젝트 구조

```
llm_multi_agent/
├── agents/                       # 핵심 에이전트 구현
│   ├── llm_planner_agent.py      # LLM 오케스트레이션
│   ├── data_loader_agent.py      # 데이터 로딩
│   ├── preprocessing_agent.py    # 전처리
│   ├── dynamic_analysis_agent.py # 분석 / 모델 학습
│   └── optimization_agent.py     # 처방적 권고
├── utils/                        # 유틸리티 모듈
│   ├── schema_discovery.py       # 자동 schema 탐지
│   ├── tool_decider.py           # 모델 / 도구 선택
│   ├── hitl_interface.py         # Human-in-loop UI
│   └── reporting.py              # 로깅 및 리포팅
├── data/                         # 샘플 데이터셋
│   ├── Smart Manufacturing Maintenance Dataset/
│   └── Intelligent Manufacturing Dataset/
├── documentation/                # 문서
│   ├── usage_guide.md            # 상세 사용 가이드
│   ├── architecture_and_workflow.md
│   └── adaptive_intelligence_system.md
├── logs/                         # 실행 로그 (gitignore)
├── webapp.py                     # Streamlit 웹앱
├── main_llm.py                   # CLI 진입점
└── requirements.txt              # 의존성
```

## 📈 출력

### 콘솔 출력

실시간 진행 상황과 최종 요약을 제공.

### 로그 파일

매 실행 시 `logs/` 폴더에 생성:
- `logs/workflow_report_*.json`: 상세 워크플로우 리포트
- `logs/detailed_results_*.json`: 전체 구조화 결과
- `logs/publication_snapshot_*.json`: 발표 / 리포트용 요약
- `logs/*_recommendations.csv`: 추천 액션 표
- `logs/hitl_audit.json`: 사용자 승인 audit trail

## 🐛 트러블슈팅

### 자주 발생하는 문제

**1. Import 오류**
```bash
# 가상환경 활성화 확인
source mas_venv/bin/activate

# 설치 확인
pip list | grep scikit-learn
```

**2. API Key 문제**
```bash
# .env 파일 존재 확인
cat .env

# API 키 유효성 빠른 검증
python -c "import google.generativeai as genai; genai.configure(api_key='your_key')"
```

**3. Ollama 연결 문제**
```bash
# Ollama 데몬 실행
ollama serve

# 모델 확인
ollama list
```

**4. 데이터셋 문제**
```bash
# 경로 확인
ls -la data/

# CSV 포맷 확인
head -5 "data/Smart Manufacturing Maintenance Dataset/smart_maintenance_dataset.csv"
```

### 디버그 모드

상세 로깅 활성화:
```bash
export LOG_LEVEL=DEBUG
python3 main_llm.py --auto
```

## 🎓 더 알아보기

- [Conda 셋업 가이드](SETUP.md) — 가상환경 + Ollama + Streamlit 단계별 가이드
- [빠른 시작 가이드](QUICKSTART.md) — venv 기반 5분 셋업
- [상세 사용 가이드](documentation/usage_guide.md)
- [아키텍처 및 워크플로우](documentation/architecture_and_workflow.md)
- [적응형 지능 시스템](documentation/adaptive_intelligence_system.md)

## 🤝 기여

기여 환영합니다! 다음 가이드라인을 따라주세요:
1. 기능 브랜치 생성
2. 새 기능에 대한 테스트 추가
3. 관련 문서 업데이트
4. Pull Request 제출

## 📝 라이선스

[라이선스 정보 추가 필요]

## 🙏 감사의 말

이 프로젝트는 지능형 제조 및 예지 보전 시스템 연구를 위해 개발되었습니다.

## 🔗 연락처

[연락처 정보 추가 필요]

---

**시작할 준비가 되셨나요?** `python3 main_llm.py` 또는 `streamlit run webapp.py --server.port 8503` 으로 첫 분석을 시작하세요!
