# 📝 세션 노트 — Farahani Smart Manufacturing MAS

> 마지막 업데이트: 2026-04-29
> 다음 세션 재개를 위한 상태 정리

---

## 🎯 프로젝트 한 줄 요약

Farahani et al. 2026 논문 "Hybrid agentic AI and multi-agent systems in smart manufacturing"의 구현체를 분석/실행/시각화하는 캡스톤 작업.

**핵심 시스템**: LLM(Gemini)이 ML 파이프라인(데이터로드→전처리→모델학습→정비추천)을 동적으로 조립하는 5-에이전트 구조.

---

## ✅ 지금까지 한 것

### 1. 코드 구조 완전 파악
- 5개 에이전트 (`agents/*.py`), 5개 utils (`utils/*.py`), entry point `main_llm.py` 모두 분석
- 분류/회귀/이상탐지 차이, R² 계산 원리, agentic loop 구조까지 깊이 이해

### 2. 데이터셋 결정
- **선택: `smart_maintenance_dataset.csv`** (1,430행, 빠른 반복)
- 거부: `manufacturing_6G_dataset.csv` (10만행) — 데이터 누수로 Accuracy 100% 나옴, 발표 시 신뢰도 ↓

### 3. 코드 패치
| 파일 | 변경 |
|---|---|
| `main_llm.py` | `--target-column`, `--problem-type`, `--feature-columns` 옵션 추가 |
| `utils/hitl_interface.py` | `WebHitlInterface`를 큐 기반으로 구현 + `HITL_AUTO=1` 환경변수 인식 |
| **`webapp.py`** (신규) | Streamlit 웹 UI 약 830줄 — 단계별 의사결정 시각화 |

### 4. 실행 검증
- ✅ Mock LLM 모드 end-to-end 통과 (Accuracy 75.17%, 30개 추천)
- ✅ Gemini 모드 1회 동작 확인 (gemini-2.5-flash, .env에 API 키 들어있음)
- ✅ 웹앱 띄워서 사이드바 → 실행 → 4단계 → 추천까지 시연 가능

---

## ⏭️ 다음에 할 일 (우선순위 순)

### 🔥 1순위: Gemini 실전 모드 시연
웹앱에서 LLM 백엔드를 **`gemini`** 로 변경 후 풀 워크플로우 1회 실행.
- **목표**: 진짜 LLM이 단계마다 어떤 reasoning을 하는지 확인 (mock과 비교)
- **유의**: Gemini 무료 등급 분당 5요청 → quota 초과 가능. 너무 빨리 돌리면 retry 메시지 나옴
- 의사결정 카드에 진짜 LLM 추론(5단계 ANALYZE→EVALUATE→IDENTIFY→DECIDE→JUSTIFY)이 보일 것

### 2순위: 회귀/이상탐지 모드도 시연 (비교용)
같은 데이터셋으로 3가지 모드 다 돌려서 결과 비교 → 캡스톤 발표 표 만들기 좋음.
```bash
# 회귀
--target-column Failure_Prob --problem-type regression
# 이상탐지
--problem-type anomaly_detection  (target 불필요)
```

### 3순위: 코드 약점 정리 → 캡스톤 contribution 카드
1. ⚠️ Adaptive Threshold 미구현 (성능 임계값이 R²<0.1, Acc<0.6 하드코딩)
2. ⚠️ Machine_ID = "Unknown" (ID 매핑 깨진 버그)
3. ⚠️ 데이터 누수 자동 검증 없음
4. ⚠️ 분류/회귀 자동 감지 부정확

→ "내가 발견한 약점" 또는 "내가 보완한 부분"으로 발표 자료에 포함 가능

---

## 🚀 실행 방법

### 웹앱 (시각화)
```bash
cd c:/Users/kjlee/paper_research/논문소스코드_Farahani
streamlit run webapp.py --server.port 8503
```
→ 브라우저 자동 오픈, http://localhost:8503

**사용:**
1. 사이드바 → 내장 데이터셋 → `smart_maintenance_dataset.csv`
2. 문제: `classification`, 타겟: `Maintenance_Priority`
3. LLM 백엔드: `mock` (빠름) 또는 `gemini` (실전)
4. **🚀 실행** 클릭

### CLI (분류 + Gemini)
```bash
cd c:/Users/kjlee/paper_research/논문소스코드_Farahani
PYTHONUTF8=1 HITL_AUTO=1 python main_llm.py \
  --dataset "data/Smart Manufacturing Maintenance Dataset/smart_maintenance_dataset.csv" \
  --target-column Maintenance_Priority --problem-type classification
```

---

## 🛠️ 환경 정보
- **OS**: Windows 10 Pro, Python 3.13.9 (anaconda3)
- **셸**: bash 또는 PowerShell
- **포트**: 8503 (Streamlit)
- **인코딩 주의**: 명령어 앞에 항상 `PYTHONUTF8=1` (이모지 깨짐 방지)
- **Gemini API**: `.env` 파일에 `GEMINI_API_KEY=...` 들어있음 (무료 등급, 분당 5요청)

---

## 📋 다음 세션 시작 프롬프트 (복붙용)

새 세션 시작할 때 이 메시지를 첫 프롬프트로 보내세요:

---

```
c:/Users/kjlee/paper_research/논문소스코드_Farahani/SESSION_NOTES.md 읽고
어제 어디까지 했는지 파악해줘. 오늘은 Gemini 실전 모드로 웹앱에서 풀 워크플로우 시연하려고 해.

1. 먼저 Streamlit 웹앱 8503 포트로 띄워줘
2. 어제 결정한 설정(데이터셋 1, 분류, target=Maintenance_Priority) 그대로 가는 건지 확인
3. Gemini 모드로 돌렸을 때 mock 모드와 어떻게 다른지 짚어줘
```

---

## 📂 참고 파일 위치
- 이 노트: `SESSION_NOTES.md` (지금 보는 파일)
- 메인 진입점: `main_llm.py`
- 웹앱: `webapp.py`
- 코어 에이전트: `agents/llm_planner_agent.py` (오케스트레이터, 53KB)
- ML 모델 학습: `agents/dynamic_analysis_agent.py`
- 추천 생성: `agents/optimization_agent.py`
- 결정 로직: `utils/tool_decider.py`
- HITL UI: `utils/hitl_interface.py`
- 결과 로그: `logs/workflow_report_*.json`, `logs/hitl_audit.json`
