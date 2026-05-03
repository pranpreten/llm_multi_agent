"""
Agentic Manufacturing Analyzer - Streamlit Web UI
=================================================
Real-time visualization of the LLM-orchestrated multi-agent workflow.
Run with: streamlit run webapp.py
"""

import os
import sys
import re
import time
import logging
import threading
import queue
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# Make local imports work when launched via `streamlit run webapp.py`
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Force UTF-8 stdout for Windows so emoji-laden logs don't crash
os.environ.setdefault("PYTHONUTF8", "1")

# ----------------------------------------------------------------------------
# Logging bridge: capture backend logs and push them to the UI event queue
# ----------------------------------------------------------------------------

# Patterns we use to recognize structured events inside log messages
_RE_TOOL_DECIDED = re.compile(r"LLM decided: tool='([^']+)', finish=(True|False), reason='(.+)'", re.DOTALL)
_RE_EXECUTING_TOOL = re.compile(r"Executing tool: '([^']+)'")
_RE_TOOL_RESULT = re.compile(r"Tool '([^']+)' result: success=(True|False), message='(.+)'", re.DOTALL)
_RE_TOOLDECIDER = re.compile(r"ToolDecider (selected tool|chose preprocessing strategy|selected): (.+)")
_RE_MODEL_NAME = re.compile(r"^(Linear Regression|Random Forest Regressor|Random Forest Classifier|Random Forest|Logistic Regression|SVM|SVC|Ridge|Lasso|SVR|Isolation Forest)\b")
_RE_LLM_RAW = re.compile(r"(Local LLM response|Gemini response) \(attempt (\d+)\): (.+)", re.DOTALL)
_RE_TURN = re.compile(r"--- LLM Turn (\d+)/(\d+) ---")
_RE_R2 = re.compile(r"R(?:²|\^2):\s*([\-0-9.]+)", re.IGNORECASE)
_RE_ACC = re.compile(r"accuracy:\s*([\-0-9.]+)", re.IGNORECASE)
_RE_MSE = re.compile(r"MSE:\s*([\-0-9.]+)", re.IGNORECASE)
_RE_ANOM = re.compile(r"detected\s+(\d+)\s+anomalies")


class QueueLogHandler(logging.Handler):
    """Forward every log record to the UI event queue, classifying as we go."""

    def __init__(self, event_queue: queue.Queue):
        super().__init__()
        self.event_queue = event_queue

    def emit(self, record: logging.LogRecord):
        try:
            msg = record.getMessage()
        except Exception:
            return

        # Always send the raw log first so the UI can show a verbose stream
        self.event_queue.put({
            "type": "log",
            "level": record.levelname,
            "message": msg,
            "ts": record.created,
        })

        # Then enrich with structured events when patterns match
        m = _RE_TURN.search(msg)
        if m:
            self.event_queue.put({"type": "turn", "current": int(m.group(1)), "total": int(m.group(2))})
            return

        m = _RE_TOOL_DECIDED.search(msg)
        if m:
            self.event_queue.put({
                "type": "llm_decision",
                "tool": m.group(1),
                "finish": m.group(2) == "True",
                "reason": m.group(3).strip(),
            })
            return

        m = _RE_EXECUTING_TOOL.search(msg)
        if m:
            self.event_queue.put({"type": "stage_start", "tool": m.group(1)})
            return

        m = _RE_TOOL_RESULT.search(msg)
        if m:
            self.event_queue.put({
                "type": "stage_end",
                "tool": m.group(1),
                "success": m.group(2) == "True",
                "message": m.group(3).strip(),
            })
            return

        m = _RE_TOOLDECIDER.search(msg)
        if m:
            self.event_queue.put({"type": "tool_decision", "label": m.group(1), "value": m.group(2).strip()})
            return

        # Heuristic: any line that contains a metric value is treated as model performance
        r = _RE_R2.search(msg)
        a = _RE_ACC.search(msg)
        mse_m = _RE_MSE.search(msg)
        if r or a or mse_m:
            metrics: Dict[str, float] = {}
            if r:
                try: metrics["r2"] = float(r.group(1))
                except ValueError: pass
            if a:
                try: metrics["accuracy"] = float(a.group(1))
                except ValueError: pass
            if mse_m:
                try: metrics["mse"] = float(mse_m.group(1))
                except ValueError: pass
            mn = _RE_MODEL_NAME.match(msg)
            self.event_queue.put({
                "type": "model_metric",
                "model": mn.group(1) if mn else None,
                "metrics": metrics,
                "raw": msg.strip(),
            })
            return

        m = _RE_ANOM.search(msg)
        if m:
            self.event_queue.put({"type": "anomalies", "count": int(m.group(1))})
            return


# ----------------------------------------------------------------------------
# Background workflow runner
# ----------------------------------------------------------------------------

def _attach_handler_to_root(handler: logging.Handler):
    root = logging.getLogger()
    if not any(isinstance(h, QueueLogHandler) for h in root.handlers):
        root.addHandler(handler)
    if root.level > logging.INFO:
        root.setLevel(logging.INFO)


def run_workflow_in_thread(config: Dict[str, Any], event_queue: queue.Queue, response_queue: queue.Queue):
    """Worker thread that runs the agentic workflow end-to-end."""

    handler = QueueLogHandler(event_queue)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    _attach_handler_to_root(handler)

    # Lazy imports so the Streamlit page loads fast
    try:
        from agents.llm_planner_agent import LLMPlannerAgent
        from utils.hitl_interface import WebHitlInterface
        from utils.schema_discovery import discover_dataset_schema

        hitl = WebHitlInterface(event_queue=event_queue, response_queue=response_queue)

        # Derive feature columns when not specified (mirror main_llm.auto_select_from_schema)
        feature_cols = config.get("feature_columns")
        problem_type = config.get("problem_type")
        target = config.get("target_column")
        df_head = None
        if not feature_cols:
            df_head = pd.read_csv(config["dataset_path"], nrows=200)
            try:
                schema = discover_dataset_schema(df_head)
                cols_info = schema.get("columns", {})
                feature_cols = [
                    c for c in df_head.columns
                    if c != target and cols_info.get(c, {}).get("role") not in ["identifier", "timestamp"]
                ]
            except Exception:
                feature_cols = [c for c in df_head.columns if c != target]
            event_queue.put({
                "type": "log", "level": "INFO",
                "message": f"Auto-derived {len(feature_cols)} feature columns: {feature_cols}",
                "ts": time.time(),
            })

        # Infer problem_type when user picked "auto" in sidebar
        if not problem_type and target:
            if df_head is None:
                df_head = pd.read_csv(config["dataset_path"], nrows=200)
            try:
                schema = discover_dataset_schema(df_head)
                suggested = schema.get("suggested_targets", []) or []
                target_match = next((s for s in suggested if s.get("column") == target), None)
                if target_match:
                    problem_type = target_match.get("suggested_task")
            except Exception:
                pass
            if not problem_type:
                col = df_head[target]
                problem_type = 'classification' if (str(col.dtype) in ['object', 'category', 'bool'] or col.nunique() <= 20) else 'regression'
            event_queue.put({
                "type": "log", "level": "INFO",
                "message": f"Auto-inferred problem_type='{problem_type}' for target='{target}' (dtype={df_head[target].dtype}, unique={df_head[target].nunique()})",
                "ts": time.time(),
            })

        # Build Planner and Decision LLMs separately (true hybrid possible)
        from agents.local_llm_agent import LocalLLMAgent

        def _build_llm(backend_name: str, model_name: str):
            if backend_name == "mock":
                return LocalLLMAgent(backend="mock")
            if backend_name == "ollama":
                return LocalLLMAgent(backend="ollama", model_name=model_name)
            return None  # gemini → LLMPlannerAgent constructs Gemini internally

        planner_backend = config.get("planner_backend", "gemini")
        decision_backend = config.get("decision_backend", planner_backend)
        planner_model_name = config.get("planner_model", "qwen3:4b")
        decision_model_name = config.get("decision_model", "qwen3:4b")

        llm_agent = _build_llm(planner_backend, planner_model_name)
        decision_llm_agent = _build_llm(decision_backend, decision_model_name)

        planner_label = f"{planner_backend}" + (f" ({planner_model_name})" if planner_backend == "ollama" else "")
        decision_label = f"{decision_backend}" + (f" ({decision_model_name})" if decision_backend == "ollama" else "")
        hybrid_tag = " ★ HYBRID" if planner_backend != decision_backend else ""
        event_queue.put({
            "type": "log", "level": "INFO",
            "message": f"LLMs — Planner: {planner_label}  |  Decision: {decision_label}{hybrid_tag}",
            "ts": time.time(),
        })

        # Auto-approve all HITL prompts when in headless mode
        if config.get("auto_approve_hitl"):
            os.environ["HITL_AUTO"] = "1"
        else:
            os.environ.pop("HITL_AUTO", None)

        planner = LLMPlannerAgent(
            dataset_path=config["dataset_path"],
            feature_columns=feature_cols,
            target_column=target,
            problem_type=problem_type,
            llm_agent=llm_agent,
            decision_llm_agent=decision_llm_agent,
            hitl_interface=hitl,
        )

        goal = config.get("goal") or (
            f"Load the selected dataset, preprocess it, analyze it to solve a "
            f"{problem_type} problem, and generate a prescriptive action plan."
        )

        t0 = time.time()
        planner.run_workflow_with_llm(goal)
        duration = time.time() - t0

        # Extract final artefacts for the UI
        recs = planner.recommendations
        if recs is not None and hasattr(recs, "to_dict"):
            try:
                event_queue.put({"type": "recommendations", "rows": recs.head(50).to_dict("records")})
            except Exception as e:
                event_queue.put({"type": "log", "level": "WARNING", "message": f"Failed to serialize recommendations: {e}", "ts": time.time()})

        # planner.analysis_results wraps raw analyzer output under 'evaluation'
        results_wrapper = planner.analysis_results or {}
        eval_block = results_wrapper.get("evaluation") or {}
        final_metrics = {
            "model": eval_block.get("model"),
            "accuracy": eval_block.get("accuracy"),
            "r2": eval_block.get("r2"),
            "mse": eval_block.get("mse"),
            "n_anomalies": eval_block.get("n_anomalies"),
        }
        event_queue.put({"type": "complete", "duration": duration, "metrics": final_metrics})

    except Exception as e:
        tb = traceback.format_exc()
        event_queue.put({"type": "error", "message": str(e), "traceback": tb})


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="Agentic Manufacturing Analyzer",
    page_icon="🏭",
    layout="wide",
)

# --- Session state initialisation -------------------------------------------

def _init_state():
    defaults = {
        "event_queue": queue.Queue(),
        "response_queue": queue.Queue(),
        "events": [],
        "running": False,
        "completed": False,
        "thread": None,
        "df_preview": None,
        "selected_dataset_path": None,
        "stages": {},     # tool -> {start, end, success, message}
        "turns": [],
        "metrics": {},
        "recommendations": [],
        "pending_prompt": None,
        "errors": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# --- Helpers ----------------------------------------------------------------

PRETTY_TOOL = {
    "load_and_inspect_data": "📥 1. 데이터 로드/검사",
    "preprocess_data": "🔧 2. 전처리/피처 엔지니어링",
    "analyze_data": "🤖 3. 모델링/평가",
    "generate_recommendations": "📋 4. 추천 생성",
}


def translate_stage_message(msg: str) -> str:
    """Translate common backend log strings to Korean."""
    if not msg:
        return ""
    m = msg
    # Shape: (N, K) → N행 K컬럼
    m = re.sub(r"Shape:\s*\((\d+),\s*(\d+)\)", lambda x: f"({x.group(1)}행 × {x.group(2)}컬럼)", m)
    replacements = [
        ("Data loaded.", "데이터 로드 완료"),
        ("Preprocessing complete.", "전처리 완료"),
        ("Dynamic analysis complete.", "모델 분석 완료"),
        ("Model:", "모델:"),
        ("Accuracy:", "정확도:"),
        ("Optimization completed with human review.", "최적화 완료 (사람 검토 포함)"),
        ("No actions to review.", "검토할 추천이 없습니다"),
    ]
    for eng, kor in replacements:
        m = m.replace(eng, kor)
    return m

def drain_events():
    """Pull all available events from the queue into session_state."""
    q = st.session_state.event_queue
    drained = 0
    while True:
        try:
            ev = q.get_nowait()
        except queue.Empty:
            break
        st.session_state.events.append(ev)
        _apply_event(ev)
        drained += 1
    return drained

def _apply_event(ev: Dict[str, Any]):
    t = ev.get("type")
    if t == "stage_start":
        tool = ev["tool"]
        st.session_state.stages[tool] = {"start": time.time(), "end": None, "success": None, "message": None}
    elif t == "stage_end":
        tool = ev["tool"]
        s = st.session_state.stages.setdefault(tool, {"start": time.time()})
        s["end"] = time.time()
        s["success"] = ev["success"]
        s["message"] = ev["message"]
    elif t == "turn":
        st.session_state.turns.append(ev)
    elif t == "model_metric":
        st.session_state.metrics["model"] = ev["model"]
        st.session_state.metrics.update(ev.get("metrics", {}))
    elif t == "anomalies":
        st.session_state.metrics["n_anomalies"] = ev["count"]
    elif t == "recommendations":
        st.session_state.recommendations = ev["rows"]
    elif t == "complete":
        st.session_state.running = False
        st.session_state.completed = True
        st.session_state.metrics.update(ev.get("metrics") or {})
        st.session_state.metrics["total_duration"] = ev.get("duration")
    elif t == "error":
        st.session_state.errors.append(ev)
        st.session_state.running = False
    elif t == "hitl_prompt":
        st.session_state.pending_prompt = ev


def reset_run():
    st.session_state.event_queue = queue.Queue()
    st.session_state.response_queue = queue.Queue()
    st.session_state.events = []
    st.session_state.running = False
    st.session_state.completed = False
    st.session_state.thread = None
    st.session_state.stages = {}
    st.session_state.turns = []
    st.session_state.metrics = {}
    st.session_state.recommendations = []
    st.session_state.pending_prompt = None
    st.session_state.errors = []


def discover_builtin_datasets() -> List[Path]:
    out: List[Path] = []
    data_dir = ROOT / "data"
    if data_dir.exists():
        for p in sorted(data_dir.rglob("*.csv")):
            out.append(p)
    return out

# --- Sidebar: configuration -------------------------------------------------

st.sidebar.title("⚙️ 설정")

# 1) Dataset selection
st.sidebar.subheader("1. 데이터셋")
builtin = discover_builtin_datasets()
builtin_labels = [str(p.relative_to(ROOT)) for p in builtin]

dataset_choice_mode = st.sidebar.radio(
    "데이터셋 출처",
    ["내장 데이터셋", "CSV 업로드"],
    horizontal=False,
    key="dataset_mode",
)

uploaded_path: Optional[str] = None
if dataset_choice_mode == "내장 데이터셋":
    if builtin_labels:
        idx = st.sidebar.selectbox("내장 CSV 선택", range(len(builtin_labels)), format_func=lambda i: builtin_labels[i])
        uploaded_path = str(builtin[idx])
    else:
        st.sidebar.warning("내장 데이터셋이 없습니다. data/ 아래에 CSV를 두거나 업로드하세요.")
else:
    upload = st.sidebar.file_uploader("CSV 파일 업로드", type=["csv"])
    if upload is not None:
        uploads_dir = ROOT / "uploads"
        uploads_dir.mkdir(exist_ok=True)
        target_path = uploads_dir / upload.name
        with open(target_path, "wb") as f:
            f.write(upload.getbuffer())
        uploaded_path = str(target_path)

# Lazy-load preview to populate target/feature dropdowns
if uploaded_path and uploaded_path != st.session_state.selected_dataset_path:
    try:
        st.session_state.df_preview = pd.read_csv(uploaded_path, nrows=200)
        st.session_state.selected_dataset_path = uploaded_path
    except Exception as e:
        st.sidebar.error(f"CSV 읽기 실패: {e}")

# 2) Problem & target
st.sidebar.subheader("2. 문제 정의")
problem_type_choice = st.sidebar.selectbox(
    "문제 유형",
    ["auto (타겟 dtype으로 자동 추론)", "classification", "regression", "anomaly_detection"],
    index=0,
    help="auto = schema_discovery 가 타겟 컬럼의 데이터 타입 보고 자동 결정. 그 외는 강제 지정.",
)
problem_type = None if problem_type_choice.startswith("auto") else problem_type_choice

target_col = None
feature_cols: List[str] = []
if st.session_state.df_preview is not None:
    cols = list(st.session_state.df_preview.columns)
    if problem_type != "anomaly_detection":
        # Suggest a sensible default target (last column)
        default_idx = len(cols) - 1
        target_col = st.sidebar.selectbox("타겟 컬럼", cols, index=default_idx)
    feature_options = [c for c in cols if c != target_col]
    feature_cols = st.sidebar.multiselect(
        "피처 컬럼 (비우면 자동: 타겟/ID 제외 전체)",
        feature_options,
        default=[],
    )

# 3) LLM backend (hybrid: Planner=강한 LLM + Decision=가벼운 SLM 가능)
st.sidebar.subheader("3. LLM 백엔드 (Hybrid)")
st.sidebar.caption("Planner=오케스트레이션, Decision=세부 결정. 다른 backend 조합으로 진정한 hybrid 가능.")

planner_backend = st.sidebar.selectbox(
    "Planner LLM (강함 권장)",
    ["gemini", "ollama", "mock"],
    index=0,
    help="다음 도구 선택 + 5단계 reasoning. 강한 모델 권장.",
)

decision_backend_choice = st.sidebar.selectbox(
    "Decision LLM (SLM 가능)",
    ["same as planner", "ollama", "mock"],
    index=0,
    help="'same as planner' = 같은 backend 재사용. 'ollama' = SLM 분리 (hybrid 모드).",
)
decision_backend = planner_backend if decision_backend_choice == "same as planner" else decision_backend_choice

planner_model = "qwen3:4b"
decision_model = "qwen3:4b"
if planner_backend == "ollama":
    planner_model = st.sidebar.text_input("Planner Ollama 모델", value="qwen3:4b")
if decision_backend == "ollama":
    if planner_backend == "ollama":
        decision_model = planner_model  # 같은 모델 재사용
    else:
        decision_model = st.sidebar.text_input("Decision Ollama 모델 (SLM)", value="qwen3:4b")

# 4) HITL behaviour
st.sidebar.subheader("4. HITL 정책")
auto_approve = st.sidebar.toggle("모든 HITL 자동 승인", value=False, help="끄면 화면에서 직접 클릭")

# Run button
st.sidebar.markdown("---")
can_run = (
    uploaded_path is not None
    and (problem_type == "anomaly_detection" or target_col is not None)
    and not st.session_state.running
)
run_clicked = st.sidebar.button("🚀 실행", type="primary", disabled=not can_run, width="stretch")
if st.sidebar.button("🔄 초기화", width="stretch"):
    reset_run()
    st.rerun()

# --- Main area --------------------------------------------------------------

st.title("🏭 Agentic Manufacturing Analyzer")
st.caption("LLM이 ML 파이프라인을 실시간으로 조립합니다 — 모든 결정이 화면에 보입니다")

# Dataset preview at top
if st.session_state.df_preview is not None:
    with st.expander(f"📊 데이터 미리보기 — {Path(uploaded_path).name}", expanded=not st.session_state.running):
        c1, c2, c3 = st.columns(3)
        c1.metric("행 수 (미리보기)", len(st.session_state.df_preview))
        c2.metric("컬럼 수", len(st.session_state.df_preview.columns))
        c3.metric("결측치 (미리보기)", int(st.session_state.df_preview.isnull().sum().sum()))
        st.dataframe(st.session_state.df_preview.head(10), width="stretch")

# Trigger run
if run_clicked:
    reset_run()
    st.session_state.running = True
    config = {
        "dataset_path": uploaded_path,
        "target_column": target_col,
        "feature_columns": feature_cols if feature_cols else None,
        "problem_type": problem_type,
        "planner_backend": planner_backend,
        "decision_backend": decision_backend,
        "planner_model": planner_model,
        "decision_model": decision_model,
        "auto_approve_hitl": auto_approve,
    }
    t = threading.Thread(
        target=run_workflow_in_thread,
        args=(config, st.session_state.event_queue, st.session_state.response_queue),
        daemon=True,
    )
    t.start()
    st.session_state.thread = t
    st.rerun()


# ===========================================================================
# Live panel — wrapped in a fragment so it can auto-refresh WITHOUT
# rerendering the whole page (preserves scroll position & avoids flicker).
# Layout is fully vertical so it stays usable on narrow monitors.
# ===========================================================================

STAGE_ORDER = ["load_and_inspect_data", "preprocess_data", "analyze_data", "generate_recommendations"]

# Korean explanations for known HITL prompts.
HITL_PROMPT_GUIDES = [
    {
        "match": "retry with alternative models",
        "title": "🤖 모델 성능이 낮습니다 — 재시도할까요?",
        "stage": "3단계: 모델링 & 평가",
        "why": (
            "방금 학습한 모델의 성능 지표(R² 또는 정확도)가 임계값(R²<0.1 또는 Accuracy<0.6)에 못 미칩니다. "
            "Adaptive Intelligence 기능으로 다른 후보 모델들(RandomForest, SVM 등)을 모두 돌려서 "
            "그 중 가장 좋은 모델을 자동 선택할 수 있습니다."
        ),
        "options": {
            "retry": ("🔁 다른 모델로 재시도", "여러 모델 후보를 모두 훈련시켜 best 선택 (수십 초 추가 소요)"),
            "proceed": ("➡️ 그대로 진행", "성능 낮아도 현재 모델 결과로 추천 생성"),
        },
    },
    {
        "match": "Approve these recommendations",
        "title": "📋 모델이 만든 정비 추천을 승인하시겠습니까?",
        "stage": "4단계: 추천 검토 (마지막)",
        "why": (
            "분석 결과를 바탕으로 OptimizationAgent가 우선순위가 높은 기계 30개에 대한 "
            "정비 액션 플랜을 만들었습니다. 사람이 최종 승인해야 추천이 채택됩니다 (HITL 게이트)."
        ),
        "options": {
            "approve": ("✅ 승인 (정상 종료)", "추천 그대로 채택하고 워크플로우 종료"),
            "modify": ("✏️ 수정", "원래 코드는 행 인덱스로 일부 제거하는 방식 — 현재 미구현"),
            "reject": ("❌ 거부", "추천 전부 폐기하고 워크플로우 종료"),
        },
    },
]


def _find_guide(message: str):
    if not message:
        return None
    for g in HITL_PROMPT_GUIDES:
        if g["match"].lower() in message.lower():
            return g
    return None


def _render_status_banner():
    """Slim one-line banner — running or completed."""
    if st.session_state.completed:
        m = st.session_state.metrics or {}
        parts = [f"✅ 완료 · 총 {m.get('total_duration', 0):.1f}s"]
        if m.get("model"): parts.append(f"모델 `{m['model']}`")
        if m.get("accuracy") is not None: parts.append(f"Accuracy `{m['accuracy']*100:.2f}%`")
        elif m.get("r2") is not None: parts.append(f"R² `{m['r2']:.4f}`")
        elif m.get("n_anomalies") is not None: parts.append(f"이상치 `{m['n_anomalies']}`")
        st.success(" · ".join(parts))
    elif st.session_state.running:
        st.info("▶ 워크플로우 실행 중... (자동 새로고침은 1.5초마다)")


def _render_timeline():
    st.subheader("🎬 워크플로우 단계")
    if not st.session_state.events and not st.session_state.running:
        st.info("좌측 사이드바에서 설정 후 **🚀 실행** 을 눌러주세요.")
        return
    for tool in STAGE_ORDER:
        s = st.session_state.stages.get(tool)
        label = PRETTY_TOOL.get(tool, tool)
        if s is None:
            st.markdown(f"⏳ &nbsp; **{label}** &nbsp; — _대기_")
        elif s.get("end") is None:
            st.markdown(f"▶ &nbsp; **{label}** &nbsp; — _실행 중..._")
        elif s.get("success"):
            dur = (s["end"] - s["start"]) if s.get("start") else 0
            msg = translate_stage_message(s.get("message", ""))
            st.markdown(f"✅ &nbsp; **{label}** &nbsp; · `{dur:.2f}s` &nbsp; — {msg}")
        else:
            msg = translate_stage_message(s.get("message", ""))
            st.markdown(f"❌ &nbsp; **{label}** &nbsp; — _실패_: {msg}")


def _build_stage_groups():
    """Walk the event stream and bucket each event under the workflow stage it belongs to."""
    stages = []          # ordered list of dicts
    pending_llm = None   # most recent LLM decision (precedes its stage_start)
    current = None
    for ev in st.session_state.events:
        t = ev.get("type")
        if t == "llm_decision":
            pending_llm = ev
        elif t == "stage_start":
            current = {
                "tool": ev.get("tool"),
                "llm": pending_llm,
                "tool_decisions": [],
                "metrics": [],
                "end": None,
            }
            stages.append(current)
            pending_llm = None
        elif t == "tool_decision" and current is not None:
            current["tool_decisions"].append(ev)
        elif t == "model_metric" and current is not None:
            current["metrics"].append(ev)
        elif t == "stage_end" and current is not None:
            current["end"] = ev
            current = None
    # Trailing llm decision after last stage end (e.g., finish=True)
    if pending_llm is not None:
        stages.append({
            "tool": pending_llm.get("tool"),
            "llm": pending_llm,
            "tool_decisions": [],
            "metrics": [],
            "end": None,
            "trailing": True,
        })
    return stages


def _render_decisions():
    st.subheader("💭 에이전트 의사결정 — 단계별 흐름")
    stages = _build_stage_groups()
    if not stages:
        st.caption("아직 결정 이벤트 없음")
        return

    for idx, s in enumerate(stages, start=1):
        tool = s.get("tool")
        pretty = PRETTY_TOOL.get(tool, tool or "—")
        end = s.get("end")
        if s.get("trailing"):
            status_chip = "🏁 종료"
            duration_chip = ""
        elif end is None:
            status_chip = "▶ 진행 중"
            duration_chip = ""
        elif end.get("success"):
            status_chip = "✅ 성공"
            # Duration via stages dict already tracked elsewhere; recompute from session_state.stages if present
            ss = st.session_state.stages.get(tool, {})
            if ss.get("end") and ss.get("start"):
                duration_chip = f" · `{(ss['end'] - ss['start']):.2f}s`"
            else:
                duration_chip = ""
        else:
            status_chip = "❌ 실패"
            duration_chip = ""

        # Header card per stage
        with st.container(border=True):
            st.markdown(f"### 단계 {idx} · {pretty} &nbsp; — &nbsp; {status_chip}{duration_chip}")

            # 1) LLM Planner decision (the "why this step")
            llm = s.get("llm")
            if llm:
                finish_emoji = "🏁" if llm.get("finish") else "🧠"
                finish_text = " (마지막 단계, 워크플로우 종료)" if llm.get("finish") else ""
                st.markdown(
                    f"**{finish_emoji} LLM Planner의 결정** — `{tool or llm.get('tool','—')}` 선택{finish_text}"
                )
                reason = llm.get("reason") or "(추론 없음)"
                st.markdown(f"> 💬 _{reason}_")

            # 2) ToolDecider tactical decisions (model, preprocessing strategy)
            tds = s.get("tool_decisions") or []
            if tds:
                st.markdown("**🛠 ToolDecider의 결정** (전술적)")
                for td in tds:
                    label_kor = {
                        "selected tool": "ML 모델 선택",
                        "chose preprocessing strategy": "전처리 전략 선택",
                        "selected": "도구 선택",
                    }.get(td.get("label", ""), td.get("label", ""))
                    st.markdown(f"- **{label_kor}**: `{td.get('value','')}`")

            # 3) Stage end message (data shape, accuracy, etc)
            if end and end.get("message"):
                st.markdown(f"**📤 결과**: {translate_stage_message(end['message'])}")

            # 4) Captured numerical metrics
            mts = s.get("metrics") or []
            if mts:
                last = mts[-1].get("metrics", {})
                parts = []
                if last.get("accuracy") is not None:
                    parts.append(f"Accuracy `{last['accuracy']*100:.2f}%`")
                if last.get("r2") is not None:
                    parts.append(f"R² `{last['r2']:.4f}`")
                if last.get("mse") is not None:
                    parts.append(f"MSE `{last['mse']:.4f}`")
                if parts:
                    st.markdown("**📈 메트릭**: " + " · ".join(parts))


def _render_metrics_inline():
    m = st.session_state.metrics or {}
    if not m or st.session_state.completed:
        return
    st.subheader("📈 모델 성능 (실시간)")
    parts = []
    if m.get("model"): parts.append(f"**모델**: `{m['model']}`")
    if m.get("accuracy") is not None: parts.append(f"**Accuracy**: `{m['accuracy']*100:.2f}%`")
    if m.get("r2") is not None: parts.append(f"**R²**: `{m['r2']:.4f}`")
    if m.get("mse") is not None: parts.append(f"**MSE**: `{m['mse']:.4f}`")
    if m.get("n_anomalies") is not None: parts.append(f"**이상치**: `{m['n_anomalies']}`")
    if parts:
        st.markdown(" · ".join(parts))


def _render_hitl_prompt():
    if st.session_state.pending_prompt is None:
        return
    p = st.session_state.pending_prompt
    guide = _find_guide(p.get("message", ""))
    st.markdown("---")
    st.warning("👤 **HITL — 사람의 결정이 필요합니다**")
    if guide:
        st.markdown(f"### {guide['title']}")
        st.caption(f"📍 단계: {guide['stage']}")
        st.info(f"**왜 묻고 있나?** {guide['why']}")
        with st.expander("원문 메시지 보기"):
            st.code(p.get("message", ""), language="text")
    else:
        st.markdown(f"### {p.get('message', '')}")

    if p.get("options"):
        if p.get("multi_select"):
            picks = st.multiselect("선택", p["options"], default=p["options"], key=f"hitl_{p['prompt_id']}")
            if st.button("제출", key=f"submit_{p['prompt_id']}", type="primary"):
                st.session_state.response_queue.put({"prompt_id": p["prompt_id"], "value": picks})
                st.session_state.pending_prompt = None
                st.rerun()
        else:
            for i, opt in enumerate(p["options"]):
                label = opt
                help_text = None
                btn_type = "secondary"
                if guide and opt in guide["options"]:
                    label, help_text = guide["options"][opt]
                    if opt in ("approve", "proceed"):
                        btn_type = "primary"
                if st.button(label, key=f"opt_{p['prompt_id']}_{i}", help=help_text, type=btn_type, width="stretch"):
                    st.session_state.response_queue.put({"prompt_id": p["prompt_id"], "value": opt})
                    st.session_state.pending_prompt = None
                    st.rerun()
    else:
        text = st.text_input("입력", key=f"hitl_text_{p['prompt_id']}")
        if st.button("제출", key=f"submit_text_{p['prompt_id']}", type="primary"):
            st.session_state.response_queue.put({"prompt_id": p["prompt_id"], "value": text})
            st.session_state.pending_prompt = None
            st.rerun()


def _render_recommendations_at_bottom():
    """추천 액션 — 워크플로우의 마지막 산출물. 페이지 맨 아래에 표시."""
    if not st.session_state.recommendations:
        return
    st.markdown("---")
    st.subheader(f"📋 최종 산출물 — 추천 액션 (Top {len(st.session_state.recommendations)})")
    st.caption("워크플로우 4단계 (추천 생성)에서 OptimizationAgent가 만든 정비 액션 플랜입니다.")
    rec_df = pd.DataFrame(st.session_state.recommendations)
    st.dataframe(rec_df, width="stretch", height=420)
    csv = rec_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ CSV 다운로드", csv, file_name="recommendations.csv", mime="text/csv")


def _render_logs():
    with st.expander(f"📜 전체 로그 ({len(st.session_state.events)} events)", expanded=False):
        log_lines = []
        for e in st.session_state.events[-300:]:
            if e.get("type") == "log":
                log_lines.append(f"[{e.get('level', 'INFO')}] {e.get('message', '')}")
            else:
                log_lines.append(f"[EVENT:{e.get('type')}] {e}")
        st.code("\n".join(log_lines) if log_lines else "(empty)", language="text")


def _render_errors():
    if not st.session_state.errors:
        return
    st.markdown("---")
    for err in st.session_state.errors:
        st.error(f"❌ {err.get('message')}")
        if err.get("traceback"):
            with st.expander("Traceback"):
                st.code(err["traceback"], language="text")


# ---------------------------------------------------------------------------
# Linear page render — plain top-to-bottom, no fragments.
# Order matches workflow execution order so user just scrolls naturally.
# ---------------------------------------------------------------------------

drain_events()

_render_status_banner()
_render_hitl_prompt()
st.markdown("---")
_render_timeline()
st.markdown("---")
_render_decisions()
_render_metrics_inline()
_render_errors()
_render_logs()
_render_recommendations_at_bottom()  # ← 마지막 단계의 산출물이므로 맨 아래

# Auto-refresh ONLY while actively running.
# When complete, page is static → user can scroll freely without jumps.
if st.session_state.running and not st.session_state.completed:
    time.sleep(1.5)
    st.rerun()

# (모든 동적 렌더링은 위쪽 live_panel() fragment 안에서 처리됩니다.)
