"""EMMDS — Home page"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import streamlit as st

st.set_page_config(page_title="EMMDS", page_icon="🧠", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
code, pre, .stCode          { font-family: 'JetBrains Mono', monospace !important; }

/* ── Background ─────────────────────────────────────────────── */
.stApp                      { background: #05090f; }
.block-container            { padding: 1.5rem 2rem 3rem; max-width: 1280px; }

/* ── Sidebar ─────────────────────────────────────────────────── */
[data-testid="stSidebar"]               { background: #080f1a !important; border-right: 1px solid #0f2340; }
[data-testid="stSidebar"] *             { color: #94a3b8; }
[data-testid="stSidebar"] strong        { color: #e2e8f0; }

/* ── Metrics ─────────────────────────────────────────────────── */
[data-testid="metric-container"]        { background: #0b1628; border: 1px solid #0f2340; border-radius: 12px; padding: .9rem 1.1rem !important; }
[data-testid="metric-container"] label  { color: #475569 !important; font-size: .72rem !important; text-transform: uppercase; letter-spacing: .08em; font-weight: 600; }
[data-testid="metric-container"] [data-testid="metric-value"] { color: #f1f5f9 !important; font-size: 1.6rem !important; font-weight: 700 !important; }
[data-testid="metric-container"] [data-testid="metric-delta"] { font-size: .78rem !important; }

/* ── Tabs ────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"]   { background: #0b1628; border-radius: 10px; padding: .25rem; gap: .15rem; border: 1px solid #0f2340; }
.stTabs [data-baseweb="tab"]        { background: transparent; color: #475569; border-radius: 8px; font-size: .84rem; font-weight: 500; padding: .45rem 1rem; border: none; }
.stTabs [aria-selected="true"]      { background: #112240 !important; color: #60a5fa !important; }

/* ── Buttons ─────────────────────────────────────────────────── */
.stButton > button                  { background: #0f2040; color: #60a5fa; border: 1px solid #1e3a5f; border-radius: 9px; font-weight: 500; font-size: .875rem; padding: .5rem 1.2rem; transition: all .2s ease; }
.stButton > button:hover            { background: #162f55; border-color: #3b82f6; color: #93c5fd; transform: translateY(-1px); }
.stButton > button[kind="primary"]  { background: linear-gradient(135deg, #1d4ed8 0%, #7c3aed 100%); border: none; color: #fff; font-weight: 600; box-shadow: 0 4px 15px #3b82f620; }
.stButton > button[kind="primary"]:hover { opacity: .92; transform: translateY(-2px); box-shadow: 0 8px 25px #7c3aed35; }

/* ── Progress ────────────────────────────────────────────────── */
.stProgress > div > div             { background: linear-gradient(90deg, #3b82f6, #8b5cf6, #06b6d4); border-radius: 4px; }

/* ── Inputs ──────────────────────────────────────────────────── */
[data-baseweb="select"] > div       { background: #0b1628 !important; border-color: #0f2340 !important; border-radius: 8px !important; color: #e2e8f0 !important; }
[data-baseweb="input"] > div        { background: #0b1628 !important; border-color: #0f2340 !important; border-radius: 8px !important; }
[data-testid="stFileUploader"]      { background: #0b1628; border: 2px dashed #1e3a5f; border-radius: 12px; padding: 1rem; }

/* ── Alerts ──────────────────────────────────────────────────── */
.stSuccess  { background: #042010 !important; border: 1px solid #16613620 !important; border-left: 3px solid #22c55e !important; border-radius: 8px !important; }
.stWarning  { background: #15100 !important; border: 1px solid #78350f20 !important; border-left: 3px solid #f59e0b !important; border-radius: 8px !important; }
.stError    { background: #130805 !important; border: 1px solid #7f1d1d20 !important; border-left: 3px solid #ef4444 !important; border-radius: 8px !important; }
.stInfo     { background: #060f1f !important; border: 1px solid #1e3a5f20 !important; border-left: 3px solid #3b82f6 !important; border-radius: 8px !important; }

/* ── Expander ────────────────────────────────────────────────── */
[data-testid="stExpander"]          { background: #0b1628; border: 1px solid #0f2340; border-radius: 10px; }
[data-testid="stExpander"] summary  { color: #94a3b8 !important; }

/* ── DataFrames ──────────────────────────────────────────────── */
[data-testid="stDataFrame"]         { border: 1px solid #0f2340; border-radius: 10px; }
.stDataFrame thead th               { background: #0f2340 !important; color: #60a5fa !important; }

/* ── Checkbox ────────────────────────────────────────────────── */
[data-testid="stCheckbox"] label    { color: #94a3b8 !important; }

/* ── Download btn ────────────────────────────────────────────── */
.stDownloadButton > button          { background: #041f12; color: #4ade80; border: 1px solid #14532d; border-radius: 9px; }

/* ── Divider ─────────────────────────────────────────────────── */
hr { border-color: #0f2340 !important; margin: 1.8rem 0 !important; }

/* ── Nav pills ───────────────────────────────────────────────── */
.pill { display:inline-block; background:#0f2040; border:1px solid #1e3a5f; border-radius:20px; padding:.2rem .75rem; font-size:.78rem; color:#60a5fa; margin:.15rem; }

/* ── Step badge ──────────────────────────────────────────────── */
.step-badge { display:inline-flex; align-items:center; justify-content:center; width:1.9rem; height:1.9rem; background:linear-gradient(135deg,#1d4ed8,#7c3aed); border-radius:50%; color:#fff; font-weight:700; font-size:.85rem; margin-right:.5rem; flex-shrink:0; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:.5rem 0 1rem;">
      <div style="font-size:1.3rem;font-weight:800;background:linear-gradient(90deg,#60a5fa,#a78bfa);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;">🧠 EMMDS</div>
      <div style="font-size:.75rem;color:#334155;margin-top:.2rem;">Explainable AI Decision System</div>
    </div>
    """, unsafe_allow_html=True)

    has_data   = "df" in st.session_state and st.session_state.df is not None
    has_result = "pipeline_result" in st.session_state and st.session_state.pipeline_result

    def step_row(num, label, done):
        color = "#22c55e" if done else "#1e3a5f"
        icon  = "✓" if done else num
        st.markdown(f"""
        <div style="display:flex;align-items:center;padding:.35rem .5rem;border-radius:8px;
                    margin:.15rem 0;{'background:#042010' if done else ''}">
          <span style="display:inline-flex;align-items:center;justify-content:center;
                       width:1.5rem;height:1.5rem;border-radius:50%;background:{color};
                       color:{'#fff' if done else '#334155'};font-size:.72rem;font-weight:700;
                       margin-right:.6rem;flex-shrink:0;">{icon}</span>
          <span style="font-size:.83rem;color:{'#4ade80' if done else '#334155'};
                       font-weight:{'600' if done else '400'};">{label}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;color:#1e3a5f;margin-bottom:.4rem;'>Workflow</div>", unsafe_allow_html=True)
    step_row("1", "Upload Dataset",   has_data)
    step_row("2", "Analyse Data",     bool(st.session_state.get("analysis")))
    step_row("3", "Train Pipeline",   has_result)
    step_row("4", "View Results",     has_result)

    st.markdown("<hr style='border-color:#0f2340;margin:.8rem 0;'>", unsafe_allow_html=True)

    if has_data:
        df = st.session_state.df
        st.markdown(f"""
        <div style="background:#0b1628;border:1px solid #0f2340;border-radius:9px;padding:.8rem;">
          <div style="font-size:.72rem;color:#334155;text-transform:uppercase;letter-spacing:.08em;">Dataset</div>
          <div style="color:#93c5fd;font-size:.82rem;font-weight:500;margin:.25rem 0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{st.session_state.get('filename','—')}</div>
          <div style="font-size:.78rem;color:#475569;">{df.shape[0]:,} rows · {df.shape[1]} cols</div>
        </div>""", unsafe_allow_html=True)

    if has_result:
        d = st.session_state.pipeline_result.get("decision", {})
        ts = d.get("trust_score", 0)
        col = "#22c55e" if ts >= 0.7 else "#f59e0b" if ts >= 0.5 else "#ef4444"
        st.markdown(f"""
        <div style="background:#0b1628;border:1px solid #0f2340;border-radius:9px;padding:.8rem;margin-top:.5rem;">
          <div style="font-size:.72rem;color:#334155;text-transform:uppercase;letter-spacing:.08em;">Last Result</div>
          <div style="color:#e2e8f0;font-size:.85rem;font-weight:600;margin:.25rem 0;">{d.get('best_model','—').replace('_',' ').title()}</div>
          <div style="font-size:.78rem;color:{col};">Trust {ts:.3f} · {d.get('trust_label','')}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='position:absolute;bottom:1rem;font-size:.7rem;color:#1e3a5f;'>v2.0 · sklearn · SHAP · LIME</div>", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#060e1f 0%,#0a0f1e 50%,#060e1f 100%);
            border:1px solid #0f2340;border-radius:20px;padding:3.5rem 3rem 3rem;
            margin-bottom:2rem;position:relative;overflow:hidden;">
  <div style="position:absolute;top:-80px;right:-80px;width:300px;height:300px;
              background:radial-gradient(circle,#3b82f618,transparent 65%);border-radius:50%;pointer-events:none;"></div>
  <div style="position:absolute;bottom:-60px;left:20%;width:200px;height:200px;
              background:radial-gradient(circle,#8b5cf615,transparent 65%);border-radius:50%;pointer-events:none;"></div>
  <div style="position:absolute;top:30%;left:-50px;width:150px;height:150px;
              background:radial-gradient(circle,#06b6d410,transparent 65%);border-radius:50%;pointer-events:none;"></div>
  <div style="position:relative;z-index:1;">
    <div style="font-size:3rem;font-weight:800;line-height:1.1;letter-spacing:-.03em;
                background:linear-gradient(90deg,#60a5fa 0%,#a78bfa 50%,#34d399 100%);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.6rem;">
      Explainable AI<br>Multi-Model Decision System
    </div>
    <div style="font-size:1.05rem;color:#475569;margin-bottom:1.6rem;max-width:560px;line-height:1.6;">
      Upload any dataset. EMMDS trains every model, scores them on trust — not just accuracy —
      and explains why the best one was chosen.
    </div>
    <div>
      <span class="pill">🤖 AutoML</span>
      <span class="pill">🛡️ 5-Component Trust</span>
      <span class="pill">🔍 SHAP + LIME</span>
      <span class="pill">🤝 Model Agreement</span>
      <span class="pill">⚡ Parallel Training</span>
      <span class="pill">📊 9-Stage Pipeline</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Feature cards ─────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, (icon, title, desc, accent) in zip([c1,c2,c3,c4], [
    ("🧠", "Dataset Intelligence",  "Auto-detects task type, feature types, data quality score, drift, and meta-features before a single model trains.", "#3b82f6"),
    ("⚔️", "Smart Selection",       "Heuristic recommender excludes unsuitable models upfront — no wasted compute on the wrong algorithms.",           "#8b5cf6"),
    ("🔥", "Full Explainability",   "SHAP global importance + LIME local explanations on every prediction. You always know why.",                    "#f59e0b"),
    ("💡", "Trust — not Accuracy",  "5-component composite: Accuracy · Calibration · Stability · Agreement · Data Quality. Built for production.",   "#22c55e"),
]):
    with col:
        st.markdown(f"""
        <div style="background:#080f1a;border:1px solid #0f2340;border-radius:14px;
                    padding:1.5rem 1.3rem;height:100%;transition:border-color .2s;
                    border-top:3px solid {accent}30;">
          <div style="font-size:1.7rem;margin-bottom:.7rem;">{icon}</div>
          <div style="color:#e2e8f0;font-weight:600;font-size:.95rem;margin-bottom:.5rem;">{title}</div>
          <div style="color:#334155;font-size:.82rem;line-height:1.65;">{desc}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Steps ─────────────────────────────────────────────────────────────
st.markdown("<div style='font-size:.8rem;text-transform:uppercase;letter-spacing:.1em;color:#334155;margin-bottom:1rem;font-weight:600;'>Get started in 4 steps</div>", unsafe_allow_html=True)
sc = st.columns(4)
for col, (num, title, desc) in zip(sc, [
    ("1", "📁 Upload",  "Drop a CSV or pick a built-in sklearn dataset to begin."),
    ("2", "📊 Analyse", "Select your target column. EMMDS profiles the data instantly."),
    ("3", "🏋️ Train",   "Click Start — all 9 pipeline stages run automatically."),
    ("4", "🏆 Results", "Decision card, trust breakdown, SHAP plots, and a prediction playground."),
]):
    with col:
        st.markdown(f"""
        <div style="background:#080f1a;border:1px solid #0f2340;border-radius:12px;padding:1.3rem;">
          <div style="display:flex;align-items:center;margin-bottom:.6rem;">
            <span class="step-badge">{num}</span>
            <span style="color:#e2e8f0;font-weight:600;font-size:.9rem;">{title}</span>
          </div>
          <div style="color:#334155;font-size:.82rem;line-height:1.55;">{desc}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Pipeline accordion ────────────────────────────────────────────────
with st.expander("🔬 View the full 9-stage pipeline"):
    stages = [
        ("Validation",           "#3b82f6", "Data integrity: missing targets, class counts, infinities, type issues."),
        ("Analysis",             "#8b5cf6", "Task detection, profiling, meta-feature extraction, data quality scoring."),
        ("Model Recommendation", "#f59e0b", "Heuristic rules exclude unsuitable models before any training begins."),
        ("Parallel Training",    "#ef4444", "All recommended models trained simultaneously via joblib."),
        ("Cross-Validation",     "#22c55e", "5-fold stratified CV per model — zero data leakage."),
        ("Calibration",          "#06b6d4", "CalibratedClassifierCV for reliable probability estimates."),
        ("Explainability",       "#f472b6", "SHAP global importance + LIME local instance explanations."),
        ("Model Agreement",      "#a78bfa", "Global, pairwise, and entropy-based consensus across all models."),
        ("Trust + Decision",     "#34d399", "5-component trust score → ranking → final model selection."),
    ]
    for i, (name, color, desc) in enumerate(stages):
        st.markdown(f"""
        <div style="display:flex;align-items:flex-start;gap:.9rem;padding:.6rem 0;
                    border-bottom:1px solid #0a1828;{'border-bottom:none' if i==len(stages)-1 else ''}">
          <div style="min-width:2rem;height:2rem;background:{color}18;border:1px solid {color}40;
                      border-radius:8px;display:flex;align-items:center;justify-content:center;
                      color:{color};font-size:.78rem;font-weight:700;flex-shrink:0;">{i+1}</div>
          <div>
            <div style="color:#e2e8f0;font-size:.85rem;font-weight:600;margin-bottom:.2rem;">{name}</div>
            <div style="color:#334155;font-size:.8rem;line-height:1.5;">{desc}</div>
          </div>
        </div>""", unsafe_allow_html=True)
