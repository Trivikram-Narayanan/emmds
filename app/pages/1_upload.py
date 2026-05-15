"""EMMDS — Upload page"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Upload | EMMDS", page_icon="📁", layout="wide")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*,[class*="css"]{font-family:'Inter',sans-serif!important}
.stApp{background:#05090f}
[data-testid="stSidebar"]{background:#080f1a!important;border-right:1px solid #0f2340}
[data-testid="metric-container"]{background:#0b1628;border:1px solid #0f2340;border-radius:12px;padding:.9rem 1.1rem!important}
[data-testid="metric-container"] label{color:#475569!important;font-size:.72rem!important;text-transform:uppercase;letter-spacing:.08em;font-weight:600}
[data-testid="metric-container"] [data-testid="metric-value"]{color:#f1f5f9!important;font-size:1.5rem!important;font-weight:700!important}
.stButton>button{background:#0f2040;color:#60a5fa;border:1px solid #1e3a5f;border-radius:9px;font-weight:500;transition:all .2s}
.stButton>button:hover{background:#162f55;border-color:#3b82f6;transform:translateY(-1px)}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#1d4ed8,#7c3aed);border:none;color:#fff;font-weight:600}
.stTabs [data-baseweb="tab-list"]{background:#0b1628;border-radius:10px;padding:.25rem;border:1px solid #0f2340}
.stTabs [data-baseweb="tab"]{background:transparent;color:#475569;border-radius:8px;font-size:.84rem;font-weight:500;padding:.45rem 1rem;border:none}
.stTabs [aria-selected="true"]{background:#112240!important;color:#60a5fa!important}
[data-testid="stDataFrame"]{border:1px solid #0f2340;border-radius:10px}
hr{border-color:#0f2340!important;margin:1.5rem 0!important}
.stSuccess{background:#042010!important;border-left:3px solid #22c55e!important;border-radius:8px!important}
.stWarning{background:#15100!important;border-left:3px solid #f59e0b!important;border-radius:8px!important}
.stError{background:#130805!important;border-left:3px solid #ef4444!important;border-radius:8px!important}
.stInfo{background:#060f1f!important;border-left:3px solid #3b82f6!important;border-radius:8px!important}
[data-testid="stFileUploader"]{background:#0b1628;border:2px dashed #0f2340;border-radius:12px}
</style>""", unsafe_allow_html=True)

for key in ["df","filename","pipeline_result","target_col","analysis","profile","validation"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ── Header ────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:2rem;">
  <div style="display:flex;align-items:center;gap:.75rem;margin-bottom:.4rem;">
    <div style="width:3rem;height:3rem;background:linear-gradient(135deg,#1d4ed820,#7c3aed20);
                border:1px solid #1e3a5f;border-radius:12px;display:flex;align-items:center;
                justify-content:center;font-size:1.4rem;">📁</div>
    <div>
      <div style="font-size:1.6rem;font-weight:700;color:#f1f5f9;letter-spacing:-.02em;">Upload Dataset</div>
      <div style="font-size:.85rem;color:#334155;">Upload a CSV or pick a built-in sample to get started</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([3, 1], gap="large")

with col_right:
    # ── Sample datasets ───────────────────────────────────────────────
    st.markdown("""
    <div style="background:#080f1a;border:1px solid #0f2340;border-radius:14px;padding:1.3rem;margin-bottom:1rem;">
      <div style="font-size:.72rem;text-transform:uppercase;letter-spacing:.1em;color:#334155;font-weight:600;margin-bottom:.9rem;">Quick Start Samples</div>
    </div>""", unsafe_allow_html=True)

    def load_sklearn(name, label):
        from sklearn import datasets as skd
        loaders = {"breast_cancer": skd.load_breast_cancer,
                   "iris": skd.load_iris, "wine": skd.load_wine,
                   "diabetes": skd.load_diabetes}
        data = loaders[name](as_frame=True)
        df   = data.frame.copy()
        df["target"] = data.target
        st.session_state.df = df
        st.session_state.filename = f"{name}.csv"
        st.session_state.pipeline_result = None
        st.session_state.target_col = "target"

    for name, icon, label, desc in [
        ("breast_cancer", "🩺", "Breast Cancer", "569 rows · 30 features · binary"),
        ("iris",          "🌸", "Iris",          "150 rows · 4 features · 3-class"),
        ("wine",          "🍷", "Wine",          "178 rows · 13 features · 3-class"),
        ("diabetes",      "💊", "Diabetes",      "442 rows · 10 features · regression"),
    ]:
        with st.container():
            st.markdown(f"""
            <div style="background:#0b1628;border:1px solid #0f2340;border-radius:10px;
                        padding:.85rem 1rem;margin-bottom:.5rem;">
              <div style="color:#e2e8f0;font-weight:500;font-size:.88rem;">{icon} {label}</div>
              <div style="color:#334155;font-size:.76rem;margin-top:.15rem;">{desc}</div>
            </div>""", unsafe_allow_html=True)
            if st.button(f"Load {label}", key=f"load_{name}", use_container_width=True):
                load_sklearn(name, label)
                st.rerun()

    st.markdown("""
    <div style="background:#080f1a;border:1px solid #0f2340;border-radius:14px;padding:1.3rem;margin-top:1rem;">
      <div style="font-size:.72rem;text-transform:uppercase;letter-spacing:.1em;color:#334155;font-weight:600;margin-bottom:.8rem;">File Requirements</div>
      <div style="font-size:.8rem;color:#334155;line-height:2;">
        ✓ CSV, TSV, Excel, JSON, Parquet<br>
        ✓ Header row required<br>
        ✓ Min 10 rows recommended<br>
        ✓ Mixed numerical + categorical OK<br>
        ✓ Classification or regression
      </div>
    </div>""", unsafe_allow_html=True)

with col_left:
    uploaded = st.file_uploader(
        "Drop your dataset here",
        type=["csv", "tsv", "xlsx", "xls", "json", "parquet"],
        help="Supports CSV · TSV · Excel · JSON · Parquet"
    )
    if uploaded:
        try:
            ext = Path(uploaded.name).suffix.lower()
            if ext == ".csv":
                df = pd.read_csv(uploaded)
            elif ext == ".tsv":
                df = pd.read_csv(uploaded, sep="\t")
            elif ext in (".xlsx", ".xls"):
                df = pd.read_excel(uploaded)
            elif ext == ".json":
                df = pd.read_json(uploaded)
            elif ext == ".parquet":
                df = pd.read_parquet(uploaded)
            else:
                df = pd.read_csv(uploaded)
            st.session_state.df = df
            st.session_state.filename = uploaded.name
            st.session_state.pipeline_result = None
        except Exception as e:
            st.error(f"Could not parse file: {e}")

    if st.session_state.df is not None:
        df = st.session_state.df

        # ── Banner ────────────────────────────────────────────────────
        st.markdown(f"""
        <div style="background:linear-gradient(90deg,#042010,#041a18);border:1px solid #166534;
                    border-radius:12px;padding:1rem 1.3rem;margin:.8rem 0;
                    display:flex;align-items:center;gap:.8rem;">
          <div style="font-size:1.4rem;">✅</div>
          <div>
            <div style="color:#4ade80;font-weight:600;font-size:.92rem;">{st.session_state.filename}</div>
            <div style="color:#334155;font-size:.78rem;">Loaded successfully — ready to analyse</div>
          </div>
        </div>""", unsafe_allow_html=True)

        # ── Metrics ───────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows",       f"{df.shape[0]:,}")
        m2.metric("Columns",    f"{df.shape[1]}")
        m3.metric("Missing",    f"{df.isnull().sum().sum():,}")
        m4.metric("Duplicates", f"{df.duplicated().sum():,}")

        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

        tab_prev, tab_info, tab_stats = st.tabs(["👀 Preview", "📋 Column Info", "📊 Quick Stats"])

        with tab_prev:
            st.dataframe(
                df.head(15).style.set_properties(**{"background-color": "#0b1628", "color": "#e2e8f0"}),
                use_container_width=True, height=300
            )

        with tab_info:
            info_df = pd.DataFrame({
                "Column":    df.columns,
                "Type":      df.dtypes.astype(str).values,
                "Non-Null":  df.notnull().sum().values,
                "Unique":    df.nunique().values,
                "Missing %": (df.isnull().mean() * 100).round(1).values,
            })
            st.dataframe(info_df, use_container_width=True, height=320)

        with tab_stats:
            num_df = df.select_dtypes(include=["number"])
            if not num_df.empty:
                st.dataframe(num_df.describe().round(3), use_container_width=True)
            else:
                st.info("No numerical columns detected.")

        st.markdown("---")
        st.success("✅ Dataset ready! Go to **📊 Analysis** to choose your target column.")

    else:
        st.markdown("""
        <div style="border:2px dashed #0f2340;border-radius:16px;padding:4rem 2rem;
                    text-align:center;margin-top:1.5rem;background:#080f1a;">
          <div style="font-size:3rem;margin-bottom:.8rem;opacity:.4;">📂</div>
          <div style="color:#1e3a5f;font-size:.95rem;font-weight:500;">
            Upload a CSV above or pick a sample dataset →
          </div>
          <div style="color:#0f2040;font-size:.82rem;margin-top:.4rem;">
            Supports CSV · TSV · Excel · JSON · Parquet
          </div>
        </div>""", unsafe_allow_html=True)
