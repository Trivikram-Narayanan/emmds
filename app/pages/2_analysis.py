"""EMMDS Page 2 — Dataset Analysis (polished v3)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Analysis | EMMDS", page_icon="📊", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');
html,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:#070c18}
[data-testid="stSidebar"]{background:#0d1220!important;border-right:1px solid #1e2d45}
[data-testid="metric-container"]{background:#0f1929;border:1px solid #1e2d45;border-radius:10px}
.stButton>button{background:#1e3a5f;color:#7dd3fc;border:1px solid #2d6a9f;border-radius:8px;font-weight:500}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#1e40af,#7c3aed);border:none;color:#fff;font-weight:600}
.stTabs [data-baseweb="tab-list"]{background:#0f1929;border-radius:8px;padding:.2rem}
.stTabs [aria-selected="true"]{background:#1e3a5f!important;color:#7dd3fc!important}
hr{border-color:#1e2d45!important}
.stSuccess{background:#052e16!important;border-left:3px solid #4ade80!important}
.stWarning{background:#1c1003!important;border-left:3px solid #fbbf24!important}
.stError{background:#1c0505!important;border-left:3px solid #f87171!important}
</style>""", unsafe_allow_html=True)

DARK = dict(plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e2e8f0",
            margin=dict(l=10,r=10,t=45,b=10))

if "df" not in st.session_state or st.session_state.df is None:
    st.warning("⚠️ No dataset loaded. Go to **📁 Upload** first.")
    st.stop()

df = st.session_state.df

st.markdown("""
<div style="margin-bottom:1.5rem;">
  <h1 style="color:#e2e8f0;margin:0;">📊 Dataset Analysis</h1>
  <p style="color:#64748b;margin:.3rem 0 0;">Explore your data before training. EMMDS auto-profiles everything.</p>
</div>""", unsafe_allow_html=True)

# ── Target selector + run ─────────────────────────────────────────────
c_sel, c_btn = st.columns([3, 1])
with c_sel:
    default_idx = len(df.columns) - 1
    target_col = st.selectbox("🎯 Select target column", df.columns, index=default_idx)
with c_btn:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    run = st.button("🔍 Analyse", type="primary", use_container_width=True)

if run:
    with st.spinner("Analysing…"):
        from src.data_engine.validator import DataValidator
        from src.data_engine.analyzer import DataAnalyzer
        from src.data_engine.profiler import DataProfiler
        from src.data_engine.data_quality import DataQualityScorer
        from src.data_engine.meta_features import MetaFeatureExtractor

        st.session_state.validation = DataValidator().validate(df, target_col)
        st.session_state.analysis   = DataAnalyzer().analyze(df, target_col)
        st.session_state.profile    = DataProfiler().profile_dataframe(df, target_col)
        dq = DataQualityScorer(); dq.score_dataset(df, target_col)
        st.session_state.dq_report  = dq.get_breakdown()
        meta = MetaFeatureExtractor(); meta.extract(df, target_col)
        st.session_state.meta       = meta.get_meta()
        st.session_state.target_col = target_col

if "analysis" not in st.session_state or st.session_state.analysis is None:
    st.info("Select your target column above and click **🔍 Analyse**.")
    st.stop()

analysis = st.session_state.analysis
profile  = st.session_state.profile
val      = st.session_state.validation
dq_info  = st.session_state.get("dq_report", {})
meta     = st.session_state.get("meta", {})

# ── Validation banner ─────────────────────────────────────────────────
if val["passed"]:
    st.success(val["summary"])
else:
    st.error(val["summary"])
    for e in val["errors"]: st.error(f"  • {e}")
for w in val.get("warnings", []): st.warning(f"  ⚠️ {w}")

st.markdown("---")

# ── Overview row ──────────────────────────────────────────────────────
c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Task",        analysis["task"].title())
c2.metric("Rows",        f"{analysis['rows']:,}")
c3.metric("Features",    analysis["feature_count"])
c4.metric("Numerical",   analysis["feature_types"]["numerical_count"])
c5.metric("Categorical", analysis["feature_types"]["categorical_count"])
c6.metric("Quality",     f"{dq_info.get('quality_score',0):.2f}  {dq_info.get('label','')[:2]}")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Distributions", "🔗 Correlations", "⚠️ Issues", "🧬 Meta-Features", "🗂️ Data Quality"
])

num_cols = analysis["feature_types"]["numerical"]
y_col    = st.session_state.target_col

with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        sel = st.selectbox("Feature", num_cols) if num_cols else None
        if sel:
            fig = px.histogram(df, x=sel, nbins=40, marginal="box",
                               color_discrete_sequence=["#818cf8"],
                               title=f"Distribution — {sel}")
            fig.update_layout(**DARK); st.plotly_chart(fig, use_container_width=True)
    with col_b:
        y = df[y_col]
        if analysis["task"] == "classification":
            vc = y.value_counts().reset_index(); vc.columns = ["Class","Count"]
            fig = px.bar(vc, x="Class", y="Count",
                         color="Count", color_continuous_scale="Viridis",
                         title=f"Target Distribution (imbalance={analysis['imbalance_ratio']})")
        else:
            fig = px.histogram(df, x=y_col, nbins=40, color_discrete_sequence=["#34d399"],
                               title="Target Distribution")
        fig.update_layout(**DARK); st.plotly_chart(fig, use_container_width=True)

with tab2:
    top20 = num_cols[:20]
    if len(top20) > 1:
        corr = df[top20].corr().round(3)
        fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="Correlation Matrix", aspect="auto")
        fig.update_layout(**DARK); st.plotly_chart(fig, use_container_width=True)
        high = profile.get("high_correlation_pairs", [])
        if high:
            st.warning(f"{len(high)} highly correlated pairs (|r| > 0.85):")
            st.dataframe(pd.DataFrame(high), use_container_width=True)
    else:
        st.info("Not enough numerical features for a correlation matrix.")

with tab3:
    c_miss, c_skew = st.columns(2)
    with c_miss:
        miss = analysis["missing"]
        if miss["has_missing"]:
            miss_df = pd.DataFrame([{"Column": c, "Missing %": v["percent"]}
                                     for c,v in miss["columns_with_missing"].items()])
            fig = px.bar(miss_df.sort_values("Missing %"), x="Missing %", y="Column",
                         orientation="h", color="Missing %", color_continuous_scale="Reds",
                         title="Missing Values")
            fig.update_layout(**DARK, height=max(250,len(miss_df)*30+80))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("🎉 No missing values!")
    with c_skew:
        skewed = profile.get("skewed_features", [])
        if skewed:
            sk_df = pd.DataFrame(skewed[:15])
            fig = px.bar(sk_df.sort_values("skewness"), x="skewness", y="feature",
                         orientation="h", color="skewness", color_continuous_scale="Oranges",
                         title="Skewed Features (|skew| > 1)")
            fig.update_layout(**DARK, height=max(250,len(sk_df)*28+80))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No highly skewed features.")

with tab4:
    if meta:
        mc1,mc2 = st.columns(2)
        with mc1:
            st.markdown("**Dataset Characteristics**")
            items = [("Samples",f"{meta.get('n_samples',0):,}"),
                     ("Features",meta.get('n_features')),
                     ("Numerical",meta.get('n_numerical')),
                     ("Categorical",meta.get('n_categorical')),
                     ("Missing ratio",f"{meta.get('missing_ratio',0):.3f}"),
                     ("Imbalance ratio",meta.get('imbalance_ratio')),
                     ("Dimensionality p/n",f"{meta.get('dimensionality_ratio',0):.4f}"),]
            for k,v in items:
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;padding:.3rem .5rem;
                            border-bottom:1px solid #1e2d45;">
                  <span style="color:#94a3b8;font-size:.85rem;">{k}</span>
                  <span style="color:#e2e8f0;font-weight:600;font-size:.85rem;">{v}</span>
                </div>""", unsafe_allow_html=True)
        with mc2:
            st.markdown("**Statistical Properties**")
            items2 = [("Avg abs correlation",f"{meta.get('avg_abs_correlation',0):.3f}"),
                      ("Max correlation",f"{meta.get('max_correlation',0):.3f}"),
                      ("Mean skewness",f"{meta.get('mean_skewness',0):.3f}"),
                      ("Skewed feature ratio",f"{meta.get('skewed_feature_ratio',0):.3f}"),
                      ("Noise estimate",f"{meta.get('noise_estimate',0):.3f}"),
                      ("High cardinality cats",str(meta.get('has_high_cardinality',False))),]
            for k,v in items2:
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;padding:.3rem .5rem;
                            border-bottom:1px solid #1e2d45;">
                  <span style="color:#94a3b8;font-size:.85rem;">{k}</span>
                  <span style="color:#e2e8f0;font-weight:600;font-size:.85rem;">{v}</span>
                </div>""", unsafe_allow_html=True)

with tab5:
    if dq_info:
        dq_score = dq_info.get("quality_score",0)
        color = "#4ade80" if dq_score >= 0.7 else "#fbbf24" if dq_score >= 0.5 else "#f87171"
        st.markdown(f"""
        <div style="background:#0f1929;border:2px solid {color}44;border-radius:14px;
                    padding:1.5rem;margin-bottom:1rem;text-align:center;">
          <div style="font-size:.85rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;">
            Overall Data Quality</div>
          <div style="font-size:3rem;font-weight:700;color:{color};">{dq_score:.3f}</div>
          <div style="color:{color};font-size:1rem;">{dq_info.get('label','')}</div>
        </div>""", unsafe_allow_html=True)

        dims = [("Completeness",dq_info.get("completeness",0)),
                ("Uniqueness",dq_info.get("uniqueness",0)),
                ("Consistency",dq_info.get("consistency",0)),
                ("Balance",dq_info.get("balance",0)),
                ("Noise Score",dq_info.get("noise_score",0))]
        dq_df = pd.DataFrame(dims, columns=["Dimension","Score"])
        fig = px.bar(dq_df, x="Dimension", y="Score",
                     color="Score", color_continuous_scale=["#f87171","#fbbf24","#4ade80"],
                     range_color=[0,1], text="Score", title="Quality Dimensions")
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(**DARK, yaxis=dict(range=[0,1.15]))
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.success("✅ Analysis complete! Head to **🏋️ Training** to run the pipeline.")
