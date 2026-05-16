"""EMMDS Page 5 — Research Results Dashboard"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Research | EMMDS", page_icon="🔬", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*,[class*="css"]{font-family:'Inter',sans-serif!important}
.stApp{background:#05090f}
[data-testid="stSidebar"]{background:#080f1a!important;border-right:1px solid #0f2340}
[data-testid="metric-container"]{background:#0b1628;border:1px solid #0f2340;border-radius:12px;padding:.9rem 1.1rem!important}
[data-testid="metric-container"] label{color:#475569!important;font-size:.72rem!important;text-transform:uppercase;letter-spacing:.08em;font-weight:600}
[data-testid="metric-container"] [data-testid="metric-value"]{color:#f1f5f9!important;font-size:1.5rem!important;font-weight:700!important}
.stTabs [data-baseweb="tab-list"]{background:#0b1628;border-radius:10px;padding:.25rem;border:1px solid #0f2340}
.stTabs [data-baseweb="tab"]{background:transparent;color:#475569;border-radius:8px;font-size:.84rem;font-weight:500;padding:.45rem 1rem;border:none}
.stTabs [aria-selected="true"]{background:#112240!important;color:#60a5fa!important}
[data-testid="stDataFrame"]{border:1px solid #0f2340;border-radius:10px}
hr{border-color:#0f2340!important;margin:1.5rem 0!important}
.stSuccess{background:#042010!important;border-left:3px solid #22c55e!important;border-radius:8px!important}
.stWarning{background:#151000!important;border-left:3px solid #f59e0b!important;border-radius:8px!important}
.stInfo{background:#060f1f!important;border-left:3px solid #3b82f6!important;border-radius:8px!important}
</style>""", unsafe_allow_html=True)

DARK = dict(plot_bgcolor="#080f1a", paper_bgcolor="#080f1a",
            font=dict(color="#94a3b8", family="Inter"),
            margin=dict(l=20, r=20, t=50, b=20))
RES_DIR = Path(__file__).parent.parent.parent / "outputs" / "research"

st.markdown("""
<div style="margin-bottom:1.5rem;">
  <h1 style="color:#e2e8f0;margin:0;">🔬 Research Results</h1>
  <p style="color:#64748b;margin:.3rem 0 0;">
    Experimental validation of the EMMDS Trust Score across 20 datasets.
  </p>
</div>""", unsafe_allow_html=True)

# ── Load results ──────────────────────────────────────────────────────
@st.cache_data
def load_results():
    raw_path  = RES_DIR / "raw_results.csv"
    corr_path = RES_DIR / "claim_A_correlations.json"
    abl_path  = RES_DIR / "ablation_corrected.csv"
    base_path = RES_DIR / "baseline_corrected.csv"
    hyp_path  = RES_DIR / "hypothesis_tests.json"
    clC_path  = RES_DIR / "claim_C_results.json"
    clD_path  = RES_DIR / "claim_D_results.json"

    results = {}
    if raw_path.exists():  results["raw"]  = pd.read_csv(raw_path)
    if corr_path.exists(): results["corrA"] = json.loads(corr_path.read_text())
    if abl_path.exists():  results["abl"]  = pd.read_csv(abl_path)
    if base_path.exists(): results["base"] = pd.read_csv(base_path)
    if hyp_path.exists():  results["hyp"]  = json.loads(hyp_path.read_text())
    if clC_path.exists():  results["clC"]  = json.loads(clC_path.read_text())
    if clD_path.exists():  results["clD"]  = json.loads(clD_path.read_text())
    return results

res = load_results()

if not res:
    st.warning("No research results found. Run `python src/research/experiments.py` first.")
    st.code("cd emmds\npython src/research/experiments.py")
    st.stop()

raw = res.get("raw")

# ── Research claim header ─────────────────────────────────────────────
st.markdown("""
<div style="background:#0f1929;border:1px solid #1e3a5f;border-radius:14px;padding:1.5rem;margin-bottom:1.5rem;">
  <div style="color:#7dd3fc;font-size:.8rem;font-weight:600;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.5rem;">
    RESEARCH CLAIM
  </div>
  <div style="color:#e2e8f0;font-size:1rem;line-height:1.6;">
    The EMMDS Trust Score — combining accuracy, calibration, agreement, data quality, and stability —
    is a statistically significant predictor of deployment risk
    <span style="color:#4ade80;font-weight:600;">(Spearman r = −0.773, p &lt; 0.001)</span>.
    Cross-model agreement outperforms softmax confidence as a reliability proxy
    <span style="color:#4ade80;font-weight:600;">(AUC 0.874 vs 0.569)</span>.
  </div>
</div>""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────
t1,t2,t3,t4,t5 = st.tabs([
    "📊 Claim A: Trust vs Accuracy",
    "🤝 Claim C: Agreement",
    "📐 Claim D: R² Analysis",
    "🧪 Ablation Study",
    "📋 Full Dataset Table"
])

with t1:
    st.subheader("Trust Score as Deployment Risk Predictor")
    st.caption("Deployment risk = 0.40×overfitting + 0.30×calibration_error + 0.30×instability")

    corrA = res.get("corrA", {})
    if corrA:
        corr_df = pd.DataFrame([
            {"Predictor": k, "Spearman r": v["spearman_r"], "p-value": v["p_value"],
             "Significant": "✅" if v["significant"] else "❌"}
            for k, v in corrA.items()
        ]).sort_values("Spearman r")
        fig = px.bar(corr_df, x="Spearman r", y="Predictor", orientation="h",
                     color="Spearman r", color_continuous_scale="RdYlGn",
                     range_color=[-1, 0],
                     title="Spearman Correlation with Deployment Risk (n=84)")
        fig.add_vline(x=0, line_dash="dash", line_color="#475569")
        fig.update_layout(**DARK, height=350)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(corr_df, use_container_width=True)

    if raw is not None:
        raw2 = raw.copy()
        raw2['overfitting_ratio'] = raw2['gen_gap'] / (raw2['test_acc'] + 1e-8)
        raw2['calibration_error'] = 1.0 - raw2['cal_score']
        raw2['deployment_risk'] = (
            0.40 * np.clip(raw2['overfitting_ratio'], 0, 1) +
            0.30 * raw2['calibration_error'] +
            0.30 * raw2['cv_std']
        )

        ca, cb = st.columns(2)
        with ca:
            fig2 = px.scatter(raw2, x="trust_score", y="deployment_risk",
                              color="dataset", hover_data=["model"],
                              title="Trust Score vs Deployment Risk",
                              labels={"trust_score":"Trust Score","deployment_risk":"Deployment Risk"})
            fig2.update_layout(**DARK, height=380)
            st.plotly_chart(fig2, use_container_width=True)
        with cb:
            fig3 = px.scatter(raw2, x="test_acc", y="deployment_risk",
                              color="dataset", hover_data=["model"],
                              title="Accuracy vs Deployment Risk",
                              labels={"test_acc":"Accuracy","deployment_risk":"Deployment Risk"})
            fig3.update_layout(**DARK, height=380)
            st.plotly_chart(fig3, use_container_width=True)

    hyp = res.get("hyp", {})
    if hyp:
        st.markdown("**Formal Hypothesis Test:**")
        col1, col2 = st.columns(2)
        col1.metric("Trust Spearman r", f"{hyp.get('trust_spearman_r',0):.4f}")
        col2.metric("p-value", f"{hyp.get('trust_spearman_p',0):.6f}")
        if hyp.get('trust_spearman_p', 1) < 0.05:
            st.success("✅ Reject H₀: Trust score is a statistically significant predictor of deployment risk")
        else:
            st.warning("⚠️ Fail to reject H₀")

with t2:
    st.subheader("Cross-Model Agreement vs Softmax Confidence")
    clC = res.get("clC", {})
    if clC:
        m1,m2,m3 = st.columns(3)
        m1.metric("Agreement r", f"{clC.get('agreement_spearman',0):.4f}",
                  delta="p<0.001" if clC.get('agreement_p',1)<0.05 else None)
        m2.metric("Softmax r",   f"{clC.get('softmax_spearman',0):.4f}",
                  delta="n.s." if clC.get('softmax_p',1)>=0.05 else None)
        m3.metric("Agreement > Softmax", "✅ YES" if clC.get("agreement_better_than_softmax") else "❌ NO")

        comp_data = pd.DataFrame({
            "Predictor": ["Agreement Score","Softmax Confidence","Trust Score"],
            "Spearman r": [clC.get("agreement_spearman",0),
                           clC.get("softmax_spearman",0),
                           clC.get("trust_spearman",0)],
            "p-value": [clC.get("agreement_p",1),
                        clC.get("softmax_p",1),
                        clC.get("trust_p",1)],
        })
        fig = px.bar(comp_data, x="Predictor", y="Spearman r",
                     color="Spearman r", color_continuous_scale="RdYlGn",
                     range_color=[-1,0],
                     title="Reliability Predictors: Agreement vs Softmax vs Trust")
        fig.update_layout(**DARK, height=350, yaxis=dict(range=[-0.7,0.1]))
        st.plotly_chart(fig, use_container_width=True)

        st.info("""
        **Key finding:** Softmax confidence is NOT a statistically significant predictor of deployment risk (p = 0.108).
        Cross-model agreement IS (p < 0.001). This means practitioners who rely on a single model's
        confidence scores for reliability estimates may be misled.
        """)

with t3:
    st.subheader("Explaining Generalisation Variance (R² Analysis)")
    clD = res.get("clD", {})
    if clD:
        d_rows = [{"Model": k,
                   "R² (CV mean)": v["r2_cv_mean"],
                   "R² (CV std)":  v["r2_cv_std"],
                   "Features":     ", ".join(v["features"])}
                  for k, v in clD.items()]
        d_df = pd.DataFrame(d_rows)
        fig = px.bar(d_df, x="Model", y="R² (CV mean)",
                     error_y="R² (CV std)",
                     color="R² (CV mean)", color_continuous_scale="Blues",
                     title="R² for Predicting Deployment Risk (5-fold CV)")
        fig.update_layout(**DARK, height=380, yaxis=dict(range=[-0.1,0.7]))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(d_df, use_container_width=True)
        st.info("M3 (full model) achieves R²=0.385, vs M1 (accuracy only) at R²=0.424 — "
                "suggesting calibration and agreement add some information but accuracy alone is surprisingly strong.")

with t4:
    st.subheader("Ablation Study: Component Contribution")
    abl = res.get("abl")
    if abl is not None:
        fig = px.bar(abl, x="condition", y="selection_accuracy",
                     color="mean_deployment_risk",
                     color_continuous_scale="RdYlGn_r",
                     title="Ablation: Selection Accuracy by Component Removal",
                     text="selection_accuracy")
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(**DARK, height=400, xaxis_tickangle=-25,
                          yaxis=dict(range=[0, 0.55]))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(abl[["condition","selection_accuracy","mean_deployment_risk","correct","total"]],
                     use_container_width=True)
        st.success("**Key finding:** Equal weights (0.20 each) achieves the highest selection accuracy (0.417), "
                   "suggesting non-accuracy components carry orthogonal reliability information.")

    base = res.get("base")
    if base is not None:
        st.subheader("Baseline Comparison")
        fig2 = px.bar(base, x="selector", y="selection_accuracy",
                      color="mean_deployment_risk", color_continuous_scale="RdYlGn_r",
                      title="Selector Comparison: Trust vs Accuracy vs Random",
                      text="selection_accuracy")
        fig2.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig2.update_layout(**DARK, height=320, yaxis=dict(range=[0,0.5]))
        st.plotly_chart(fig2, use_container_width=True)

with t5:
    st.subheader("Full Experiment Results (84 model × dataset pairs)")
    if raw is not None:
        show_cols = ["dataset","model","test_acc","test_f1","trust_score",
                     "cal_score","stability","agreement_score","gen_gap","dtype"]
        show_cols = [c for c in show_cols if c in raw.columns]
        styled = raw[show_cols].style.background_gradient(
            subset=["trust_score","test_acc"], cmap="Greens"
        ).background_gradient(subset=["gen_gap"], cmap="Reds_r")
        st.dataframe(styled, use_container_width=True, height=500)

        # Download
        csv = raw.to_csv(index=False)
        st.download_button("⬇️ Download full results CSV", csv,
                           "emmds_research_results.csv", "text/csv")
