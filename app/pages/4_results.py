"""EMMDS — Results dashboard"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Results | EMMDS", page_icon="🏆", layout="wide")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*,[class*="css"]{font-family:'Inter',sans-serif!important}
.stApp{background:#05090f}
[data-testid="stSidebar"]{background:#080f1a!important;border-right:1px solid #0f2340}
[data-testid="metric-container"]{background:#0b1628;border:1px solid #0f2340;border-radius:12px;padding:.9rem 1.1rem!important}
[data-testid="metric-container"] label{color:#475569!important;font-size:.72rem!important;text-transform:uppercase;letter-spacing:.08em;font-weight:600}
[data-testid="metric-container"] [data-testid="metric-value"]{color:#f1f5f9!important;font-size:1.5rem!important;font-weight:700!important}
.stButton>button{background:#0f2040;color:#60a5fa;border:1px solid #1e3a5f;border-radius:9px;font-weight:500;transition:all .2s}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#1d4ed8,#7c3aed);border:none;color:#fff;font-weight:600}
.stTabs [data-baseweb="tab-list"]{background:#0b1628;border-radius:10px;padding:.25rem;border:1px solid #0f2340}
.stTabs [data-baseweb="tab"]{background:transparent;color:#475569;border-radius:8px;font-size:.84rem;font-weight:500;padding:.45rem 1rem;border:none}
.stTabs [aria-selected="true"]{background:#112240!important;color:#60a5fa!important}
[data-testid="stExpander"]{background:#0b1628;border:1px solid #0f2340;border-radius:10px}
[data-testid="stDataFrame"]{border:1px solid #0f2340;border-radius:10px}
hr{border-color:#0f2340!important;margin:1.5rem 0!important}
.stSuccess{background:#042010!important;border-left:3px solid #22c55e!important;border-radius:8px!important}
.stWarning{background:#100c01!important;border-left:3px solid #f59e0b!important;border-radius:8px!important}
.stInfo{background:#060f1f!important;border-left:3px solid #3b82f6!important;border-radius:8px!important}
.stNumberInput>div>div{background:#0b1628!important;border-color:#0f2340!important;border-radius:8px!important}
[data-baseweb="input"]>div{background:#0b1628!important;border-color:#0f2340!important;border-radius:8px!important}
</style>""", unsafe_allow_html=True)

DARK = dict(plot_bgcolor="#080f1a", paper_bgcolor="#080f1a",
            font=dict(color="#94a3b8", family="Inter"),
            margin=dict(l=10, r=10, t=45, b=10))

if "pipeline_result" not in st.session_state or not st.session_state.pipeline_result:
    st.warning("⚠️ No results yet. Go to **🏋️ Training** first.")
    st.stop()

r        = st.session_state.pipeline_result
decision = r.get("decision", {})
steps    = r.get("steps", {})
lb       = steps.get("leaderboard", [])
ev       = steps.get("evaluation", {})
cv       = steps.get("cv_results", {})
cal      = steps.get("calibration_scores", {})
shap     = steps.get("shap_global", {})
agree    = steps.get("agreement", {})
dq       = steps.get("data_quality", {})
rec      = steps.get("recommendation", {})
meta     = steps.get("meta_features", {})
pm       = decision.get("primary_metric", "f1")
bm       = decision.get("best_model", "—")
ts       = decision.get("trust_score", 0.0)
tb       = decision.get("trust_breakdown", {})
tc       = "#22c55e" if ts >= 0.7 else "#f59e0b" if ts >= 0.5 else "#ef4444"

# ── Header ────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:2rem;">
  <div style="display:flex;align-items:center;gap:.75rem;">
    <div style="width:3rem;height:3rem;background:linear-gradient(135deg,#1d4ed820,#7c3aed20);
                border:1px solid #1e3a5f;border-radius:12px;display:flex;align-items:center;
                justify-content:center;font-size:1.4rem;">🏆</div>
    <div>
      <div style="font-size:1.6rem;font-weight:700;color:#f1f5f9;letter-spacing:-.02em;">Results Dashboard</div>
      <div style="font-size:.85rem;color:#334155;">Decision · Trust breakdown · SHAP · Agreement · Prediction</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# Decision card + Trust gauge
# ═══════════════════════════════════════════════════════════════════════
col_card, col_gauge, col_trust = st.columns([1.4, 1, 1.6])

with col_card:
    ps = decision.get("primary_score", 0)
    ag = agree.get("agreement_score", 0)
    dqs = dq.get("quality_score", 0)
    st.markdown(f"""
    <div style="background:linear-gradient(160deg,#080f1a,#0b1628);
                border:1px solid {tc}35;border-radius:16px;padding:1.8rem 1.5rem;height:100%;">
      <div style="font-size:.65rem;text-transform:uppercase;letter-spacing:.12em;
                  color:#334155;font-weight:600;margin-bottom:.6rem;">EMMDS Decision</div>
      <div style="font-size:1.6rem;font-weight:800;color:#f1f5f9;letter-spacing:-.02em;margin-bottom:.3rem;">
        {bm.replace('_',' ').title()}</div>
      <div style="display:inline-block;background:{tc}18;border:1px solid {tc}40;
                  border-radius:20px;padding:.2rem .75rem;font-size:.78rem;color:{tc};
                  font-weight:600;margin-bottom:1.2rem;">{decision.get('trust_label','')}</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:.6rem;">
        <div style="background:#0f2040;border-radius:10px;padding:.7rem .9rem;">
          <div style="font-size:.62rem;color:#334155;text-transform:uppercase;letter-spacing:.08em;">{pm.upper()}</div>
          <div style="font-size:1.25rem;font-weight:700;color:#60a5fa;">{ps:.4f}</div>
        </div>
        <div style="background:#0f2040;border-radius:10px;padding:.7rem .9rem;">
          <div style="font-size:.62rem;color:#334155;text-transform:uppercase;letter-spacing:.08em;">Agreement</div>
          <div style="font-size:1.25rem;font-weight:700;color:#a78bfa;">{ag:.4f}</div>
        </div>
        <div style="background:#0f2040;border-radius:10px;padding:.7rem .9rem;">
          <div style="font-size:.62rem;color:#334155;text-transform:uppercase;letter-spacing:.08em;">Data Quality</div>
          <div style="font-size:1.25rem;font-weight:700;color:#06b6d4;">{dqs:.4f}</div>
        </div>
        <div style="background:#0f2040;border-radius:10px;padding:.7rem .9rem;">
          <div style="font-size:.62rem;color:#334155;text-transform:uppercase;letter-spacing:.08em;">Task</div>
          <div style="font-size:1.1rem;font-weight:700;color:#94a3b8;">{r.get('task','—').title()}</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

with col_gauge:
    gauge_color = "#22c55e" if ts >= 0.7 else "#f59e0b" if ts >= 0.5 else "#ef4444"
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(ts, 4),
        number={"font": {"size": 28, "color": gauge_color, "family": "Inter"}, "suffix": ""},
        title={"text": "Trust Score", "font": {"size": 12, "color": "#475569", "family": "Inter"}},
        gauge={
            "axis":      {"range": [0, 1], "tickwidth": 1, "tickcolor": "#1e3a5f",
                          "tickfont": {"color": "#334155", "size": 9}},
            "bar":       {"color": gauge_color, "thickness": 0.22},
            "bgcolor":   "#080f1a",
            "borderwidth": 0,
            "steps": [
                {"range": [0,   0.5], "color": "#ef444410"},
                {"range": [0.5, 0.7], "color": "#f59e0b10"},
                {"range": [0.7, 1.0], "color": "#22c55e10"},
            ],
            "threshold": {
                "line": {"color": gauge_color, "width": 3},
                "thickness": 0.75, "value": ts,
            },
        },
    ))
    fig_gauge.update_layout(
        paper_bgcolor="#080f1a", font_color="#94a3b8",
        height=220, margin=dict(l=20, r=20, t=40, b=10),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_trust:
    if tb:
        comp_map = {
            "accuracy_component":     ("Accuracy",     "#3b82f6"),
            "calibration_component":  ("Calibration",  "#8b5cf6"),
            "agreement_component":    ("Agreement",    "#a78bfa"),
            "data_quality_component": ("Data Quality", "#06b6d4"),
            "stability_component":    ("Stability",    "#22c55e"),
        }
        st.markdown("""<div style="font-size:.65rem;text-transform:uppercase;letter-spacing:.12em;
                    color:#334155;font-weight:600;margin-bottom:.8rem;">5-Component Trust</div>""",
                    unsafe_allow_html=True)
        for key, (label, color) in comp_map.items():
            val = tb.get(key, 0)
            pct = int(val * 100)
            st.markdown(f"""
            <div style="margin-bottom:.55rem;">
              <div style="display:flex;justify-content:space-between;margin-bottom:.2rem;">
                <span style="font-size:.78rem;color:#94a3b8;">{label}</span>
                <span style="font-size:.78rem;color:{color};font-weight:600;">{val:.3f}</span>
              </div>
              <div style="background:#0b1628;border-radius:4px;height:6px;overflow:hidden;">
                <div style="width:{pct}%;background:{color};height:100%;border-radius:4px;
                            transition:width .5s ease;"></div>
              </div>
            </div>""", unsafe_allow_html=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════
# Tab sections
# ═══════════════════════════════════════════════════════════════════════
tab_lb, tab_shap, tab_agree, tab_cv, tab_dq, tab_predict = st.tabs([
    "📊 Leaderboard", "🔥 SHAP", "🤝 Agreement", "📈 Cross-Val", "🗂️ Data Quality", "🎯 Predict"
])

# ── Leaderboard ───────────────────────────────────────────────────────
with tab_lb:
    if lb:
        lb_df = pd.DataFrame(lb)

        # Trust scores bar (all models)
        all_ts = decision.get("all_trust_scores", {})
        if all_ts:
            ts_df = pd.DataFrame([{"Model": k, "Trust Score": v}
                                   for k, v in all_ts.items()]).sort_values("Trust Score", ascending=True)
            l1, l2 = st.columns(2)
            with l1:
                fig = px.bar(ts_df, x="Trust Score", y="Model", orientation="h",
                             color="Trust Score",
                             color_continuous_scale=["#ef4444","#f59e0b","#22c55e"],
                             range_color=[0, 1], text="Trust Score",
                             title="Trust Score — All Models")
                fig.update_traces(texttemplate="%{text:.3f}", textposition="outside",
                                  marker_line_width=0)
                fig.update_layout(**DARK, height=max(260, len(ts_df)*50+80),
                                  xaxis=dict(range=[0,1.1], showgrid=True, gridcolor="#0f2340"),
                                  yaxis_showgrid=False, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            with l2:
                if pm in lb_df.columns:
                    lb_s = lb_df.sort_values(pm, ascending=True)
                    fig2 = px.bar(lb_s, x=pm, y="model", orientation="h",
                                  color=pm,
                                  color_continuous_scale=["#112240","#3b82f6","#8b5cf6"],
                                  range_color=[lb_s[pm].min()*0.95, lb_s[pm].max()],
                                  title=f"{pm.upper()} Score — All Models", text=pm)
                    fig2.update_traces(texttemplate="%{text:.4f}", textposition="outside",
                                       marker_line_width=0)
                    fig2.update_layout(**DARK, height=max(260, len(lb_s)*50+80),
                                       xaxis=dict(showgrid=True, gridcolor="#0f2340"),
                                       yaxis_showgrid=False, coloraxis_showscale=False)
                    st.plotly_chart(fig2, use_container_width=True)

        # Full table
        show_cols = [c for c in ["model", pm, "accuracy", "f1", "precision", "recall", "auc"]
                     if c in lb_df.columns]
        st.dataframe(lb_df[show_cols].round(4), use_container_width=True, hide_index=True)

# ── SHAP ──────────────────────────────────────────────────────────────
with tab_shap:
    if shap and shap.get("feature_importances"):
        fi = shap.get("feature_importances", {})
        top_n = 20
        items = sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        shap_df = pd.DataFrame(items, columns=["Feature","Importance"])

        fig = px.bar(shap_df.sort_values("Importance"), x="Importance", y="Feature",
                     orientation="h", color="Importance",
                     color_continuous_scale=["#1e3a5f","#60a5fa","#f472b6"],
                     title=f"SHAP Global Feature Importance — {bm.replace('_',' ').title()}",
                     text="Importance")
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside", marker_line_width=0)
        fig.update_layout(**DARK, coloraxis_showscale=False,
                          height=max(320, len(shap_df)*30+100),
                          xaxis=dict(showgrid=True, gridcolor="#0f2340"),
                          yaxis_showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

        top_feats = shap.get("top_features", [])
        if top_feats:
            st.markdown(f"""
            <div style="background:#0b1628;border:1px solid #0f2340;border-radius:10px;
                        padding:1rem 1.2rem;margin-top:.5rem;">
              <div style="font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;
                          color:#334155;font-weight:600;margin-bottom:.6rem;">Top Features</div>
              <div style="display:flex;flex-wrap:wrap;gap:.4rem;">
                {"".join(f'<span style="background:#0f2040;border:1px solid #1e3a5f;border-radius:20px;padding:.2rem .75rem;font-size:.78rem;color:#60a5fa;">{f}</span>' for f in top_feats[:12])}
              </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("SHAP explanations not available for this run.")

# ── Agreement ─────────────────────────────────────────────────────────
with tab_agree:
    a1, a2 = st.columns(2)
    with a1:
        m1, m2, m3 = st.columns(3)
        m1.metric("Global", f"{agree.get('global_agreement',0):.3f}")
        m2.metric("Pairwise", f"{agree.get('mean_pairwise',0):.3f}")
        m3.metric("Score", f"{agree.get('agreement_score',0):.3f}")

        per_model = agree.get("per_model_agreement", {})
        if per_model:
            ag_df = pd.DataFrame([{"Model": k, "Agreement": v}
                                   for k, v in per_model.items()]).sort_values("Agreement", ascending=True)
            fig = px.bar(ag_df, x="Agreement", y="Model", orientation="h",
                         color="Agreement", color_continuous_scale=["#1e3a5f","#3b82f6"],
                         range_color=[0,1], text="Agreement",
                         title="Per-Model Agreement vs Majority Vote")
            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside",
                              marker_line_width=0)
            fig.update_layout(**DARK, coloraxis_showscale=False,
                              xaxis=dict(range=[0,1.1], showgrid=True, gridcolor="#0f2340"),
                              yaxis_showgrid=False,
                              height=max(250, len(ag_df)*50+80))
            st.plotly_chart(fig, use_container_width=True)

    with a2:
        entropy = agree.get("entropy", 0)
        pw      = agree.get("pairwise_matrix", {})
        if pw:
            models   = list(pw.keys())
            mat_data = [[pw[r].get(c, 0) for c in models] for r in models]
            fig = px.imshow(mat_data, x=models, y=models,
                            color_continuous_scale="Blues", zmin=0, zmax=1,
                            title="Pairwise Agreement Matrix", text_auto=".2f")
            fig.update_layout(**DARK, height=320)
            st.plotly_chart(fig, use_container_width=True)
        st.metric("Prediction Entropy", f"{entropy:.4f}" if entropy else "—")

# ── Cross-Val ─────────────────────────────────────────────────────────
with tab_cv:
    if cv:
        rows = []
        for model_name, res in cv.items():
            if isinstance(res, dict):
                rows.append({"Model": model_name,
                             "CV Mean": round(res.get("mean",0),4),
                             "CV Std":  round(res.get("std",0),4),
                             "CV Min":  round(res.get("min",0),4),
                             "CV Max":  round(res.get("max",0),4)})
        if rows:
            cv_df = pd.DataFrame(rows).sort_values("CV Mean", ascending=False)
            st.dataframe(cv_df, use_container_width=True, hide_index=True)

            fig = go.Figure()
            for _, row in cv_df.iterrows():
                fig.add_trace(go.Bar(
                    x=[row["Model"]], y=[row["CV Mean"]],
                    error_y=dict(type="data", array=[row["CV Std"]], visible=True,
                                 color="#475569"),
                    name=row["Model"],
                    text=[f"{row['CV Mean']:.4f}"],
                    textposition="outside",
                    marker_color="#3b82f6",
                    marker_line_width=0,
                ))
            fig.update_layout(**DARK, showlegend=False, title="Cross-Validation Mean ± Std",
                              title_font=dict(color="#e2e8f0", size=13),
                              xaxis=dict(showgrid=False, tickangle=-30),
                              yaxis=dict(showgrid=True, gridcolor="#0f2340"),
                              height=320, barmode="group")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Cross-validation results not available.")

# ── Data Quality ──────────────────────────────────────────────────────
with tab_dq:
    if dq:
        dq_score = dq.get("quality_score", 0)
        dq_color = "#22c55e" if dq_score >= 0.7 else "#f59e0b" if dq_score >= 0.5 else "#ef4444"

        d1, d2 = st.columns([1, 2])
        with d1:
            st.markdown(f"""
            <div style="background:linear-gradient(160deg,#080f1a,#0b1628);
                        border:2px solid {dq_color}30;border-radius:16px;
                        padding:1.5rem;text-align:center;">
              <div style="font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;
                          color:#334155;font-weight:600;">Data Quality</div>
              <div style="font-size:3.5rem;font-weight:800;color:{dq_color};
                          line-height:1.1;margin:.4rem 0;">{dq_score:.3f}</div>
              <div style="font-size:.88rem;color:{dq_color};font-weight:500;">
                {dq.get('label','')}</div>
            </div>""", unsafe_allow_html=True)
        with d2:
            dims = [
                ("Completeness",  dq.get("completeness", 0),  "#3b82f6"),
                ("Uniqueness",    dq.get("uniqueness", 0),    "#8b5cf6"),
                ("Consistency",   dq.get("consistency", 0),   "#06b6d4"),
                ("Balance",       dq.get("balance", 0),       "#22c55e"),
                ("Noise Score",   dq.get("noise_score", 0),   "#f472b6"),
            ]
            for label, val, color in dims:
                pct = int(val * 100)
                st.markdown(f"""
                <div style="margin-bottom:.6rem;">
                  <div style="display:flex;justify-content:space-between;margin-bottom:.25rem;">
                    <span style="font-size:.82rem;color:#94a3b8;">{label}</span>
                    <span style="font-size:.82rem;color:{color};font-weight:600;">{val:.3f}</span>
                  </div>
                  <div style="background:#0b1628;border-radius:5px;height:7px;overflow:hidden;">
                    <div style="width:{pct}%;background:{color};height:100%;border-radius:5px;"></div>
                  </div>
                </div>""", unsafe_allow_html=True)

# ── Predict ───────────────────────────────────────────────────────────
with tab_predict:
    feat_names = r.get("_feature_names", [])
    trained    = r.get("_trained_models", {})
    le         = r.get("_label_encoder")
    task_type  = r.get("task", "classification")

    if not trained or not feat_names:
        st.info("No trained models in session. Re-run the pipeline to use predictions.")
    else:
        st.markdown(f"""
        <div style="background:#0b1628;border:1px solid #0f2340;border-radius:12px;
                    padding:1rem 1.2rem;margin-bottom:1rem;">
          <div style="font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;
                      color:#334155;font-weight:600;margin-bottom:.4rem;">Prediction Playground</div>
          <div style="font-size:.82rem;color:#475569;">
            Model: <span style="color:#60a5fa;font-weight:500;">{bm.replace('_',' ').title()}</span>
            &nbsp;·&nbsp; {len(feat_names)} features
          </div>
        </div>""", unsafe_allow_html=True)

        X_test = r.get("_X_test")
        if X_test is not None and len(X_test) > 0:
            sample_idx = st.slider("Pick a test sample", 0, len(X_test)-1, 0)
            sample_row = X_test.iloc[sample_idx]
            defaults   = {f: float(sample_row[f]) if f in sample_row.index else 0.0
                          for f in feat_names}
        else:
            defaults = {f: 0.0 for f in feat_names}

        cols_per_row = 4
        feat_vals    = {}
        feat_list    = list(feat_names)
        for row_start in range(0, len(feat_list), cols_per_row):
            batch = feat_list[row_start: row_start + cols_per_row]
            row_cols = st.columns(len(batch))
            for col, feat in zip(row_cols, batch):
                with col:
                    feat_vals[feat] = st.number_input(
                        feat[:20], value=float(defaults.get(feat, 0.0)),
                        key=f"pred_{feat}", label_visibility="visible")

        if st.button("⚡  Run Prediction", type="primary"):
            try:
                import pandas as pd
                best_pipe = trained[bm]
                input_df  = pd.DataFrame([[feat_vals[f] for f in feat_names]], columns=feat_names)
                raw_pred  = best_pipe.predict(input_df)[0]
                label     = le.inverse_transform([int(raw_pred)])[0] if le else raw_pred

                proba = None
                if hasattr(best_pipe, "predict_proba"):
                    proba = best_pipe.predict_proba(input_df)[0]

                conf_str = ""
                if proba is not None:
                    conf = float(np.max(proba))
                    conf_str = f"· Confidence <span style='color:#60a5fa;font-weight:700;'>{conf:.1%}</span>"

                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#042010,#041a10);
                            border:1px solid #166534;border-radius:14px;
                            padding:1.5rem 2rem;margin-top:.8rem;text-align:center;">
                  <div style="font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;
                              color:#334155;font-weight:600;margin-bottom:.4rem;">Prediction</div>
                  <div style="font-size:2.5rem;font-weight:800;color:#4ade80;">{label}</div>
                  <div style="font-size:.85rem;color:#334155;margin-top:.3rem;">{conf_str}</div>
                </div>""", unsafe_allow_html=True)

                if proba is not None and le is not None:
                    classes = le.classes_
                    prob_df = pd.DataFrame({"Class": classes, "Probability": proba})
                    fig = px.bar(prob_df, x="Class", y="Probability",
                                 color="Probability",
                                 color_continuous_scale=["#1e3a5f","#22c55e"],
                                 range_color=[0,1], text="Probability",
                                 title="Class Probabilities")
                    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside",
                                      marker_line_width=0)
                    fig.update_layout(**DARK, coloraxis_showscale=False,
                                      yaxis=dict(range=[0,1.12]),
                                      height=280)
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")
