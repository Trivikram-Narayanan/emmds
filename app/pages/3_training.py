"""EMMDS — Training page"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Training | EMMDS", page_icon="🏋️", layout="wide")

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
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#1d4ed8,#7c3aed);border:none;color:#fff;font-weight:600;font-size:.95rem}
.stButton>button[kind="primary"]:hover{opacity:.92;transform:translateY(-2px);box-shadow:0 8px 25px #7c3aed35}
[data-baseweb="select"]>div{background:#0b1628!important;border-color:#0f2340!important;border-radius:8px!important;color:#e2e8f0!important}
[data-testid="stCheckbox"] label{color:#475569!important}
.stProgress>div>div{background:linear-gradient(90deg,#3b82f6,#8b5cf6,#06b6d4);border-radius:4px}
[data-testid="stExpander"]{background:#0b1628;border:1px solid #0f2340;border-radius:10px}
hr{border-color:#0f2340!important;margin:1.5rem 0!important}
.stSuccess{background:#042010!important;border-left:3px solid #22c55e!important;border-radius:8px!important}
.stWarning{background:#100c01!important;border-left:3px solid #f59e0b!important;border-radius:8px!important}
.stError{background:#130805!important;border-left:3px solid #ef4444!important;border-radius:8px!important}
</style>""", unsafe_allow_html=True)

if "df" not in st.session_state or st.session_state.df is None:
    st.warning("⚠️ No dataset loaded. Go to **📁 Upload** first.")
    st.stop()
if "target_col" not in st.session_state or not st.session_state.target_col:
    st.warning("⚠️ No target column set. Go to **📊 Analysis** first.")
    st.stop()

df         = st.session_state.df
target_col = st.session_state.target_col

# ── Header ────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-bottom:2rem;">
  <div style="display:flex;align-items:center;gap:.75rem;">
    <div style="width:3rem;height:3rem;background:linear-gradient(135deg,#1d4ed820,#7c3aed20);
                border:1px solid #1e3a5f;border-radius:12px;display:flex;align-items:center;
                justify-content:center;font-size:1.4rem;">🏋️</div>
    <div>
      <div style="font-size:1.6rem;font-weight:700;color:#f1f5f9;letter-spacing:-.02em;">Model Training</div>
      <div style="font-size:.85rem;color:#334155;">9-stage pipeline · target:
        <code style="background:#0f2040;color:#60a5fa;padding:.1rem .4rem;border-radius:5px;font-size:.8rem;">{target_col}</code>
      </div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

# ── Config panel ──────────────────────────────────────────────────────
st.markdown("""<div style="background:#080f1a;border:1px solid #0f2340;border-radius:14px;
            padding:1.3rem 1.5rem 1rem;">
<div style="font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;color:#334155;
            font-weight:600;margin-bottom:.9rem;">Pipeline Configuration</div>""",
            unsafe_allow_html=True)

cfg1, cfg2, cfg3 = st.columns(3)
with cfg1:
    task_opt = st.selectbox("Task type", ["Auto-detect", "classification", "regression"])
    task = None if task_opt == "Auto-detect" else task_opt
with cfg2:
    scaler = st.selectbox("Feature scaling", ["standard", "minmax", "none"])
with cfg3:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    track_runs  = st.checkbox("📊 Track experiment", value=True)
    save_models = st.checkbox("💾 Save models to disk", value=False)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)

# ── Stage tracker ─────────────────────────────────────────────────────
STAGES = [
    ("Validate",   "#3b82f6"),
    ("Analyse",    "#8b5cf6"),
    ("Recommend",  "#f59e0b"),
    ("Train",      "#ef4444"),
    ("Cross-Val",  "#22c55e"),
    ("Calibrate",  "#06b6d4"),
    ("SHAP",       "#f472b6"),
    ("Agreement",  "#a78bfa"),
    ("Trust",      "#34d399"),
]

def render_stages(done: int, active: int = -1):
    cols = st.columns(len(STAGES))
    for i, (col, (name, color)) in enumerate(zip(cols, STAGES)):
        if i < done:
            icon, bg, tc, brd = "✓", f"{color}18", color, f"1px solid {color}50"
        elif i == active:
            icon, bg, tc, brd = "●", f"{color}28", color, f"2px solid {color}"
        else:
            icon, bg, tc, brd = str(i+1), "#0b1628", "#1e3a5f", "1px solid #0a1525"
        col.markdown(f"""
        <div style="background:{bg};border:{brd};border-radius:10px;
                    padding:.55rem .3rem;text-align:center;">
          <div style="font-size:.95rem;font-weight:700;color:{tc};">{icon}</div>
          <div style="font-size:.58rem;color:{tc if i<=max(done-1,active) else '#1e3a5f'};
                      margin-top:.2rem;font-weight:{'600' if i<done else '400'};">{name}</div>
        </div>""", unsafe_allow_html=True)

stage_slot = st.empty()
with stage_slot.container():
    render_stages(0)

st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
run = st.button("🚀  Start Full Pipeline", type="primary", use_container_width=True)

# ── Pipeline execution ────────────────────────────────────────────────
if run:
    progress  = st.progress(0, text="Initialising…")
    log_slot  = st.expander("📋 Live log", expanded=True)
    log_lines = []

    def log(msg, pct, done=None, active=-1):
        log_lines.append(msg)
        progress.progress(pct, text=msg)
        if done is not None:
            with stage_slot.container():
                render_stages(done, active)
        with log_slot:
            for ln in log_lines[-16:]:
                ic = "🟢" if "✅" in ln else "🔴" if "❌" in ln else "🟡" if "⚠" in ln else "⚪"
                st.markdown(
                    f'<div style="font-family:monospace;font-size:.76rem;color:#64748b;'
                    f'padding:.1rem 0;border-bottom:1px solid #080f1a;">{ic}&nbsp;{ln}</div>',
                    unsafe_allow_html=True)

    try:
        log("Stage 1/9 · Validating data…", 5, 0, 0)
        from src.data_engine.validator import DataValidator
        val = DataValidator().validate(df, target_col)
        if not val["passed"]:
            st.error(f"Validation failed: {val['errors']}")
            st.stop()
        log(f"  ✅ Passed — {val['warning_count']} warnings", 10, 1, 1)

        log("Stage 2/9 · Analysing dataset…", 12, 1, 1)
        from src.data_engine.analyzer import DataAnalyzer
        from src.data_engine.meta_features import MetaFeatureExtractor
        from src.data_engine.data_quality import DataQualityScorer
        analysis  = DataAnalyzer().analyze(df, target_col)
        det_task  = task or analysis["task"]
        meta      = MetaFeatureExtractor().extract(df, target_col)
        dq_scorer = DataQualityScorer()
        dq_score  = dq_scorer.score_dataset(df, target_col, task=det_task)
        dq_info   = dq_scorer.get_breakdown()
        log(f"  ✅ Task: {det_task} · Quality: {dq_score:.3f}", 18, 2, 2)

        log("Stage 3/9 · Model recommendation…", 20, 2, 2)
        from src.decision.model_recommender import ModelRecommender
        recommender = ModelRecommender()
        recommended = recommender.recommend(meta)
        rec_report  = recommender.get_report()
        log(f"  ✅ Selected: {', '.join(recommended)}", 25, 3, 3)

        log("Stage 4/9 · Training models…", 28, 3, 3)
        from src.training.data_split import DataSplitter
        from src.training.pipeline_builder import build_all_pipelines
        from src.models.model_registry import get_all_models
        from sklearn.preprocessing import LabelEncoder
        X_raw = df.drop(columns=[target_col])
        y_raw = df[target_col]
        splitter = DataSplitter(task=det_task)
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = splitter.split(X_raw, y_raw)
        le = None
        if det_task == "classification":
            le = LabelEncoder()
            y_train = le.fit_transform(y_train_raw)
            y_test  = le.transform(y_test_raw)
        else:
            y_train = y_train_raw.to_numpy()
            y_test  = y_test_raw.to_numpy()
        num_cols   = list(X_train_raw.select_dtypes(include=[np.number]).columns)
        cat_cols   = list(X_train_raw.select_dtypes(include=["object","category","bool"]).columns)
        feat_names = num_cols + cat_cols
        all_models = get_all_models(enabled_only=True)
        model_sub  = {k: v for k, v in all_models.items() if k in recommended}
        pipelines  = build_all_pipelines(model_sub, num_cols, cat_cols, scaler)
        trained_models, train_times = {}, {}
        for name, pipe in pipelines.items():
            t0 = time.time()
            try:
                pipe.fit(X_train_raw, y_train)
                trained_models[name] = pipe
                train_times[name]    = round(time.time()-t0, 3)
                log(f"  ✅ {name} — {train_times[name]}s", 38)
            except Exception as e:
                log(f"  ❌ {name}: {e}", 38)
        log(f"  ✅ {len(trained_models)} models trained", 42, 4, 4)

        log("Stage 5/9 · Cross-validation (5-fold)…", 44, 4, 4)
        from src.training.cross_validation import CrossValidator
        X_all_raw  = pd.concat([X_train_raw, X_test_raw]).reset_index(drop=True)
        y_all      = np.concatenate([y_train, y_test])
        cv_results = CrossValidator(task=det_task).run(trained_models, X_all_raw, y_all)
        log(f"  ✅ CV complete", 55, 5, 5)

        log("Stage 6/9 · Evaluating + calibrating…", 57, 5, 5)
        from src.evaluation.evaluator import ModelEvaluator
        from src.evaluation.ranking import ModelRanker
        from src.calibration.calibrator import ModelCalibrator
        eval_results = ModelEvaluator(task=det_task).evaluate_all(trained_models, X_test_raw, y_test)
        calibrator   = ModelCalibrator()
        calibrated   = calibrator.calibrate_all(trained_models, X_train_raw, y_train, X_test_raw, y_test)
        cal_scores   = calibrator.get_calibration_scores()
        leaderboard  = ModelRanker(task=det_task).rank(eval_results, cv_results)
        log(f"  ✅ Best: {leaderboard[0]['model']}", 65, 6, 6)

        log("Stage 7/9 · SHAP explanations…", 67, 6, 6)
        best_name   = leaderboard[0]["model"]
        best_model  = calibrated.get(best_name)
        shap_global = {}
        try:
            from src.explainability.shap_explainer import SHAPExplainer
            shap_exp = SHAPExplainer(max_samples=100)
            if hasattr(best_model, "named_steps"):
                Xt = best_model.named_steps["preprocessor"].transform(X_train_raw)
                Xs = best_model.named_steps["preprocessor"].transform(X_test_raw)
                shap_exp.fit(best_model.named_steps["model"], Xt, feature_names=feat_names)
                shap_global = shap_exp.explain_global(Xs)
            else:
                shap_exp.fit(best_model, X_train_raw.values, feature_names=feat_names)
                shap_global = shap_exp.explain_global(X_test_raw.values)
            log(f"  ✅ Top feature: {shap_global.get('top_features',['?'])[0]}", 76, 7, 7)
        except Exception as e:
            log(f"  ⚠️  SHAP skipped: {e}", 76, 7, 7)

        log("Stage 8/9 · Model agreement…", 78, 7, 7)
        from src.decision.model_agreement import ModelAgreementEngine
        agree_result = ModelAgreementEngine().compute(calibrated, X_test_raw, task=det_task)
        agree_score  = agree_result.get("agreement_score", 0.5)
        log(f"  ✅ Agreement: {agree_score:.4f}", 82, 8, 8)

        log("Stage 9/9 · Trust scoring + decision…", 84, 8, 8)
        from src.decision.trust_score import TrustScoreEngine
        from src.decision.decision_engine import DecisionEngine
        trust_engine   = TrustScoreEngine()
        trust_scores   = trust_engine.compute_all(
            eval_results=eval_results, calibration_scores=cal_scores,
            cv_results=cv_results, task=det_task,
            agreement_score=agree_score, data_quality_score=dq_score,
        )
        engine         = DecisionEngine(task=det_task)
        final_decision = engine.decide(
            trained_models=calibrated, eval_results=eval_results,
            calibration_scores=cal_scores, cv_results=cv_results,
            leaderboard=leaderboard, shap_global=shap_global, analysis_report=analysis,
        )
        final_decision["all_trust_scores"] = trust_scores
        final_decision["trust_score"]      = trust_scores.get(final_decision["best_model"], 0.0)
        final_decision["trust_label"]      = trust_engine.get_trust_label(final_decision["trust_score"])
        final_decision["trust_breakdown"]  = trust_engine.get_breakdown().get(final_decision["best_model"], {})
        final_decision["agreement"]        = agree_result
        final_decision["data_quality"]     = dq_info
        final_decision["recommendation"]   = rec_report
        log(f"  ✅ {final_decision['best_model']} · Trust {final_decision['trust_score']:.4f}", 95, 9)

        if track_runs:
            try:
                from src.utils.experiment_tracker import ExperimentTracker
                t = ExperimentTracker()
                t.start_run(dataset_name=st.session_state.get("filename","dataset"),
                            task=det_task, target_col=target_col,
                            n_samples=meta["n_samples"], n_features=meta["n_features"],
                            params={"scaler": scaler, "recommended": recommended})
                for n, m in eval_results.items():
                    t.log_model_result(n, m)
                t.log_trust_scores(trust_scores)
                t.log_best_model(final_decision["best_model"],
                                 eval_results.get(final_decision["best_model"], {}),
                                 final_decision["trust_score"])
                t.end_run()
            except Exception:
                pass

        st.session_state.pipeline_result = {
            "status": "success", "task": det_task, "target_col": target_col,
            "decision": final_decision,
            "steps": {
                "training":    {"trained": list(trained_models.keys()), "training_times": train_times},
                "evaluation":  eval_results, "calibration_scores": cal_scores,
                "cv_results":  cv_results,   "leaderboard": leaderboard,
                "shap_global": shap_global,  "agreement":   agree_result,
                "data_quality": dq_info,     "recommendation": rec_report,
                "meta_features": meta,
                "preprocessing": {"train_shape": list(X_train_raw.shape),
                                  "test_shape":  list(X_test_raw.shape),
                                  "feature_names": feat_names},
            },
            "_trained_models": calibrated,
            "_feature_names":  feat_names,
            "_label_encoder":  le,
            "_X_test":         X_test_raw,
            "_y_test":         y_test,
        }
        progress.progress(100, text="✅ Pipeline complete!")
        st.balloons()

    except Exception as e:
        import traceback
        st.error(f"Pipeline error: {e}")
        with st.expander("Stack trace"):
            st.code(traceback.format_exc())
        st.stop()

# ── Post-run summary ──────────────────────────────────────────────────
if "pipeline_result" in st.session_state and st.session_state.pipeline_result:
    r        = st.session_state.pipeline_result
    decision = r.get("decision", {})
    steps    = r.get("steps", {})
    ts       = decision.get("trust_score", 0)
    bm       = decision.get("best_model", "—")
    pm       = decision.get("primary_metric", "f1")
    tc       = "#22c55e" if ts >= 0.7 else "#f59e0b" if ts >= 0.5 else "#ef4444"

    st.markdown("---")
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#060d1a,#080f1a);
                border:1px solid {tc}25;border-left:4px solid {tc};border-radius:14px;
                padding:1.4rem 1.8rem;margin-bottom:1.5rem;
                display:flex;align-items:center;gap:2.5rem;flex-wrap:wrap;">
      <div><div style="font-size:.65rem;text-transform:uppercase;letter-spacing:.1em;color:#334155;font-weight:600;">Best Model</div>
           <div style="font-size:1.4rem;font-weight:700;color:#f1f5f9;margin:.2rem 0;">{bm.replace('_',' ').title()}</div></div>
      <div style="width:1px;height:2.5rem;background:#0f2340;"></div>
      <div><div style="font-size:.65rem;text-transform:uppercase;letter-spacing:.1em;color:#334155;font-weight:600;">{pm.upper()}</div>
           <div style="font-size:1.4rem;font-weight:700;color:#60a5fa;margin:.2rem 0;">{decision.get('primary_score',0):.4f}</div></div>
      <div style="width:1px;height:2.5rem;background:#0f2340;"></div>
      <div><div style="font-size:.65rem;text-transform:uppercase;letter-spacing:.1em;color:#334155;font-weight:600;">Trust Score</div>
           <div style="font-size:1.4rem;font-weight:700;color:{tc};margin:.2rem 0;">{ts:.4f}
             <span style="font-size:.72rem;"> {decision.get('trust_label','')}</span></div></div>
      <div style="width:1px;height:2.5rem;background:#0f2340;"></div>
      <div><div style="font-size:.65rem;text-transform:uppercase;letter-spacing:.1em;color:#334155;font-weight:600;">Agreement</div>
           <div style="font-size:1.4rem;font-weight:700;color:#a78bfa;margin:.2rem 0;">
             {steps.get('agreement',{}).get('agreement_score',0):.4f}</div></div>
    </div>""", unsafe_allow_html=True)

    lb = steps.get("leaderboard", [])
    if lb:
        lb_df = pd.DataFrame(lb)
        if pm in lb_df.columns:
            lb_df = lb_df.sort_values(pm, ascending=True)
            fig   = px.bar(lb_df, x=pm, y="model", orientation="h",
                           color=pm,
                           color_continuous_scale=["#112240", "#3b82f6", "#8b5cf6"],
                           range_color=[lb_df[pm].min() * 0.95, lb_df[pm].max()],
                           title=f"Model Leaderboard — {pm.upper()}", text=pm)
            fig.update_traces(texttemplate="%{text:.4f}", textposition="outside",
                              marker_line_width=0)
            fig.update_layout(
                plot_bgcolor="#080f1a", paper_bgcolor="#080f1a",
                font=dict(color="#94a3b8", family="Inter"),
                title_font=dict(color="#e2e8f0", size=13),
                xaxis=dict(showgrid=True, gridcolor="#0f2340", tickfont=dict(size=11)),
                yaxis=dict(showgrid=False, tickfont=dict(size=11)),
                coloraxis_showscale=False,
                height=max(260, len(lb_df) * 50 + 80),
                margin=dict(l=10, r=70, t=45, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.success("✅ Done! Head to **🏆 Results** for the full breakdown.")
