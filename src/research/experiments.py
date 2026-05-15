"""
EMMDS Research Experiment Engine
=================================
Validates all four research claims across 12 datasets.

Claims tested:
  A. Trust score predicts deployment risk better than accuracy alone
  B. Meta-features predict optimal model families
  C. Agreement correlates with reliability better than softmax confidence
  D. Calibration + agreement explain generalisation variance beyond accuracy

Each dataset × each model produces:
  - test_accuracy, train_accuracy, generalisation_gap
  - trust_score (5-component)
  - calibration_score, agreement_score, cv_stability
  - softmax_confidence (max predicted probability)
  - is_selected_by_accuracy (bool)
  - is_selected_by_trust (bool)
  - actual_best (ground truth: model with smallest gen gap)
"""

import sys, warnings, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.datasets import (
    load_breast_cancer, load_iris, load_wine, load_digits
)
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, accuracy_score, brier_score_loss
from sklearn.base import clone

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.model_registry import get_all_models
from src.calibration.calibrator import ModelCalibrator
from src.training.cross_validation import CrossValidator
from src.decision.model_agreement import ModelAgreementEngine
from src.data_engine.data_quality import DataQualityScorer
from src.data_engine.meta_features import MetaFeatureExtractor

OUT = Path("outputs/research")
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
CV_FOLDS     = 5
TEST_SIZE    = 0.25


# ══════════════════════════════════════════════════════════════════════
# DATASET COLLECTION  (12 datasets)
# ══════════════════════════════════════════════════════════════════════

def build_dataset_collection():
    datasets = {}

    # ── 4 real sklearn datasets ───────────────────────────────────
    for name, loader in [
        ("breast_cancer", load_breast_cancer),
        ("wine",          load_wine),
        ("iris",          load_iris),
        ("digits",        load_digits),
    ]:
        d = loader(as_frame=True)
        df = d.frame.copy(); df["target"] = d.target
        datasets[name] = dict(df=df, target="target", dtype="real",
            desc=f"sklearn {name} — {df.shape[0]} samples, {df.shape[1]-1} features, "
                 f"{int(df['target'].nunique())} classes")

    # ── 8 synthetic with controlled properties ────────────────────
    specs = [
        dict(name="synth_clean",        n=800,  f=20, inf=15, noise=0.00, w=None,         sep=1.5, nc=2, desc="Clean balanced — high separability"),
        dict(name="synth_imbal_10_1",   n=1000, f=20, inf=10, noise=0.02, w=[.91,.09],    sep=1.0, nc=2, desc="Severe imbalance 10:1 — accuracy inflated"),
        dict(name="synth_high_noise",   n=600,  f=20, inf=5,  noise=0.20, w=None,         sep=0.5, nc=2, desc="High noise 20% label flip — low agreement expected"),
        dict(name="synth_imbal_3_1",    n=700,  f=15, inf=10, noise=0.03, w=[.75,.25],    sep=1.2, nc=2, desc="Moderate imbalance 3:1"),
        dict(name="synth_high_dim",     n=400,  f=60, inf=20, noise=0.05, w=None,         sep=1.0, nc=2, desc="High dimensionality p/n=0.15 — KNN struggles"),
        dict(name="synth_multiclass4",  n=800,  f=20, inf=15, noise=0.02, w=None,         sep=1.0, nc=4, desc="4-class balanced — multi-class trust"),
        dict(name="synth_small_n150",   n=150,  f=10, inf=7,  noise=0.05, w=None,         sep=1.0, nc=2, desc="Small dataset n=150 — high CV variance"),
        dict(name="synth_noisy_imbal",  n=500,  f=20, inf=8,  noise=0.15, w=[.80,.20],    sep=0.7, nc=2, desc="Noisy + imbalanced — hardest case"),
    ]

    for s in specs:
        kw = dict(n_samples=s["n"], n_features=s["f"], n_informative=s["inf"],
                  flip_y=s["noise"], random_state=RANDOM_STATE, class_sep=s["sep"],
                  n_classes=s["nc"], n_clusters_per_class=1)
        if s["w"]:
            kw["weights"] = s["w"]
        if s["nc"] > 2:
            kw["n_redundant"] = min(3, s["f"] - s["inf"] - 1)
        else:
            kw["n_redundant"] = min(s["f"] - s["inf"] - 1, 5)

        X, y = make_classification(**kw)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(s["f"])])
        df["target"] = y
        datasets[s["name"]] = dict(df=df, target="target", dtype="synthetic", desc=s["desc"])

    return datasets


# ══════════════════════════════════════════════════════════════════════
# PER-MODEL MEASUREMENT
# ══════════════════════════════════════════════════════════════════════

def measure_single_model(name, model, X_train, X_test, y_train, y_test,
                          X_all, y_all, scaler):
    """Train + measure everything for one model on one dataset."""
    m = clone(model)

    # Scale
    X_tr_s = scaler.transform(X_train)
    X_te_s = scaler.transform(X_test)
    X_all_s = scaler.transform(X_all)

    # Fit
    t0 = time.time()
    m.fit(X_tr_s, y_train)
    train_time = round(time.time() - t0, 4)

    # Accuracy
    train_acc = float(accuracy_score(y_train, m.predict(X_tr_s)))
    test_acc  = float(accuracy_score(y_test,  m.predict(X_te_s)))
    gen_gap   = round(train_acc - test_acc, 6)   # deployment risk proxy

    # F1
    test_f1   = float(f1_score(y_test, m.predict(X_te_s), average="weighted", zero_division=0))

    # Softmax confidence (Option C)
    softmax_conf = None
    if hasattr(m, "predict_proba"):
        proba = m.predict_proba(X_te_s)
        softmax_conf = float(np.mean(np.max(proba, axis=1)))

    # Calibration score (1 - Brier)
    cal_score = 0.5
    try:
        from sklearn.base import clone as _clone
        try:
            cm = CalibratedClassifierCV(estimator=m, method="isotonic", cv="prefit")
        except TypeError:
            cm = CalibratedClassifierCV(estimator=_clone(m), method="isotonic", cv=3)
        cm.fit(X_tr_s, y_train)
        proba_cal = cm.predict_proba(X_te_s)
        classes = np.unique(y_test)
        if len(classes) == 2:
            bs = brier_score_loss(y_test, proba_cal[:, 1], pos_label=classes[1])
        else:
            bscores = []
            for i, c in enumerate(classes):
                bscores.append(brier_score_loss((y_test == c).astype(int), proba_cal[:, i]))
            bs = float(np.mean(bscores))
        cal_score = round(1.0 - bs, 6)
    except Exception:
        pass

    # CV stability
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(clone(model), X_all_s, y_all, cv=cv,
                                scoring="f1_weighted", n_jobs=-1)
    cv_mean  = float(np.mean(cv_scores))
    cv_std   = float(np.std(cv_scores))
    stability = float(np.clip(1.0 - cv_std / max(abs(cv_mean), 1e-8), 0, 1))

    return {
        "model":          name,
        "train_acc":      round(train_acc,  6),
        "test_acc":       round(test_acc,   6),
        "test_f1":        round(test_f1,    6),
        "gen_gap":        round(gen_gap,    6),
        "cal_score":      round(cal_score,  6),
        "cv_mean":        round(cv_mean,    6),
        "cv_std":         round(cv_std,     6),
        "stability":      round(stability,  6),
        "softmax_conf":   round(softmax_conf, 6) if softmax_conf else None,
        "train_time_s":   train_time,
    }

from sklearn.calibration import CalibratedClassifierCV


# ══════════════════════════════════════════════════════════════════════
# TRUST SCORE COMPUTATION
# ══════════════════════════════════════════════════════════════════════

def compute_trust(row, agreement_score, dq_score):
    """
    5-component trust score per model row.
    Weights: accuracy=0.25, calibration=0.20, agreement=0.20,
             data_quality=0.20, stability=0.15
    """
    acc  = np.clip(row["test_f1"],    0, 1)
    cal  = np.clip(row["cal_score"],  0, 1)
    agr  = np.clip(agreement_score,   0, 1)
    dq   = np.clip(dq_score,          0, 1)
    stab = np.clip(row["stability"],  0, 1)

    trust = 0.25*acc + 0.20*cal + 0.20*agr + 0.20*dq + 0.15*stab
    return round(float(trust), 6)


# ══════════════════════════════════════════════════════════════════════
# SINGLE DATASET EXPERIMENT
# ══════════════════════════════════════════════════════════════════════

def run_dataset_experiment(ds_name, ds_info):
    """Full experiment on one dataset. Returns list of per-model result dicts."""
    print(f"  Running: {ds_name}  [{ds_info['desc']}]")
    df    = ds_info["df"]
    tcol  = ds_info["target"]

    X = df.drop(columns=[tcol]).select_dtypes(include=[np.number])
    y_raw = df[tcol]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X.values, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=y if len(np.unique(y)) > 1 else None
    )
    X_all = np.vstack([X_tr_raw, X_te_raw])
    y_all = np.concatenate([y_tr, y_te])

    scaler = StandardScaler()
    scaler.fit(X_tr_raw)

    # Data quality + meta-features
    dq_score = DataQualityScorer().score_dataset(df, tcol, task="classification")

    # Extract meta-features
    meta = MetaFeatureExtractor().extract(df, tcol)

    # Per-model measurements
    models = get_all_models(enabled_only=True)
    rows   = []

    trained_fitted = {}
    for mname, model in models.items():
        try:
            row = measure_single_model(
                mname, model,
                X_tr_raw, X_te_raw, y_tr, y_te,
                X_all, y_all, scaler
            )
            rows.append(row)
            # Keep fitted model for agreement
            m2 = clone(model); m2.fit(scaler.transform(X_tr_raw), y_tr)
            trained_fitted[mname] = m2
        except Exception as e:
            print(f"    ⚠ {mname} failed: {e}")

    if not rows:
        return [], {}

    # Model agreement (computed once across all models)
    X_te_scaled = scaler.transform(X_te_raw)
    try:
        agree_result = ModelAgreementEngine().compute(trained_fitted, X_te_scaled)
        agreement_score = agree_result.get("agreement_score", 0.5)
        per_model_agree = agree_result.get("per_model_agreement", {})
    except Exception:
        agreement_score = 0.5
        per_model_agree = {}

    # Add trust score + agreement to each row
    for row in rows:
        row["agreement_score"] = round(agreement_score, 6)
        row["per_model_agree"] = round(per_model_agree.get(row["model"], agreement_score), 6)
        row["dq_score"]        = round(dq_score, 6)
        row["trust_score"]     = compute_trust(row, agreement_score, dq_score)
        row["dataset"]         = ds_name
        row["dtype"]           = ds_info["dtype"]
        row["n_samples"]       = len(df)
        row["n_features"]      = X.shape[1]
        row["n_classes"]       = int(y_raw.nunique())

    # Ground truth: actual best = model with smallest gen_gap (most reliable)
    best_by_gap      = min(rows, key=lambda r: r["gen_gap"])["model"]
    best_by_accuracy = max(rows, key=lambda r: r["test_acc"])["model"]
    best_by_trust    = max(rows, key=lambda r: r["trust_score"])["model"]
    best_by_f1       = max(rows, key=lambda r: r["test_f1"])["model"]

    for row in rows:
        row["selected_by_accuracy"] = (row["model"] == best_by_accuracy)
        row["selected_by_trust"]    = (row["model"] == best_by_trust)
        row["actual_best_gen"]      = (row["model"] == best_by_gap)
        row["actual_best_f1"]       = (row["model"] == best_by_f1)

    dataset_summary = {
        "dataset":          ds_name,
        "dtype":            ds_info["dtype"],
        "desc":             ds_info["desc"],
        "n_samples":        len(df),
        "n_features":       X.shape[1],
        "n_classes":        int(y_raw.nunique()),
        "dq_score":         round(dq_score, 6),
        "agreement_score":  round(agreement_score, 6),
        "best_by_accuracy": best_by_accuracy,
        "best_by_trust":    best_by_trust,
        "best_by_gen_gap":  best_by_gap,
        "accuracy_correct": (best_by_accuracy == best_by_gap),
        "trust_correct":    (best_by_trust    == best_by_gap),
        "disagreement":     (best_by_accuracy != best_by_trust),
        "meta_features":    meta,
        "models_run":       len(rows),
    }

    return rows, dataset_summary


# ══════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT RUNNER
# ══════════════════════════════════════════════════════════════════════

def run_all_experiments():
    print("=" * 60)
    print("  EMMDS RESEARCH EXPERIMENTS")
    print("  Running all 4 research directions")
    print("=" * 60)

    datasets = build_dataset_collection()
    all_rows = []
    all_summaries = []

    t_start = time.time()
    for ds_name, ds_info in datasets.items():
        rows, summary = run_dataset_experiment(ds_name, ds_info)
        all_rows.extend(rows)
        if summary:
            all_summaries.append(summary)

    df_results  = pd.DataFrame(all_rows)
    df_summaries = pd.DataFrame(all_summaries)

    print(f"\n  Total experiments: {len(all_rows)} ({len(datasets)} datasets × ~7 models)")
    print(f"  Total time: {round(time.time()-t_start,1)}s")

    # Save raw results
    df_results.to_csv(OUT / "raw_results.csv", index=False)
    df_summaries.to_csv(OUT / "dataset_summaries.csv", index=False)

    return df_results, df_summaries, datasets


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def analyse_claim_A(df):
    """
    Claim A: Trust score predicts deployment risk (gen_gap)
    better than accuracy alone.

    Metric: Spearman correlation between predictor and gen_gap.
    Higher |correlation| with gen_gap = better predictor.
    """
    print("\n" + "═"*60)
    print("  CLAIM A: Trust vs Accuracy as Deployment Risk Predictor")
    print("═"*60)

    results = {}
    predictors = {
        "test_accuracy": "test_acc",
        "trust_score":   "trust_score",
        "calibration":   "cal_score",
        "cv_stability":  "stability",
        "agreement":     "agreement_score",
    }

    for label, col in predictors.items():
        if col not in df.columns:
            continue
        valid = df[[col, "gen_gap"]].dropna()
        r, p  = stats.spearmanr(valid[col], valid["gen_gap"])
        results[label] = {
            "spearman_r":  round(float(r), 4),
            "p_value":     round(float(p), 6),
            "significant": p < 0.05,
            "n":           len(valid),
        }
        direction = "↓ gap" if r < 0 else "↑ gap"
        sig = "✅" if p < 0.05 else "—"
        print(f"  {label:20s}  r={r:+.4f}  p={p:.4f}  {sig}  {direction}")

    # Per-dataset: did trust select better than accuracy?
    selector_comparison = []
    for ds, grp in df.groupby("dataset"):
        if len(grp) < 2:
            continue
        best_acc_row   = grp.loc[grp["test_acc"].idxmax()]
        best_trust_row = grp.loc[grp["trust_score"].idxmax()]
        actual_best    = grp.loc[grp["gen_gap"].idxmin()]

        acc_gap   = float(best_acc_row["gen_gap"])
        trust_gap = float(best_trust_row["gen_gap"])
        actual_gap = float(actual_best["gen_gap"])

        selector_comparison.append({
            "dataset":             ds,
            "accuracy_selected":   best_acc_row["model"],
            "trust_selected":      best_trust_row["model"],
            "actual_best":         actual_best["model"],
            "accuracy_gen_gap":    round(acc_gap, 6),
            "trust_gen_gap":       round(trust_gap, 6),
            "actual_best_gap":     round(actual_gap, 6),
            "trust_wins":          trust_gap < acc_gap,
            "disagreement":        best_acc_row["model"] != best_trust_row["model"],
        })

    comp_df = pd.DataFrame(selector_comparison)
    trust_wins = int(comp_df["trust_wins"].sum())
    n_total    = len(comp_df)
    disagree   = int(comp_df["disagreement"].sum())

    print(f"\n  Selector comparison ({n_total} datasets):")
    print(f"  Trust selector wins:    {trust_wins}/{n_total} datasets ({100*trust_wins//n_total}%)")
    print(f"  Disagreement cases:     {disagree}/{n_total}")

    print(f"\n  Disagreement detail (trust ≠ accuracy):")
    for _, r in comp_df[comp_df["disagreement"]].iterrows():
        winner = "TRUST ✅" if r["trust_wins"] else "ACCURACY"
        print(f"    {r['dataset']:25s}  acc→{r['accuracy_selected']:20s}(gap={r['accuracy_gen_gap']:.4f})"
              f"  trust→{r['trust_selected']:20s}(gap={r['trust_gen_gap']:.4f})  winner={winner}")

    comp_df.to_csv(OUT / "claim_A_selector_comparison.csv", index=False)
    with open(OUT / "claim_A_correlations.json", "w") as f:
        json.dump(results, f, indent=2)

    return results, comp_df


def analyse_claim_B(df, summaries_df, datasets):
    """
    Claim B: Meta-features predict optimal model family.
    For each dataset we know the actual best model.
    We build a simple meta-learner and measure its accuracy.
    """
    print("\n" + "═"*60)
    print("  CLAIM B: Meta-Learning for Model Selection")
    print("═"*60)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import LabelEncoder as LE

    # Build meta-dataset: features = meta-features, label = best model name
    meta_rows = []
    for _, row in summaries_df.iterrows():
        mf = row.get("meta_features", {})
        if not mf:
            continue
        meta_rows.append({
            "dataset":          row["dataset"],
            "best_model":       row["best_by_gen_gap"],
            "n_samples":        mf.get("n_samples", 0),
            "n_features":       mf.get("n_features", 0),
            "imbalance_ratio":  mf.get("imbalance_ratio") or 1.0,
            "missing_ratio":    mf.get("missing_ratio", 0),
            "avg_correlation":  mf.get("avg_abs_correlation", 0),
            "noise_estimate":   mf.get("noise_estimate", 0),
            "dim_ratio":        mf.get("dimensionality_ratio", 0),
            "mean_skewness":    mf.get("mean_skewness", 0),
            "n_classes":        mf.get("n_classes", 2),
        })

    meta_df = pd.DataFrame(meta_rows)
    if len(meta_df) < 5:
        print("  ⚠ Not enough datasets for meta-learning (need ≥5)")
        return {}, meta_df

    feature_cols = ["n_samples","n_features","imbalance_ratio","missing_ratio",
                    "avg_correlation","noise_estimate","dim_ratio","mean_skewness","n_classes"]
    X_meta = meta_df[feature_cols].fillna(0).values
    le_meta = LE()
    y_meta  = le_meta.fit_transform(meta_df["best_model"])

    # Leave-one-out cross-validation on the meta-learner
    loo  = LeaveOneOut()
    meta_model = RandomForestClassifier(n_estimators=100, random_state=42)
    correct = 0
    loo_results = []

    for train_idx, test_idx in loo.split(X_meta):
        if len(np.unique(y_meta[train_idx])) < 2:
            continue
        meta_model.fit(X_meta[train_idx], y_meta[train_idx])
        pred  = meta_model.predict(X_meta[test_idx])[0]
        truth = y_meta[test_idx][0]
        is_correct = (pred == truth)
        correct += int(is_correct)
        loo_results.append({
            "dataset":    meta_df.iloc[test_idx[0]]["dataset"],
            "true_best":  le_meta.inverse_transform([truth])[0],
            "predicted":  le_meta.inverse_transform([pred])[0],
            "correct":    is_correct,
        })

    n_loo = len(loo_results)
    meta_acc = correct / max(n_loo, 1)

    # Random baseline
    from collections import Counter
    counts = Counter(meta_df["best_model"])
    majority = counts.most_common(1)[0][0]
    majority_acc = counts[majority] / len(meta_df)

    print(f"  Meta-learner LOO accuracy: {meta_acc:.3f} ({correct}/{n_loo})")
    print(f"  Random majority baseline:  {majority_acc:.3f}")
    print(f"  Improvement over baseline: {(meta_acc - majority_acc):+.3f}")

    # Feature importance
    meta_model.fit(X_meta, y_meta)
    imp = pd.DataFrame({
        "feature":    feature_cols,
        "importance": meta_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print(f"\n  Top meta-features for model selection:")
    for _, r in imp.head(5).iterrows():
        print(f"    {r['feature']:20s}  {r['importance']:.4f}")

    loo_df = pd.DataFrame(loo_results)
    loo_df.to_csv(OUT / "claim_B_meta_learning_loo.csv", index=False)
    imp.to_csv(OUT / "claim_B_feature_importance.csv", index=False)

    result = {
        "meta_learner_accuracy": round(meta_acc, 4),
        "majority_baseline":     round(majority_acc, 4),
        "improvement":           round(meta_acc - majority_acc, 4),
        "n_datasets":            n_loo,
        "top_features":          imp.head(5).to_dict("records"),
    }
    with open(OUT / "claim_B_results.json", "w") as f:
        json.dump(result, f, indent=2)

    return result, loo_df


def analyse_claim_C(df):
    """
    Claim C: Cross-model agreement correlates with reliability
    better than individual softmax confidence.

    We compare:
      softmax_conf vs gen_gap  (individual model confidence)
      agreement_score vs gen_gap (cross-model agreement)
    """
    print("\n" + "═"*60)
    print("  CLAIM C: Agreement vs Softmax as Reliability Proxy")
    print("═"*60)

    valid = df[["softmax_conf","agreement_score","gen_gap","trust_score"]].dropna()

    r_soft, p_soft = stats.spearmanr(valid["softmax_conf"],    valid["gen_gap"])
    r_agr,  p_agr  = stats.spearmanr(valid["agreement_score"], valid["gen_gap"])
    r_trust, p_trust = stats.spearmanr(valid["trust_score"],   valid["gen_gap"])

    print(f"  Softmax conf  vs gen_gap:  r={r_soft:+.4f}  p={p_soft:.4f}")
    print(f"  Agreement     vs gen_gap:  r={r_agr:+.4f}  p={p_agr:.4f}")
    print(f"  Trust score   vs gen_gap:  r={r_trust:+.4f}  p={p_trust:.4f}")

    agreement_better = abs(r_agr) > abs(r_soft)
    print(f"\n  Agreement better than softmax: {'✅ YES' if agreement_better else '❌ NO'}")

    # Per-dataset analysis
    per_ds = []
    for ds, grp in df.groupby("dataset"):
        g = grp[["softmax_conf","agreement_score","gen_gap"]].dropna()
        if len(g) < 3:
            continue
        rs, _ = stats.spearmanr(g["softmax_conf"],    g["gen_gap"])
        ra, _ = stats.spearmanr(g["agreement_score"], g["gen_gap"])
        per_ds.append({"dataset": ds,
                       "softmax_r": round(float(rs),4),
                       "agreement_r": round(float(ra),4),
                       "agreement_wins": abs(ra) > abs(rs)})

    per_ds_df = pd.DataFrame(per_ds)
    agr_wins = int(per_ds_df["agreement_wins"].sum()) if len(per_ds_df) else 0
    print(f"  Per-dataset: agreement wins in {agr_wins}/{len(per_ds_df)} datasets")

    per_ds_df.to_csv(OUT / "claim_C_per_dataset.csv", index=False)

    result = {
        "softmax_spearman":    round(float(r_soft), 4),
        "agreement_spearman":  round(float(r_agr),  4),
        "trust_spearman":      round(float(r_trust), 4),
        "softmax_p":           round(float(p_soft),  6),
        "agreement_p":         round(float(p_agr),   6),
        "agreement_better_than_softmax": bool(agreement_better),
        "datasets_agreement_wins": agr_wins,
        "datasets_total": len(per_ds_df),
    }
    with open(OUT / "claim_C_results.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def analyse_claim_D(df):
    """
    Claim D: Calibration + agreement jointly explain gen_gap
    variance beyond accuracy alone.

    We compare R² of three regression models:
      M1: gen_gap ~ accuracy_only
      M2: gen_gap ~ calibration + agreement
      M3: gen_gap ~ accuracy + calibration + agreement (full)
    """
    print("\n" + "═"*60)
    print("  CLAIM D: Calibration-Agreement-Accuracy Triangle")
    print("═"*60)

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.model_selection import cross_val_score

    valid = df[["test_acc","cal_score","agreement_score","stability","trust_score","gen_gap"]].dropna()

    models_spec = {
        "M1_accuracy_only":       ["test_acc"],
        "M2_cal_agreement":       ["cal_score","agreement_score"],
        "M3_full":                ["test_acc","cal_score","agreement_score","stability"],
        "M4_trust_only":          ["trust_score"],
    }

    results = {}
    print(f"\n  Model               R²(CV)   Features")
    print(f"  {'-'*55}")
    for mname, feats in models_spec.items():
        Xm = valid[feats].values
        ym = valid["gen_gap"].values
        lm = LinearRegression()
        cv_r2 = cross_val_score(lm, Xm, ym, cv=5, scoring="r2")
        r2 = float(np.mean(cv_r2))
        lm.fit(Xm, ym)
        results[mname] = {
            "r2_cv_mean": round(r2, 4),
            "r2_cv_std":  round(float(np.std(cv_r2)), 4),
            "features":   feats,
        }
        print(f"  {mname:25s}  {r2:+.4f}   {feats}")

    r2_acc   = results["M1_accuracy_only"]["r2_cv_mean"]
    r2_cal_agr = results["M2_cal_agreement"]["r2_cv_mean"]
    r2_full  = results["M3_full"]["r2_cv_mean"]

    print(f"\n  Calibration+Agreement explains {r2_cal_agr:.4f} R² vs accuracy-only {r2_acc:.4f}")
    print(f"  Full model R²: {r2_full:.4f}  (improvement over accuracy-only: {r2_full-r2_acc:+.4f})")

    with open(OUT / "claim_D_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ══════════════════════════════════════════════════════════════════════
# ABLATION STUDY
# ══════════════════════════════════════════════════════════════════════

def run_ablation_study(df):
    """
    Ablation: systematically zero out each trust component
    and measure how much selector quality drops.

    For each ablation condition we re-rank models using the
    modified trust score and check if the right model is selected.
    """
    print("\n" + "═"*60)
    print("  ABLATION STUDY: Component Contribution Analysis")
    print("═"*60)

    conditions = {
        "Full System":            dict(w_acc=0.25, w_cal=0.20, w_agr=0.20, w_dq=0.20, w_stab=0.15),
        "No Calibration":         dict(w_acc=0.35, w_cal=0.00, w_agr=0.25, w_dq=0.25, w_stab=0.15),
        "No Agreement":           dict(w_acc=0.30, w_cal=0.25, w_agr=0.00, w_dq=0.30, w_stab=0.15),
        "No Data Quality":        dict(w_acc=0.30, w_cal=0.25, w_agr=0.25, w_dq=0.00, w_stab=0.20),
        "No Stability":           dict(w_acc=0.30, w_cal=0.25, w_agr=0.25, w_dq=0.20, w_stab=0.00),
        "Accuracy Only":          dict(w_acc=1.00, w_cal=0.00, w_agr=0.00, w_dq=0.00, w_stab=0.00),
        "Equal Weights":          dict(w_acc=0.20, w_cal=0.20, w_agr=0.20, w_dq=0.20, w_stab=0.20),
    }

    ablation_rows = []
    for cond_name, weights in conditions.items():
        correct_selections = 0
        total_datasets     = 0
        mean_gap_selected  = []

        for ds, grp in df.groupby("dataset"):
            if len(grp) < 2:
                continue
            total_datasets += 1

            # Recompute trust with ablated weights
            def ablated_trust(row):
                return (weights["w_acc"]  * np.clip(row["test_f1"],        0, 1)
                      + weights["w_cal"]  * np.clip(row["cal_score"],       0, 1)
                      + weights["w_agr"]  * np.clip(row["agreement_score"], 0, 1)
                      + weights["w_dq"]   * np.clip(row["dq_score"],        0, 1)
                      + weights["w_stab"] * np.clip(row["stability"],       0, 1))

            grp = grp.copy()
            grp["ablated_trust"] = grp.apply(ablated_trust, axis=1)
            selected   = grp.loc[grp["ablated_trust"].idxmax()]
            actual_best = grp.loc[grp["gen_gap"].idxmin()]

            is_correct = (selected["model"] == actual_best["model"])
            correct_selections += int(is_correct)
            mean_gap_selected.append(float(selected["gen_gap"]))

        selection_acc  = correct_selections / max(total_datasets, 1)
        mean_gap       = np.mean(mean_gap_selected) if mean_gap_selected else 0

        ablation_rows.append({
            "condition":           cond_name,
            "selection_accuracy":  round(selection_acc, 4),
            "mean_gen_gap":        round(mean_gap, 6),
            "correct":             correct_selections,
            "total":               total_datasets,
            **{f"w_{k}": v for k, v in weights.items()},
        })
        print(f"  {cond_name:25s}  selection_acc={selection_acc:.3f}  mean_gap={mean_gap:.6f}")

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(OUT / "ablation_study.csv", index=False)
    return ablation_df


# ══════════════════════════════════════════════════════════════════════
# WEIGHT SENSITIVITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def run_weight_sensitivity(df):
    """
    Vary each weight from 0 to 0.6 in 0.1 steps while keeping
    others proportional. Find optimal weights empirically.
    This is the 'novel technical idea' — we don't just assert weights,
    we derive them from the data.
    """
    print("\n" + "═"*60)
    print("  WEIGHT SENSITIVITY ANALYSIS")
    print("═"*60)

    components = ["acc", "cal", "agr", "dq", "stab"]
    col_map    = {"acc": "test_f1", "cal": "cal_score",
                  "agr": "agreement_score", "dq": "dq_score", "stab": "stability"}
    weight_range = np.arange(0.0, 0.61, 0.1)

    sensitivity_rows = []
    best_config = None
    best_score  = -np.inf

    for comp in components:
        for w_focal in weight_range:
            remaining = max(1.0 - w_focal, 0.0)
            other = [c for c in components if c != comp]
            w_others = remaining / len(other)

            weights = {c: w_others for c in other}
            weights[comp] = w_focal

            correct = 0; total = 0
            for ds, grp in df.groupby("dataset"):
                if len(grp) < 2: continue
                total += 1
                def score(row):
                    return sum(weights[c] * np.clip(row[col_map[c]], 0, 1)
                               for c in components)
                grp2 = grp.copy()
                grp2["w_trust"] = grp2.apply(score, axis=1)
                sel  = grp2.loc[grp2["w_trust"].idxmax()]
                best = grp2.loc[grp2["gen_gap"].idxmin()]
                correct += int(sel["model"] == best["model"])

            sel_acc = correct / max(total, 1)
            sensitivity_rows.append({
                "varied_component": comp,
                "focal_weight": round(w_focal, 2),
                "selection_accuracy": round(sel_acc, 4),
                **{f"w_{c}": round(weights[c], 4) for c in components},
            })

            if sel_acc > best_score:
                best_score  = sel_acc
                best_config = dict(component=comp, weight=w_focal, config=weights.copy())

    sens_df = pd.DataFrame(sensitivity_rows)
    sens_df.to_csv(OUT / "weight_sensitivity.csv", index=False)

    print(f"  Best weight config found: {best_config}")
    print(f"  Best selection accuracy:  {best_score:.4f}")
    print(f"\n  Current EMMDS weights vs empirically optimal:")
    current = {"acc":0.25,"cal":0.20,"agr":0.20,"dq":0.20,"stab":0.15}
    for c in components:
        opt = best_config["config"][c] if best_config else "—"
        print(f"    {c:6s}  current={current[c]:.2f}  optimal≈{opt:.2f}")

    return sens_df, best_config


# ══════════════════════════════════════════════════════════════════════
# BASELINE COMPARISON
# ══════════════════════════════════════════════════════════════════════

def run_baseline_comparison(df):
    """
    Compare EMMDS trust selector against:
      1. Random selection
      2. Accuracy-only selection
      3. F1-only selection
    """
    print("\n" + "═"*60)
    print("  BASELINE COMPARISON")
    print("═"*60)

    selectors = {
        "Random":         lambda g: g.sample(1, random_state=42).iloc[0],
        "Accuracy Only":  lambda g: g.loc[g["test_acc"].idxmax()],
        "F1 Only":        lambda g: g.loc[g["test_f1"].idxmax()],
        "EMMDS Trust":    lambda g: g.loc[g["trust_score"].idxmax()],
    }

    baseline_rows = []
    for sel_name, selector in selectors.items():
        correct = 0; total = 0; gaps = []
        for ds, grp in df.groupby("dataset"):
            if len(grp) < 2: continue
            total += 1
            selected  = selector(grp)
            actual_best = grp.loc[grp["gen_gap"].idxmin()]
            correct += int(selected["model"] == actual_best["model"])
            gaps.append(float(selected["gen_gap"]))

        sel_acc = correct / max(total, 1)
        mean_gap = np.mean(gaps)
        baseline_rows.append({
            "selector":           sel_name,
            "selection_accuracy": round(sel_acc, 4),
            "mean_gen_gap":       round(mean_gap, 6),
            "correct":            correct,
            "total":              total,
        })
        print(f"  {sel_name:20s}  accuracy={sel_acc:.3f}  mean_gap={mean_gap:.6f}")

    base_df = pd.DataFrame(baseline_rows)
    base_df.to_csv(OUT / "baseline_comparison.csv", index=False)
    return base_df


# ══════════════════════════════════════════════════════════════════════
# HYPOTHESIS TEST
# ══════════════════════════════════════════════════════════════════════

def run_hypothesis_tests(df, comp_df):
    """
    Formal statistical tests for the thesis.

    H0: Trust score does not predict generalisation gap better than accuracy.
    H1: Trust score is a significantly stronger predictor.

    Test: Wilcoxon signed-rank test on per-dataset gen_gap of
    trust-selected vs accuracy-selected models.
    """
    print("\n" + "═"*60)
    print("  FORMAL HYPOTHESIS TESTS")
    print("═"*60)

    # Paired test: gen_gap(trust-selected) vs gen_gap(accuracy-selected)
    trust_gaps = comp_df["trust_gen_gap"].values
    acc_gaps   = comp_df["accuracy_gen_gap"].values

    if len(trust_gaps) >= 5:
        stat, p = stats.wilcoxon(trust_gaps, acc_gaps, alternative="less")
        print(f"\n  Wilcoxon signed-rank (H1: trust_gap < acc_gap)")
        print(f"  Statistic: {stat:.4f}   p-value: {p:.6f}")
        print(f"  Result: {'REJECT H0 ✅ (trust significantly better)' if p < 0.05 else 'FAIL TO REJECT H0'}")
    else:
        stat, p = 0, 1.0
        print("  ⚠ Not enough paired samples for Wilcoxon — using t-test")
        stat, p = stats.ttest_rel(trust_gaps, acc_gaps)
        print(f"  Paired t-test p-value: {p:.6f}")

    # Spearman correlation significance test
    r_trust, p_trust = stats.spearmanr(df["trust_score"].dropna(), df["gen_gap"].dropna())
    r_acc,   p_acc   = stats.spearmanr(df["test_acc"].dropna(),    df["gen_gap"].dropna())

    print(f"\n  Spearman trust↔gen_gap:    r={r_trust:+.4f}  p={p_trust:.6f}")
    print(f"  Spearman accuracy↔gen_gap: r={r_acc:+.4f}  p={p_acc:.6f}")

    results = {
        "wilcoxon_stat": float(stat),
        "wilcoxon_p":    float(p),
        "h0_rejected":   bool(p < 0.05),
        "trust_spearman_r": round(float(r_trust), 4),
        "trust_spearman_p": round(float(p_trust), 6),
        "acc_spearman_r":   round(float(r_acc),   4),
        "acc_spearman_p":   round(float(p_acc),   6),
        "conclusion": ("Trust score is a statistically significant predictor of deployment risk "
                       f"(r={r_trust:.4f}, p={p_trust:.4f})" if p_trust < 0.05
                       else "No significant correlation found."),
    }

    with open(OUT / "hypothesis_tests.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ══════════════════════════════════════════════════════════════════════
# MASTER RESULTS COMPILER
# ══════════════════════════════════════════════════════════════════════

def compile_master_results(claim_A, claim_B, claim_C, claim_D,
                            ablation_df, baseline_df, hyp_tests,
                            df_results, df_summaries):
    """Compile all results into one JSON for report generation."""
    master = {
        "generated_at":  datetime.now().isoformat(),
        "n_datasets":    int(df_summaries.shape[0]),
        "n_experiments": int(df_results.shape[0]),

        "claim_A": {
            "title": "Trust score predicts deployment risk better than accuracy",
            "correlations":   claim_A[0],
            "selector_wins":  int(claim_A[1]["trust_wins"].sum()),
            "selector_total": int(len(claim_A[1])),
            "selector_win_rate": round(claim_A[1]["trust_wins"].mean(), 4),
            "disagreement_cases": int(claim_A[1]["disagreement"].sum()),
        },
        "claim_B": {
            "title": "Meta-features predict optimal model family",
            **claim_B[0],
        },
        "claim_C": {
            "title": "Agreement correlates with reliability better than softmax confidence",
            **claim_C,
        },
        "claim_D": {
            "title": "Calibration + agreement explain generalisation variance beyond accuracy",
            **claim_D,
        },
        "ablation": ablation_df.to_dict("records"),
        "baselines": baseline_df.to_dict("records"),
        "hypothesis_tests": hyp_tests,
        "dataset_summaries": df_summaries.fillna("").to_dict("records"),
    }

    from datetime import datetime
    with open(OUT / "master_results.json", "w") as f:
        json.dump(master, f, indent=2, default=str)

    print(f"\n  Master results saved → {OUT}/master_results.json")
    return master


if __name__ == "__main__":
    from datetime import datetime
    import json

    df_results, df_summaries, datasets = run_all_experiments()

    claim_A_corr, claim_A_comp = analyse_claim_A(df_results)
    claim_B_result, _           = analyse_claim_B(df_results, df_summaries, datasets)
    claim_C_result              = analyse_claim_C(df_results)
    claim_D_result              = analyse_claim_D(df_results)
    ablation_df                 = run_ablation_study(df_results)
    sens_df, best_weights       = run_weight_sensitivity(df_results)
    baseline_df                 = run_baseline_comparison(df_results)
    hyp_tests                   = run_hypothesis_tests(df_results, claim_A_comp)

    master = compile_master_results(
        (claim_A_corr, claim_A_comp),
        (claim_B_result, {}),
        claim_C_result,
        claim_D_result,
        ablation_df,
        baseline_df,
        hyp_tests,
        df_results,
        df_summaries,
    )

    print("\n" + "="*60)
    print("  ALL EXPERIMENTS COMPLETE")
    print("="*60)
