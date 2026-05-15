"""
EMMDS Direction 3: Distribution Shift Robustness
=================================================
Research Question:
  "Does trust-based model selection degrade more gracefully under
   distribution shift than accuracy-based selection?"

Methodology:
  1. Train models on clean in-distribution data
  2. Generate shifted test sets at 4 magnitudes:
     δ = 0.5σ, 1σ, 2σ, 3σ (covariate shift of feature means)
  3. For each shift level, measure performance degradation
     per model and per selector
  4. Compare: does the trust-selected model degrade less than
     the accuracy-selected model as shift increases?
  5. Find trust score threshold above which models are shift-robust

This is practically the most important direction because distribution
shift is the primary cause of production ML failures.
"""

import sys, warnings, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.datasets import (
    load_breast_cancer, load_wine, load_iris, load_digits
)
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, brier_score_loss
from sklearn.base import clone

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.model_registry import get_all_models
from src.data_engine.meta_features import MetaFeatureExtractor
from src.data_engine.data_quality import DataQualityScorer
from src.decision.model_agreement import ModelAgreementEngine

OUT = Path("outputs/research/direction3")
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.25
CV_FOLDS     = 5
SHIFT_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]   # multiples of σ


# ══════════════════════════════════════════════════════════════════════
# COVARIATE SHIFT GENERATOR
# ══════════════════════════════════════════════════════════════════════

def apply_covariate_shift(X_test: np.ndarray, X_train: np.ndarray,
                           shift_magnitude: float,
                           rng: np.random.RandomState) -> np.ndarray:
    """
    Apply covariate shift by shifting feature means by shift_magnitude × σ.

    This simulates the most common real-world distribution shift:
    the marginal distribution P(X) changes while P(Y|X) stays the same.
    Examples: seasonal sensor drift, demographic change, new market segment.

    shift_magnitude = 0.0: no shift (in-distribution)
    shift_magnitude = 0.5: subtle shift (hard to detect)
    shift_magnitude = 1.0: moderate shift (detectable)
    shift_magnitude = 2.0: large shift (clearly different)
    shift_magnitude = 3.0: severe shift (completely different range)
    """
    if shift_magnitude == 0.0:
        return X_test.copy()

    # Compute per-feature std from training data
    feature_stds = X_train.std(axis=0)

    # Random shift direction (different per feature to be realistic)
    shift_direction = rng.choice([-1, 1], size=X_test.shape[1])

    # Apply shift
    shift_vector = shift_magnitude * feature_stds * shift_direction
    X_shifted    = X_test + shift_vector

    return X_shifted


# ══════════════════════════════════════════════════════════════════════
# MEASURE TRUST COMPONENTS AT TRAINING TIME
# ══════════════════════════════════════════════════════════════════════

def measure_trust_components(model, X_tr_s, X_te_s, y_tr, y_te,
                               X_all_s, y_all, agreement_score,
                               dq_score, base_model):
    """Compute all trust components for a single model."""
    # Accuracy
    test_f1 = float(f1_score(y_te, model.predict(X_te_s),
                             average='weighted', zero_division=0))

    # Calibration
    cal_score = 0.5
    try:
        from sklearn.calibration import CalibratedClassifierCV
        try:
            cm = CalibratedClassifierCV(estimator=model,
                                        method='isotonic', cv='prefit')
            cm.fit(X_tr_s, y_tr)
        except TypeError:
            cm = CalibratedClassifierCV(estimator=clone(base_model),
                                        method='isotonic', cv=3)
            cm.fit(X_tr_s, y_tr)
        proba = cm.predict_proba(X_te_s)
        classes = np.unique(y_te)
        if len(classes) == 2:
            bs = brier_score_loss(y_te, proba[:,1], pos_label=classes[1])
        else:
            bs = float(np.mean([
                brier_score_loss((y_te==c).astype(int), proba[:,i])
                for i, c in enumerate(classes)
            ]))
        cal_score = float(np.clip(1.0 - bs, 0, 1))
    except Exception:
        pass

    # CV stability
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_s = cross_val_score(clone(base_model), X_all_s, y_all,
                           cv=cv, scoring='f1_weighted', n_jobs=-1)
    stability = float(np.clip(
        1.0 - cv_s.std() / max(abs(cv_s.mean()), 1e-8), 0, 1))

    # Compute trust score
    trust = (0.25 * np.clip(test_f1, 0, 1)
           + 0.20 * np.clip(cal_score, 0, 1)
           + 0.20 * np.clip(agreement_score, 0, 1)
           + 0.20 * np.clip(dq_score, 0, 1)
           + 0.15 * np.clip(stability, 0, 1))

    return {
        'test_f1':    round(test_f1,    6),
        'cal_score':  round(cal_score,  6),
        'stability':  round(stability,  6),
        'trust_score': round(float(trust), 6),
    }


# ══════════════════════════════════════════════════════════════════════
# SINGLE DATASET SHIFT EXPERIMENT
# ══════════════════════════════════════════════════════════════════════

def run_shift_experiment(ds_name, df, target_col):
    """
    Full distribution shift experiment for one dataset.
    Returns per-model, per-shift-level performance.
    """
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y_raw = df[target_col]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X.values, y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y if len(np.unique(y)) > 1 else None
    )
    X_all = np.vstack([X_tr, X_te])
    y_all = np.concatenate([y_tr, y_te])

    scaler = StandardScaler()
    scaler.fit(X_tr)
    X_tr_s  = scaler.transform(X_tr)
    X_te_s  = scaler.transform(X_te)
    X_all_s = scaler.transform(X_all)

    dq_score = DataQualityScorer().score_dataset(df, target_col)

    # Train all models once on clean in-distribution data
    trained_models = {}
    base_models    = {}
    for mname, model in get_all_models(enabled_only=True).items():
        m = clone(model)
        try:
            m.fit(X_tr_s, y_tr)
            trained_models[mname] = m
            base_models[mname]    = model
        except Exception:
            continue

    if not trained_models:
        return []

    # Agreement (once, on clean test)
    try:
        ag = ModelAgreementEngine().compute(trained_models, X_te_s)
        agreement_score = ag.get('agreement_score', 0.5)
    except Exception:
        agreement_score = 0.5

    # Compute trust scores (on clean data — TRAINING TIME)
    trust_scores  = {}
    train_metrics = {}
    for mname, model in trained_models.items():
        comps = measure_trust_components(
            model, X_tr_s, X_te_s, y_tr, y_te,
            X_all_s, y_all, agreement_score, dq_score,
            base_models[mname]
        )
        trust_scores[mname]  = comps['trust_score']
        train_metrics[mname] = comps

    # ACCURACY selector = highest test accuracy at training time
    acc_selector   = max(trained_models.keys(),
                        key=lambda m: train_metrics[m]['test_f1'])
    trust_selector = max(trained_models.keys(),
                        key=lambda m: trust_scores[m])

    # Evaluate at each shift level
    rng = np.random.RandomState(RANDOM_STATE)
    rows = []

    for shift_mag in SHIFT_LEVELS:
        X_shifted = apply_covariate_shift(X_te, X_tr, shift_mag, rng)
        X_shifted_s = scaler.transform(X_shifted)

        for mname, model in trained_models.items():
            try:
                y_pred  = model.predict(X_shifted_s)
                shift_f1  = float(f1_score(
                    y_te, y_pred, average='weighted', zero_division=0))
                shift_acc = float(accuracy_score(y_te, y_pred))
            except Exception:
                shift_f1 = shift_acc = 0.0

            # Degradation vs baseline (shift=0)
            base_f1 = train_metrics[mname]['test_f1']

            rows.append({
                'dataset':        ds_name,
                'model':          mname,
                'shift_magnitude': shift_mag,
                'shift_f1':       round(shift_f1,   6),
                'shift_acc':      round(shift_acc,   6),
                'baseline_f1':    round(base_f1,     6),
                'degradation':    round(base_f1 - shift_f1, 6),
                'trust_score':    round(trust_scores[mname], 6),
                'is_acc_selected':   mname == acc_selector,
                'is_trust_selected': mname == trust_selector,
            })

    return rows


# ══════════════════════════════════════════════════════════════════════
# SHIFT ROBUSTNESS ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def analyse_shift_robustness(df_shift):
    """
    Key analyses:
    1. Trust vs Accuracy selector degradation curves
    2. Trust score as predictor of shift robustness
    3. Threshold: above what trust score are models shift-robust?
    """
    results = {}

    # 1. Selector degradation curves
    selector_curves = []
    for shift_mag, grp in df_shift.groupby('shift_magnitude'):
        acc_sel   = grp[grp['is_acc_selected']]['degradation'].values
        trust_sel = grp[grp['is_trust_selected']]['degradation'].values

        selector_curves.append({
            'shift_magnitude':     shift_mag,
            'acc_mean_degrad':     round(float(np.mean(acc_sel)),   6),
            'trust_mean_degrad':   round(float(np.mean(trust_sel)), 6),
            'acc_std_degrad':      round(float(np.std(acc_sel)),    6),
            'trust_std_degrad':    round(float(np.std(trust_sel)),  6),
            'trust_degrades_less': bool(np.mean(trust_sel) < np.mean(acc_sel)),
        })

    sel_df = pd.DataFrame(selector_curves)
    trust_wins = int(sel_df['trust_degrades_less'].sum())
    n_shifts   = len(sel_df[sel_df['shift_magnitude'] > 0])

    print(f"\n  Selector comparison across shift levels:")
    print(f"  {'Shift δ':8s}  {'Acc Degrad':12s}  {'Trust Degrad':12s}  {'Winner':10s}")
    print(f"  {'-'*55}")
    for _, r in sel_df.iterrows():
        w = 'TRUST ✅' if r['trust_degrades_less'] else 'ACCURACY'
        print(f"  δ={r['shift_magnitude']:.1f}σ     "
              f"{r['acc_mean_degrad']:+.4f}       "
              f"{r['trust_mean_degrad']:+.4f}       {w}")

    # 2. Trust as predictor of shift robustness
    # Total degradation across all shift levels per model-dataset
    total_degrad = df_shift.groupby(['dataset','model']).agg(
        total_degradation=('degradation', 'sum'),
        max_degradation=('degradation', 'max'),
        trust_score=('trust_score', 'first'),
        baseline_f1=('baseline_f1', 'first'),
    ).reset_index()

    r_trust, p_trust = stats.spearmanr(
        total_degrad['trust_score'], total_degrad['total_degradation'])
    r_acc,   p_acc   = stats.spearmanr(
        total_degrad['baseline_f1'], total_degrad['total_degradation'])

    print(f"\n  Trust score vs total degradation:   r={r_trust:+.4f}  p={p_trust:.6f}")
    print(f"  Accuracy    vs total degradation:   r={r_acc:+.4f}  p={p_acc:.6f}")
    print(f"  Trust stronger predictor of robustness: "
          f"{'✅ YES' if abs(r_trust) > abs(r_acc) else '❌ NO'}")

    # 3. Threshold analysis
    thresholds = [0.6, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    thresh_rows = []
    print(f"\n  Shift robustness by trust score threshold:")
    print(f"  {'Threshold':12s}  {'N models':10s}  {'Mean degrad':12s}  {'Max degrad':12s}")
    for t in thresholds:
        sub = total_degrad[total_degrad['trust_score'] >= t]
        if len(sub) == 0:
            continue
        thresh_rows.append({
            'threshold':      t,
            'n_models':       int(len(sub)),
            'mean_degrad':    round(float(sub['total_degradation'].mean()), 6),
            'max_degrad':     round(float(sub['max_degradation'].mean()), 6),
        })
        print(f"  trust ≥ {t:.2f}    {len(sub):5d}       "
              f"{sub['total_degradation'].mean():+.4f}       "
              f"{sub['max_degradation'].mean():+.4f}")

    thresh_df = pd.DataFrame(thresh_rows)

    results = {
        'selector_curves':    sel_df.to_dict('records'),
        'trust_wins_at_shift': int(trust_wins),
        'n_shift_levels':     int(n_shifts),
        'trust_vs_degradation': {
            'spearman_r': round(float(r_trust), 4),
            'p_value':    round(float(p_trust), 6),
            'significant': bool(p_trust < 0.05),
        },
        'accuracy_vs_degradation': {
            'spearman_r': round(float(r_acc), 4),
            'p_value':    round(float(p_acc), 6),
        },
        'trust_stronger_robustness_predictor': bool(abs(r_trust) > abs(r_acc)),
        'threshold_analysis': thresh_rows,
    }
    return results, sel_df, total_degrad, thresh_df


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def run_direction3():
    print("=" * 65)
    print("  DIRECTION 3: DISTRIBUTION SHIFT ROBUSTNESS")
    print("  Research: Does trust-based selection degrade more")
    print("            gracefully under covariate shift?")
    print("=" * 65)

    # Build datasets
    print("\n  Building datasets for shift experiments...")
    datasets = {}

    for name, loader in [
        ("breast_cancer", load_breast_cancer),
        ("wine",          load_wine),
        ("iris",          load_iris),
    ]:
        d = loader(as_frame=True)
        df = d.frame.copy(); df["target"] = d.target
        datasets[name] = (df, "target")

    synth_specs = [
        ("clean_baseline",  dict(n_samples=600, n_features=20,
                                  n_informative=12, n_redundant=4,
                                  flip_y=0.02, class_sep=1.0,
                                  n_clusters_per_class=1)),
        ("imbal_5x",        dict(n_samples=600, n_features=20,
                                  n_informative=12, n_redundant=4,
                                  flip_y=0.02, class_sep=1.0,
                                  weights=[0.833, 0.167],
                                  n_clusters_per_class=1)),
        ("high_noise",      dict(n_samples=600, n_features=20,
                                  n_informative=10, n_redundant=5,
                                  flip_y=0.15, class_sep=0.7,
                                  n_clusters_per_class=1)),
        ("high_dim",        dict(n_samples=500, n_features=50,
                                  n_informative=20, n_redundant=10,
                                  flip_y=0.03, class_sep=1.0,
                                  n_clusters_per_class=1)),
        ("small_n",         dict(n_samples=200, n_features=15,
                                  n_informative=8, n_redundant=4,
                                  flip_y=0.03, class_sep=1.0,
                                  n_clusters_per_class=1)),
    ]

    for name, spec in synth_specs:
        X, y = make_classification(**spec, random_state=RANDOM_STATE)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(spec['n_features'])])
        df["target"] = y
        datasets[f"synth_{name}"] = (df, "target")

    print(f"  {len(datasets)} datasets ready")

    # Run shift experiments
    print("\n  Running distribution shift experiments...")
    all_shift_rows = []
    t0 = time.time()

    for i, (ds_name, (df, target)) in enumerate(datasets.items()):
        print(f"    [{i+1}/{len(datasets)}] {ds_name} "
              f"({df.shape[0]} samples, {df.shape[1]-1} features)...")
        rows = run_shift_experiment(ds_name, df, target)
        all_shift_rows.extend(rows)

    print(f"  Done in {round(time.time()-t0, 1)}s")
    print(f"  Total rows: {len(all_shift_rows)}")

    df_shift = pd.DataFrame(all_shift_rows)
    df_shift.to_csv(OUT / "shift_results.csv", index=False)

    # Analyse
    print("\n  Analysing shift robustness...")
    robustness_results, sel_df, total_degrad, thresh_df = \
        analyse_shift_robustness(df_shift)

    # Per-dataset summary
    print("\n  Per-dataset: does trust win at large shift (δ≥2σ)?")
    per_ds_large_shift = (df_shift[df_shift['shift_magnitude'] >= 2.0]
                          .groupby('dataset'))
    ds_summary = []
    for ds, grp in per_ds_large_shift:
        acc_degrad   = float(grp[grp['is_acc_selected']]['degradation'].mean())
        trust_degrad = float(grp[grp['is_trust_selected']]['degradation'].mean())
        ds_summary.append({
            'dataset':     ds,
            'acc_degrad':  round(acc_degrad,   4),
            'trust_degrad':round(trust_degrad, 4),
            'trust_wins':  bool(trust_degrad < acc_degrad),
        })
        w = '✅ TRUST' if trust_degrad < acc_degrad else '  ACCURACY'
        print(f"    {ds:30s}  acc={acc_degrad:+.4f}  "
              f"trust={trust_degrad:+.4f}  {w}")

    ds_sum_df = pd.DataFrame(ds_summary)
    trust_ds_wins = int(ds_sum_df['trust_wins'].sum()) if len(ds_sum_df) else 0

    # Statistical test: is trust degradation < accuracy degradation?
    sel_nonzero = sel_df[sel_df['shift_magnitude'] > 0]
    if len(sel_nonzero) >= 3:
        diff = (sel_nonzero['acc_mean_degrad'] -
                sel_nonzero['trust_mean_degrad']).values
        if not np.allclose(diff, 0):
            stat, p = stats.wilcoxon(diff, alternative='greater')
        else:
            stat, p = 0.0, 1.0
        print(f"\n  Wilcoxon (H1: trust degrades less than accuracy):")
        print(f"  stat={stat:.4f}  p={p:.6f}  "
              f"{'REJECT H0 ✅' if p < 0.05 else 'FAIL TO REJECT H0'}")
    else:
        stat, p = 0.0, 1.0

    # Save results
    sel_df.to_csv(OUT / "selector_degradation_curves.csv", index=False)
    total_degrad.to_csv(OUT / "total_degradation_per_model.csv", index=False)
    thresh_df.to_csv(OUT / "threshold_analysis.csv", index=False)
    ds_sum_df.to_csv(OUT / "per_dataset_summary.csv", index=False)

    def _j(o):
        if isinstance(o, (np.bool_,)):    return bool(o)
        if isinstance(o, (np.integer,)):  return int(o)
        if isinstance(o, (np.floating,)):
            return None if (np.isnan(o) or np.isinf(o)) else float(o)
        if isinstance(o, np.ndarray):     return o.tolist()
        return str(o)

    final_results = {
        **robustness_results,
        'per_dataset_large_shift': ds_summary,
        'trust_wins_large_shift':  trust_ds_wins,
        'total_datasets':          len(ds_summary),
        'wilcoxon_stat':   float(stat),
        'wilcoxon_p':      float(p),
        'h0_rejected':     bool(p < 0.05),
        'key_finding': (
            f"Trust-based selection produces lower degradation at large shift "
            f"(δ≥2σ) on {trust_ds_wins}/{len(ds_summary)} datasets. "
            f"Trust score is {'a significant' if robustness_results['trust_vs_degradation']['significant'] else 'a non-significant'} "
            f"predictor of shift robustness "
            f"(r={robustness_results['trust_vs_degradation']['spearman_r']:.4f}, "
            f"p={robustness_results['trust_vs_degradation']['p_value']:.4f})."
        ),
    }

    with open(OUT / "direction3_results.json", "w") as f:
        json.dump(final_results, f, indent=2, default=_j)

    print(f"\n  Results saved → {OUT}/")
    print(f"\n  KEY FINDING:")
    print(f"  {final_results['key_finding']}")

    return final_results


if __name__ == "__main__":
    run_direction3()
