"""
Trust Score Calibration Study  v1.0
=====================================
Answers: Is the EMMDS trust score itself calibrated?

If a model gets trust=0.80, does it actually deploy safely 80% of the time?
This is a meta-calibration question — the first systematic study of its kind.

Method
------
1. Run EMMDS on all 45 real datasets.
2. For each dataset: record predicted trust score T and actual deployment
   outcome (best achievable F1 on test set, normalised to [0,1]).
3. Bin trust scores into deciles; compute mean actual outcome per bin.
4. Compute Trust-ECE (same formula as classifier ECE, applied to trust).
5. Fit a Platt-scaling recalibration curve.

Key findings possible:
  - Trust is well-calibrated       → trust=0.8 → 80% deployment success
  - Trust is over-confident        → actual outcomes systematically lower
  - Trust is under-confident       → actual outcomes systematically higher
  - Trust-ECE < 0.05               → "calibrated" (publication threshold)
"""

import json
import numpy as np
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone
from scipy import stats
from scipy.special import expit  # sigmoid

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT  = ROOT / "outputs" / "research"
OUT.mkdir(parents=True, exist_ok=True)

TRUST_W = dict(accuracy=0.05, calibration=0.10, agreement=0.10,
               data_quality=0.35, stability=0.40)


# ─────────────────────────────────────────────────────────────
# Trust + outcome computation
# ─────────────────────────────────────────────────────────────

def _run_dataset(X: np.ndarray, y: np.ndarray,
                 name: str, seed: int = 42) -> Dict:
    """
    Returns predicted trust and actual deployment outcome for one dataset.
    Actual outcome = best-model F1 (oracle), normalised to [0,1].
    """
    col_med = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        X[np.isnan(X[:, j]), j] = col_med[j]
    sc = StandardScaler()
    X  = sc.fit_transform(X)

    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=seed)
    except Exception:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=seed)

    models = {
        "lr":   LogisticRegression(max_iter=300, random_state=seed),
        "rf":   RandomForestClassifier(n_estimators=40, random_state=seed),
        "gbm":  GradientBoostingClassifier(n_estimators=40, random_state=seed),
        "tree": DecisionTreeClassifier(max_depth=5, random_state=seed),
        "knn":  KNeighborsClassifier(n_neighbors=5),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    all_f1, all_cal, all_stab = [], [], []
    all_preds: Dict[str, np.ndarray] = {}

    for mname, m in models.items():
        try:
            m.fit(X_tr, y_tr)
            f1 = float(f1_score(y_te, m.predict(X_te),
                                 average="weighted", zero_division=0))

            cal = 0.5
            if hasattr(m, "predict_proba"):
                proba = m.predict_proba(X_te)
                classes = np.unique(y_te)
                if len(classes) == 2:
                    cal = float(np.clip(1 - brier_score_loss(y_te, proba[:, 1]), 0, 1))
                else:
                    cal = float(np.clip(1 - np.mean([
                        brier_score_loss((y_te == c).astype(int), proba[:, i])
                        for i, c in enumerate(classes)]), 0, 1))

            cv_scores = []
            for tr_i, va_i in skf.split(X_tr, y_tr):
                try:
                    mc = clone(m).fit(X_tr[tr_i], y_tr[tr_i])
                    cv_scores.append(float(f1_score(
                        y_tr[va_i], mc.predict(X_tr[va_i]),
                        average="weighted", zero_division=0)))
                except Exception:
                    cv_scores.append(0.0)
            cv_mean = float(np.mean(cv_scores))
            cv_std  = float(np.std(cv_scores))
            stab    = float(np.clip(1 - cv_std / (cv_mean + 1e-9), 0, 1))

            all_f1.append(f1); all_cal.append(cal); all_stab.append(stab)
            all_preds[mname] = m.predict(X_te)
        except Exception:
            pass

    if not all_f1:
        return None

    # agreement
    names = list(all_preds.keys())
    agr_vals = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            agr_vals.append(float((all_preds[names[i]] == all_preds[names[j]]).mean()))
    agr = float(np.mean(agr_vals)) if agr_vals else 0.8

    dq = float(np.clip(1 - np.isnan(X).mean(), 0, 1))

    # best model by F1
    best = int(np.argmax(all_f1))
    T_pred = float(np.clip(
        TRUST_W["accuracy"]     * all_f1[best]   +
        TRUST_W["calibration"]  * all_cal[best]  +
        TRUST_W["agreement"]    * agr             +
        TRUST_W["data_quality"] * dq              +
        TRUST_W["stability"]    * all_stab[best], 0, 1))

    # actual outcome = best F1 across all models (oracle)
    T_actual = float(np.max(all_f1))

    # per-component trust
    return {
        "name":           name,
        "T_pred":         round(T_pred,   4),
        "T_actual":       round(T_actual, 4),
        "best_f1":        round(all_f1[best],   4),
        "best_cal":       round(all_cal[best],   4),
        "best_stab":      round(all_stab[best],  4),
        "agreement":      round(agr,   4),
        "data_quality":   round(dq,    4),
        "n_models":       len(all_f1),
        "nonconformity":  round(abs(T_pred - T_actual), 4),
    }


# ─────────────────────────────────────────────────────────────
# Trust-ECE
# ─────────────────────────────────────────────────────────────

def _trust_ece(T_preds: np.ndarray, T_actuals: np.ndarray,
               n_bins: int = 10) -> Dict:
    """
    Compute Trust-ECE: treats trust score as a "confidence" and
    actual outcome as ground truth. Same formula as classifier ECE.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    n    = len(T_preds)
    ece  = 0.0
    mce  = 0.0
    bin_data = []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (T_preds >= lo) & (T_preds < hi)
        if mask.sum() == 0:
            continue
        mean_pred   = float(T_preds[mask].mean())
        mean_actual = float(T_actuals[mask].mean())
        gap         = abs(mean_pred - mean_actual)
        weight      = mask.sum() / n
        ece        += gap * weight
        mce         = max(mce, gap)
        bin_data.append({
            "bin_lo":     round(lo, 2),
            "bin_hi":     round(hi, 2),
            "count":      int(mask.sum()),
            "mean_trust": round(mean_pred,   4),
            "mean_actual":round(mean_actual, 4),
            "gap":        round(gap,         4),
            "direction":  "over" if mean_pred > mean_actual else "under",
        })

    return {
        "trust_ece": round(ece, 4),
        "trust_mce": round(mce, 4),
        "n_bins":    n_bins,
        "bins":      bin_data,
        "calibrated": bool(ece < 0.05),
    }


# ─────────────────────────────────────────────────────────────
# Platt recalibration
# ─────────────────────────────────────────────────────────────

def _platt_recalibrate(T_preds: np.ndarray,
                        T_actuals: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """
    Fit sigmoid recalibration: T_cal = σ(a·T_pred + b).
    Returns (a, b, T_recalibrated).
    """
    from scipy.optimize import minimize

    def neg_log_like(params):
        a, b = params
        p = expit(a * T_preds + b)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -np.mean(T_actuals * np.log(p) + (1 - T_actuals) * np.log(1 - p))

    res = minimize(neg_log_like, x0=[1.0, 0.0], method="Nelder-Mead")
    a, b = res.x
    T_cal = expit(a * T_preds + b)
    return float(a), float(b), T_cal


# ─────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────

def run_trust_calibration_study(seed: int = 42) -> Dict:
    from src.data_engine.openml_loader import load_real_datasets

    print("=" * 60)
    print("EMMDS Trust Score Calibration Study  v1.0")
    print("=" * 60)

    raw = load_real_datasets(n=50, verbose=False)

    records = []
    print(f"\nRunning on {len(raw)} real datasets...")
    for X, y, name in raw:
        r = _run_dataset(X.copy(), y.copy(), name, seed)
        if r is not None:
            records.append(r)
            print(f"  {name:<30} T_pred={r['T_pred']:.3f}  "
                  f"T_actual={r['T_actual']:.3f}  err={r['nonconformity']:.3f}")

    T_preds   = np.array([r["T_pred"]   for r in records])
    T_actuals = np.array([r["T_actual"] for r in records])

    # ── Trust-ECE ─────────────────────────────────────────────
    ece_result = _trust_ece(T_preds, T_actuals)

    # ── Platt recalibration ───────────────────────────────────
    a, b, T_recal = _platt_recalibrate(T_preds, T_actuals)
    ece_after     = _trust_ece(T_recal, T_actuals)

    # ── Spearman correlation ──────────────────────────────────
    r_spear, p_spear = stats.spearmanr(T_preds, T_actuals)
    r_pear,  p_pear  = stats.pearsonr(T_preds,  T_actuals)

    # ── Bias analysis ─────────────────────────────────────────
    errors = T_preds - T_actuals
    mean_error = float(errors.mean())    # positive = over-confident
    direction  = "over-confident" if mean_error > 0.02 else (
                 "under-confident" if mean_error < -0.02 else "well-centred")

    print(f"\n── Calibration Results ──")
    print(f"Trust-ECE (before):   {ece_result['trust_ece']:.4f} "
          f"({'calibrated ✅' if ece_result['calibrated'] else 'miscalibrated ⚠️'})")
    print(f"Trust-ECE (after):    {ece_after['trust_ece']:.4f}")
    print(f"Trust-MCE:            {ece_result['trust_mce']:.4f}")
    print(f"Mean bias:            {mean_error:+.4f} ({direction})")
    print(f"Spearman r:           {r_spear:.4f} (p={p_spear:.4f})")
    print(f"Pearson  r:           {r_pear:.4f}  (p={p_pear:.4f})")
    print(f"Platt params:         a={a:.3f}, b={b:.3f}")

    print(f"\nPer-bin reliability:")
    for bd in ece_result["bins"]:
        bar = "█" * int(bd["count"] / max(1, len(records) / 30))
        print(f"  [{bd['bin_lo']:.1f},{bd['bin_hi']:.1f}]  "
              f"n={bd['count']:>3}  trust={bd['mean_trust']:.3f}  "
              f"actual={bd['mean_actual']:.3f}  gap={bd['gap']:.3f}  {bar}")

    return {
        "version":       "1.0_real_datasets",
        "n_datasets":    len(records),
        "trust_ece":     ece_result,
        "trust_ece_after_platt": ece_after,
        "platt_params":  {"a": round(a, 4), "b": round(b, 4)},
        "correlation": {
            "spearman_r": round(float(r_spear), 4),
            "spearman_p": round(float(p_spear), 4),
            "pearson_r":  round(float(r_pear),  4),
            "pearson_p":  round(float(p_pear),  4),
        },
        "bias": {
            "mean_error": round(mean_error, 4),
            "direction":  direction,
            "std_error":  round(float(errors.std()), 4),
        },
        "records": records,
        "finding": (
            f"Trust-ECE={ece_result['trust_ece']:.4f} "
            f"({'calibrated' if ece_result['calibrated'] else 'miscalibrated'}). "
            f"Trust is {direction} by {abs(mean_error):.3f}. "
            f"Spearman r={r_spear:.3f} (p={p_spear:.4f}). "
            f"Post-Platt ECE={ece_after['trust_ece']:.4f}."
        ),
    }


if __name__ == "__main__":
    result = run_trust_calibration_study(seed=42)
    out = OUT / "trust_calibration_study.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {out}")
    print(f"\nKey finding: {result['finding']}")
