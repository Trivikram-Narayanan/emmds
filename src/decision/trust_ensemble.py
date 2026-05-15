"""
Trust-Weighted Ensemble  v1.0
==============================
Instead of selecting one model, weights ensemble predictions by trust score.

    ŷ = Σ T(Mᵢ) · p̂ᵢ(x)  /  Σ T(Mᵢ)

Compared against four baselines:
  1. Accuracy-weighted   : weight ∝ test F1
  2. Uniform             : equal weights (classic ensemble)
  3. Single best         : trust-based model selection (existing EMMDS)
  4. Oracle single       : best possible single-model selection

Key hypothesis: Trust-weighted ensemble is more robust under distribution shift
because stability-heavy weights down-weight high-variance models.

Reference: Guo et al. (2017). On calibration of modern neural networks.
"""

import json
import numpy as np
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, brier_score_loss, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT  = ROOT / "outputs" / "research"
OUT.mkdir(parents=True, exist_ok=True)

TRUST_W = dict(accuracy=0.05, calibration=0.10, agreement=0.10,
               data_quality=0.35, stability=0.40)


# ─────────────────────────────────────────────────────────────
# Per-model trust + calibrated probabilities
# ─────────────────────────────────────────────────────────────

def _fit_models(X_tr, y_tr, X_te, y_te, seed: int) -> Tuple[Dict, Dict]:
    models = {
        "lr":   LogisticRegression(max_iter=300, random_state=seed),
        "rf":   RandomForestClassifier(n_estimators=50, random_state=seed),
        "gbm":  GradientBoostingClassifier(n_estimators=50, random_state=seed),
        "tree": DecisionTreeClassifier(max_depth=5, random_state=seed),
        "knn":  KNeighborsClassifier(n_neighbors=5),
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fitted, metrics = {}, {}
    classes = np.unique(np.concatenate([y_tr, y_te]))
    n_cls   = len(classes)

    for name, m in models.items():
        try:
            m.fit(X_tr, y_tr)
            f1 = float(f1_score(y_te, m.predict(X_te), average="weighted", zero_division=0))
            cal = 0.5
            if hasattr(m, "predict_proba"):
                proba = m.predict_proba(X_te)
                if n_cls == 2:
                    cal = float(np.clip(1 - brier_score_loss(y_te, proba[:, 1]), 0, 1))
                else:
                    cal = float(np.clip(1 - np.mean([
                        brier_score_loss((y_te == c).astype(int), proba[:, i])
                        for i, c in enumerate(classes)]), 0, 1))

            cv_scores = []
            for tr_i, va_i in skf.split(X_tr, y_tr):
                try:
                    mc = clone(m).fit(X_tr[tr_i], y_tr[tr_i])
                    cv_scores.append(float(f1_score(y_tr[va_i], mc.predict(X_tr[va_i]),
                                                    average="weighted", zero_division=0)))
                except Exception:
                    cv_scores.append(0.0)
            cv_mean = float(np.mean(cv_scores))
            cv_std  = float(np.std(cv_scores))
            stab    = float(np.clip(1 - cv_std / (cv_mean + 1e-9), 0, 1))

            dq = float(np.clip(1 - np.isnan(X_tr).mean(), 0, 1))
            trust = float(np.clip(
                TRUST_W["accuracy"]    * f1   +
                TRUST_W["calibration"] * cal  +
                TRUST_W["agreement"]   * 0.80 +
                TRUST_W["data_quality"]* dq   +
                TRUST_W["stability"]   * stab, 0, 1))

            fitted[name]  = m
            metrics[name] = {"f1": f1, "cal": cal, "stab": stab, "trust": trust}
        except Exception:
            pass

    return fitted, metrics


def _ensemble_predict(fitted: Dict, metrics: Dict,
                       X: np.ndarray, classes: np.ndarray,
                       weighting: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (predicted_labels, weight_vector).
    weighting: 'trust' | 'accuracy' | 'uniform'
    """
    n_cls = len(classes)
    names = list(fitted.keys())
    proba_stack = []

    for name in names:
        m = fitted[name]
        if hasattr(m, "predict_proba"):
            try:
                p = m.predict_proba(X)
                # Align to full class set
                if p.shape[1] != n_cls:
                    p_full = np.zeros((len(X), n_cls))
                    for ci, c in enumerate(m.classes_):
                        idx = np.where(classes == c)[0]
                        if len(idx) > 0:
                            p_full[:, idx[0]] = p[:, ci]
                    p = p_full
                proba_stack.append(p)
            except Exception:
                proba_stack.append(None)
        else:
            proba_stack.append(None)

    # Build weights
    if weighting == "trust":
        w = np.array([metrics[n]["trust"] for n in names])
    elif weighting == "accuracy":
        w = np.array([metrics[n]["f1"]    for n in names])
    else:  # uniform
        w = np.ones(len(names))

    w = np.clip(w, 1e-9, None)
    w /= w.sum()

    # Weighted probability
    agg = np.zeros((len(X), n_cls))
    for i, p in enumerate(proba_stack):
        if p is not None:
            agg += w[i] * p
        else:
            # fall back to hard vote
            preds = fitted[names[i]].predict(X)
            hard  = np.zeros((len(X), n_cls))
            for ci, c in enumerate(classes):
                hard[preds == c, ci] = 1.0
            agg += w[i] * hard

    y_pred = classes[agg.argmax(axis=1)]
    return y_pred, w


# ─────────────────────────────────────────────────────────────
# Shift generators (same as shift_evaluation.py)
# ─────────────────────────────────────────────────────────────

def _apply_shift(X: np.ndarray, shift_type: str, severity: float,
                 rng: np.random.Generator) -> np.ndarray:
    if shift_type == "noise":
        return X + rng.normal(0, severity * X.std(), X.shape)
    if shift_type == "missing":
        Xc = X.copy()
        Xc[rng.uniform(0, 1, X.shape) < severity] = 0.0
        return Xc
    if shift_type == "covariate":
        p = X.shape[1]
        Q, _ = np.linalg.qr(rng.normal(0, 1, (p, p)))
        return (1 - severity) * X + severity * (X @ Q)
    return X


# ─────────────────────────────────────────────────────────────
# Single dataset experiment
# ─────────────────────────────────────────────────────────────

def _run_dataset(X: np.ndarray, y: np.ndarray,
                 name: str, seed: int = 42) -> Dict:
    col_med = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        X[np.isnan(X[:, j]), j] = col_med[j]
    sc = StandardScaler()
    X  = sc.fit_transform(X)

    le = LabelEncoder()
    y  = le.fit_transform(y)
    classes = np.unique(y)

    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=seed)
    except Exception:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=seed)

    fitted, metrics = _fit_models(X_tr, y_tr, X_te, y_te, seed)
    if not fitted:
        return None

    rng = np.random.default_rng(seed)
    strategies = ["trust", "accuracy", "uniform"]
    shift_configs = [("none", 0.0), ("noise", 0.2), ("noise", 0.5),
                     ("missing", 0.2), ("covariate", 0.3)]

    results = {s: {"no_shift": None, "shift_avg": []} for s in strategies}
    # oracle: best single model by F1
    oracle_name = max(metrics, key=lambda n: metrics[n]["f1"])

    for shift_type, severity in shift_configs:
        if shift_type == "none":
            X_eval = X_te
        else:
            X_eval = _apply_shift(X_te.copy(), shift_type, severity, rng)

        for strat in strategies:
            y_pred, _ = _ensemble_predict(fitted, metrics, X_eval, classes, strat)
            f1 = float(f1_score(y_te, y_pred, average="weighted", zero_division=0))
            if shift_type == "none":
                results[strat]["no_shift"] = f1
            else:
                results[strat]["shift_avg"].append(f1)

    # Oracle single-model on no shift + avg shift
    oracle_f1_clean = metrics[oracle_name]["f1"]
    oracle_f1_shift = []
    for shift_type, severity in shift_configs:
        if shift_type == "none":
            continue
        X_eval = _apply_shift(X_te.copy(), shift_type, severity, rng)
        oracle_f1_shift.append(_f1_single(fitted[oracle_name], X_eval, y_te))

    out = {}
    for strat in strategies:
        out[strat] = {
            "clean_f1": round(results[strat]["no_shift"] or 0, 4),
            "shift_f1": round(float(np.mean(results[strat]["shift_avg"])), 4),
        }
    out["oracle"] = {
        "clean_f1": round(oracle_f1_clean, 4),
        "shift_f1": round(float(np.mean(oracle_f1_shift)) if oracle_f1_shift else 0, 4),
    }
    return out


def _f1_single(model, X, y) -> float:
    try:
        return float(f1_score(y, model.predict(X), average="weighted", zero_division=0))
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────
# Full experiment
# ─────────────────────────────────────────────────────────────

def run_ensemble_experiment(seed: int = 42) -> Dict:
    from src.data_engine.openml_loader import load_real_datasets

    print("=" * 60)
    print("Trust-Weighted Ensemble  v1.0")
    print("=" * 60)

    raw = load_real_datasets(n=50, verbose=False)
    strategies = ["trust", "accuracy", "uniform", "oracle"]

    all_clean = {s: [] for s in strategies}
    all_shift = {s: [] for s in strategies}

    print(f"\nRunning on {len(raw)} datasets...")
    for X, y, name in raw:
        r = _run_dataset(X.copy(), y.copy(), name, seed)
        if r is None:
            continue
        for s in strategies:
            if s in r:
                all_clean[s].append(r[s]["clean_f1"])
                all_shift[s].append(r[s]["shift_f1"])
        delta_clean = (r.get("trust", {}).get("clean_f1", 0) -
                       r.get("accuracy", {}).get("clean_f1", 0))
        delta_shift = (r.get("trust", {}).get("shift_f1", 0) -
                       r.get("accuracy", {}).get("shift_f1", 0))
        print(f"  {name:<30} trust_clean={r.get('trust',{}).get('clean_f1',0):.3f}  "
              f"acc_clean={r.get('accuracy',{}).get('clean_f1',0):.3f}  "
              f"Δclean={delta_clean:+.3f}  Δshift={delta_shift:+.3f}")

    print(f"\n── Aggregate Results ──")
    print(f"{'Strategy':<18}  {'Clean F1':>10}  {'Shift F1':>10}  {'Δ vs Accuracy':>14}")
    print("-" * 60)

    summary = {}
    for s in strategies:
        if not all_clean[s]:
            continue
        c = float(np.mean(all_clean[s]))
        sh = float(np.mean(all_shift[s]))
        summary[s] = {"mean_clean_f1": round(c, 4), "mean_shift_f1": round(sh, 4),
                      "n_datasets": len(all_clean[s])}

    trust_c = summary.get("trust", {}).get("mean_clean_f1", 0)
    trust_s = summary.get("trust", {}).get("mean_shift_f1", 0)

    for s in strategies:
        if s not in summary:
            continue
        d = summary[s]
        dc = round(d["mean_clean_f1"] - summary.get("accuracy", {}).get("mean_clean_f1", 0), 4)
        ds = round(d["mean_shift_f1"] - summary.get("accuracy", {}).get("mean_shift_f1", 0), 4)
        print(f"  {s:<16}  {d['mean_clean_f1']:>10.4f}  {d['mean_shift_f1']:>10.4f}  "
              f"Δclean={dc:+.4f} Δshift={ds:+.4f}")

    acc_c = summary.get("accuracy", {}).get("mean_clean_f1", 0)
    acc_s = summary.get("accuracy", {}).get("mean_shift_f1", 0)

    return {
        "version":   "1.0_real_datasets",
        "n_datasets": summary.get("trust", {}).get("n_datasets", 0),
        "summary":    summary,
        "trust_vs_accuracy": {
            "clean_f1_delta": round(trust_c - acc_c, 4),
            "shift_f1_delta": round(trust_s - acc_s, 4),
            "trust_better_clean": bool(trust_c > acc_c),
            "trust_better_shift": bool(trust_s > acc_s),
        },
        "finding": (
            f"Trust-weighted ensemble vs accuracy-weighted: "
            f"Δclean={trust_c-acc_c:+.4f}, Δshift={trust_s-acc_s:+.4f}. "
            f"Trust ensemble {'better' if trust_s > acc_s else 'not better'} under shift."
        ),
    }


if __name__ == "__main__":
    result = run_ensemble_experiment(seed=42)
    out = OUT / "trust_ensemble.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {out}")
    print(f"\nKey finding: {result['finding']}")
