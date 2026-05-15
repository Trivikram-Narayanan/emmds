"""
Distribution Shift Evaluation — v4.0  Real Datasets
=====================================================
Tests whether EMMDS trust score predicts model degradation under
distribution shift better than accuracy-only selection.

Uses real OpenML + sklearn datasets only.
"""

import json, warnings
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
# Shift generators
# ─────────────────────────────────────────────────────────────

def shift_feature_noise(X: np.ndarray, severity: float, rng) -> np.ndarray:
    return X + rng.normal(0, severity * X.std(), X.shape)


def shift_missing(X: np.ndarray, severity: float, rng) -> np.ndarray:
    Xc = X.copy()
    Xc[rng.uniform(0, 1, X.shape) < severity] = 0.0
    return Xc


def shift_covariate(X: np.ndarray, severity: float, rng) -> np.ndarray:
    p = X.shape[1]
    Q, _ = np.linalg.qr(rng.normal(0, 1, (p, p)))
    return (1 - severity) * X + severity * (X @ Q)


SHIFT_TYPES  = {"feature_noise": shift_feature_noise,
                "missing":       shift_missing,
                "covariate":     shift_covariate}
SEVERITIES   = [0.1, 0.3, 0.5]


# ─────────────────────────────────────────────────────────────
# Model metrics helpers
# ─────────────────────────────────────────────────────────────

def _f1(model, X, y) -> float:
    try:
        return float(f1_score(y, model.predict(X),
                              average="weighted", zero_division=0))
    except Exception:
        return 0.0


def _calibration(model, X, y) -> float:
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            classes = np.unique(y)
            if len(classes) == 2:
                return float(np.clip(1 - brier_score_loss(y, proba[:, 1]), 0, 1))
            return float(np.clip(1 - np.mean([
                brier_score_loss((y == c).astype(int), proba[:, i])
                for i, c in enumerate(classes)]), 0, 1))
    except Exception:
        pass
    return 0.5


def _stability(model, X_tr, y_tr, seed: int) -> float:
    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = []
    for tr, va in skf.split(X_tr, y_tr):
        try:
            m = clone(model).fit(X_tr[tr], y_tr[tr])
            scores.append(_f1(m, X_tr[va], y_tr[va]))
        except Exception:
            scores.append(0.0)
    cv_mean = np.mean(scores)
    cv_std  = np.std(scores)
    return float(np.clip(1 - cv_std / (cv_mean + 1e-9), 0, 1))


def _trust(test_f1, cal, stab, agr=0.80, dq=0.90) -> float:
    return float(np.clip(
        TRUST_W["accuracy"]    * test_f1 +
        TRUST_W["calibration"] * cal     +
        TRUST_W["agreement"]   * agr     +
        TRUST_W["data_quality"]* dq      +
        TRUST_W["stability"]   * stab, 0, 1))


# ─────────────────────────────────────────────────────────────
# Single dataset shift experiment
# ─────────────────────────────────────────────────────────────

def _run_dataset(X_tr, y_tr, X_te, y_te, seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    model_suite = {
        "logistic": LogisticRegression(max_iter=500, random_state=seed),
        "lda":      LinearDiscriminantAnalysis(),
        "rf":       RandomForestClassifier(n_estimators=50, random_state=seed),
        "gbm":      GradientBoostingClassifier(n_estimators=50, random_state=seed),
        "tree":     DecisionTreeClassifier(max_depth=6, random_state=seed),
    }

    trained = {}
    for name, m in model_suite.items():
        try:
            trained[name] = clone(m).fit(X_tr, y_tr)
        except Exception:
            pass

    if not trained:
        return {}

    orig: Dict[str, Dict] = {}
    for name, m in trained.items():
        f1   = _f1(m, X_te, y_te)
        cal  = _calibration(m, X_te, y_te)
        stab = _stability(m, X_tr, y_tr, seed)
        orig[name] = {"f1": f1, "cal": cal, "stab": stab,
                      "trust": _trust(f1, cal, stab), "accuracy": f1}

    shift_results = {}
    for shift_name, shift_fn in SHIFT_TYPES.items():
        for sev in SEVERITIES:
            key      = f"{shift_name}_sev{sev}"
            X_shifted = shift_fn(X_te.copy(), sev, rng)

            degs: Dict[str, float] = {}
            for name, m in trained.items():
                shifted_f1   = _f1(m, X_shifted, y_te)
                degs[name]   = orig[name]["f1"] - shifted_f1

            trust_scores = [orig[n]["trust"]    for n in trained]
            acc_scores   = [orig[n]["accuracy"] for n in trained]
            deg_list     = [degs[n]             for n in trained]

            r_trust, _ = stats.spearmanr(trust_scores, deg_list)
            r_acc,   _ = stats.spearmanr(acc_scores,   deg_list)

            trust_sel = max(trained, key=lambda n: orig[n]["trust"])
            acc_sel   = max(trained, key=lambda n: orig[n]["accuracy"])
            oracle    = min(trained, key=lambda n: degs[n])

            shift_results[key] = {
                "shift_type":         shift_name,
                "severity":           sev,
                "spearman_trust":     round(float(r_trust), 4),
                "spearman_acc":       round(float(r_acc),   4),
                "trust_predicts_better": bool(abs(r_trust) > abs(r_acc)),
                "trust_selected_deg": round(float(degs[trust_sel]), 4),
                "acc_selected_deg":   round(float(degs[acc_sel]),   4),
                "oracle_deg":         round(float(degs[oracle]),     4),
                "trust_wins":         bool(degs[trust_sel] < degs[acc_sel]),
                "mean_degradation":   round(float(np.mean(list(degs.values()))), 4),
            }

    return {"original": orig, "shifts": shift_results}


# ─────────────────────────────────────────────────────────────
# Full battery
# ─────────────────────────────────────────────────────────────

def run_shift_evaluation(seed: int = 42) -> Dict:
    from src.data_engine.openml_loader import load_real_datasets
    print("=" * 60)
    print("EMMDS Distribution Shift Evaluation v4.0 — Real Datasets")
    print("=" * 60)

    raw = load_real_datasets(n=50, verbose=False)
    sc  = StandardScaler()
    all_shift_results = []

    for X, y, name in raw:
        # impute NaNs
        col_med = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            X[np.isnan(X[:, j]), j] = col_med[j]
        X = sc.fit_transform(X)

        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.25, stratify=y, random_state=seed)
        except Exception:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.25, random_state=seed)

        res = _run_dataset(X_tr, y_tr, X_te, y_te, seed)
        if res and res.get("shifts"):
            all_shift_results.append({"name": name, **res})
            tw = sum(v["trust_wins"] for v in res["shifts"].values())
            print(f"  {name:<30} {tw}/{len(res['shifts'])} shifts: trust wins")

    # ── Aggregate ─────────────────────────────────────────────
    trust_wins_total, acc_wins_total = 0, 0
    trust_r_vals, acc_r_vals         = [], []
    total_shifts                     = 0

    for ds_res in all_shift_results:
        for sv in ds_res["shifts"].values():
            total_shifts += 1
            if sv["trust_wins"]:
                trust_wins_total += 1
            else:
                acc_wins_total   += 1
            trust_r_vals.append(sv["spearman_trust"])
            acc_r_vals.append(sv["spearman_acc"])

    trust_win_rate = trust_wins_total / total_shifts if total_shifts > 0 else 0
    mean_r_trust   = float(np.nanmean(trust_r_vals))
    mean_r_acc     = float(np.nanmean(acc_r_vals))

    by_type: Dict[str, Dict] = {}
    for ds_res in all_shift_results:
        for sv in ds_res["shifts"].values():
            stype = sv["shift_type"]
            if stype not in by_type:
                by_type[stype] = {"wins": 0, "total": 0, "r_trust": [], "r_acc": []}
            by_type[stype]["total"] += 1
            if sv["trust_wins"]:
                by_type[stype]["wins"] += 1
            by_type[stype]["r_trust"].append(sv["spearman_trust"])
            by_type[stype]["r_acc"].append(sv["spearman_acc"])

    type_summary = {
        t: {"win_rate":     round(d["wins"] / d["total"], 3),
            "mean_r_trust": round(float(np.nanmean(d["r_trust"])), 3),
            "mean_r_acc":   round(float(np.nanmean(d["r_acc"])),   3)}
        for t, d in by_type.items()
    }

    print(f"\nOverall trust win rate: {trust_win_rate:.1%} "
          f"({trust_wins_total}/{total_shifts})")
    print(f"Mean Spearman(trust, degradation): {mean_r_trust:.3f}")
    print(f"Mean Spearman(acc,   degradation): {mean_r_acc:.3f}")
    print("\nPer shift type:")
    for t, d in type_summary.items():
        print(f"  {t:<18} win={d['win_rate']:.1%}  "
              f"r_trust={d['mean_r_trust']:.3f}  r_acc={d['mean_r_acc']:.3f}")

    return {
        "version":             "4.0_real_datasets",
        "n_datasets":          len(all_shift_results),
        "n_shift_configs":     len(SEVERITIES) * len(SHIFT_TYPES),
        "total_comparisons":   total_shifts,
        "trust_win_rate":      round(trust_win_rate, 4),
        "acc_win_rate":        round(acc_wins_total / total_shifts, 4) if total_shifts else 0,
        "mean_spearman_trust": round(mean_r_trust, 4),
        "mean_spearman_acc":   round(mean_r_acc,   4),
        "trust_better_predictor": bool(abs(mean_r_trust) > abs(mean_r_acc)),
        "per_shift_type":      type_summary,
        "dataset_names":       [r["name"] for r in all_shift_results],
        "hypothesis":          "SUPPORTED" if trust_win_rate > 0.50 else "NOT_SUPPORTED",
    }


if __name__ == "__main__":
    result = run_shift_evaluation(seed=42)
    out = OUT / "shift_evaluation.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {out}")
