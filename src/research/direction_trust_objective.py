"""
Direction 3: Trust as Training Objective — Experiment
======================================================
Compares two training regimes on 10 synthetic classification datasets:

  M_acc:   Standard cross-entropy training (accuracy baseline)
  M_trust: Multi-objective trust training (L_ce + λ_cal·ECE + λ_stab·bootstrap_var)

Hypothesis: M_trust achieves lower deployment risk (overfit + calibration error +
            CV variance) than M_acc at the same accuracy level.

Metrics per dataset:
  - test F1 (macro)
  - 1 - Brier score (calibration score)
  - CV std (stability)
  - EMMDS deployment risk = 0.40·overfit + 0.30·cal_err + 0.30·cv_std
  - EMMDS composite trust score

Results table: dataset × {F1_acc, F1_trust, risk_acc, risk_trust, trust_wins}
"""

import json
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

from src.models.trust_objective_model import TrustObjectiveClassifier


# ─────────────────────────────────────────────────────────────
# Baseline: accuracy-only MLP (lambda_cal=0, lambda_stab=0)
# ─────────────────────────────────────────────────────────────

def _accuracy_model(seed=0):
    return TrustObjectiveClassifier(
        lambda_cal=0.0, lambda_stab=0.0,
        n_hidden=64, lr=3e-3, epochs=50, seed=seed)


def _trust_model(seed=0):
    return TrustObjectiveClassifier(
        lambda_cal=0.5, lambda_stab=1.0,
        n_hidden=64, lr=3e-3, epochs=50, seed=seed)


# ─────────────────────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────────────────────

def _deployment_risk(train_f1: float, test_f1: float,
                     brier_score: float, cv_std: float) -> float:
    overfit  = float(np.clip(train_f1 - test_f1, 0, 1))
    cal_err  = float(np.clip(brier_score, 0, 1))
    return round(0.40 * overfit + 0.30 * cal_err + 0.30 * cv_std, 6)


def _trust_score(f1: float, cal_score: float, cv_std: float,
                 agreement: float = 0.80, dq: float = 0.90) -> float:
    stability = float(np.clip(1.0 - cv_std, 0, 1))
    return round(
        0.05 * f1 +
        0.10 * cal_score +
        0.10 * agreement +
        0.35 * dq +
        0.40 * stability,
        6,
    )


def _evaluate(model, X_train, y_train, X_test, y_test, n_cv=3, seed=0):
    """Returns (train_f1, test_f1, brier, cv_std, cal_score, trust)."""
    model.fit(X_train, y_train)

    # Train metrics
    y_train_pred = model.predict(X_train)
    train_f1 = float(f1_score(y_train, y_train_pred, average="macro", zero_division=0))

    # Test metrics
    y_test_pred  = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)
    test_f1  = float(f1_score(y_test, y_test_pred, average="macro", zero_division=0))

    # Brier score (binary: use positive class; multi-class: macro average)
    classes = model._classes
    if len(classes) == 2:
        brier = float(brier_score_loss(y_test, y_test_proba[:, 1]))
    else:
        brier = float(np.mean([
            brier_score_loss((y_test == c).astype(int), y_test_proba[:, i])
            for i, c in enumerate(classes)
        ]))

    cal_score = float(np.clip(1.0 - brier, 0, 1))

    # CV std (stability)
    skf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=seed)
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    cv_scores = []
    for tr_idx, va_idx in skf.split(X_all, y_all):
        m_cv = (TrustObjectiveClassifier(
                    lambda_cal=model.lambda_cal, lambda_stab=model.lambda_stab,
                    n_hidden=model.n_hidden, lr=model.lr,
                    epochs=model.epochs, seed=seed)
                .fit(X_all[tr_idx], y_all[tr_idx]))
        cv_preds = m_cv.predict(X_all[va_idx])
        cv_scores.append(float(f1_score(y_all[va_idx], cv_preds,
                                        average="macro", zero_division=0)))
    cv_std = float(np.std(cv_scores))

    trust = _trust_score(test_f1, cal_score, cv_std)
    risk  = _deployment_risk(train_f1, test_f1, brier, cv_std)

    return {
        "train_f1":  round(train_f1,  4),
        "test_f1":   round(test_f1,   4),
        "brier":     round(brier,     4),
        "cal_score": round(cal_score, 4),
        "cv_std":    round(cv_std,    4),
        "trust":     trust,
        "risk":      risk,
    }


# ─────────────────────────────────────────────────────────────
# Dataset factory
# ─────────────────────────────────────────────────────────────

def _make_datasets(seed=0):
    rng = np.random.default_rng(seed)

    datasets = []
    configs = [
        {"n_samples": 500,  "n_features": 10, "n_informative": 6,
         "n_classes": 2, "flip_y": 0.05, "name": "easy_binary"},
        {"n_samples": 300,  "n_features": 15, "n_informative": 4,
         "n_classes": 2, "flip_y": 0.15, "name": "noisy_binary"},
        {"n_samples": 400,  "n_features": 12, "n_informative": 5,
         "n_classes": 3, "flip_y": 0.10, "name": "multiclass_3"},
        {"n_samples": 600,  "n_features": 20, "n_informative": 8,
         "n_classes": 2, "flip_y": 0.05, "name": "high_dim_binary"},
        {"n_samples": 200,  "n_features": 8,  "n_informative": 5,
         "n_classes": 2, "flip_y": 0.20, "name": "small_noisy"},
        {"n_samples": 800,  "n_features": 10, "n_informative": 7,
         "n_classes": 2, "flip_y": 0.02, "name": "large_clean"},
        {"n_samples": 350,  "n_features": 10, "n_informative": 3,
         "n_classes": 4, "flip_y": 0.12, "name": "multiclass_4"},
        {"n_samples": 250,  "n_features": 15, "n_informative": 6,
         "n_classes": 2, "flip_y": 0.25, "name": "high_noise"},
        {"n_samples": 500,  "n_features": 8,  "n_informative": 4,
         "n_classes": 2, "flip_y": 0.08, "name": "balanced_medium"},
        {"n_samples": 400,  "n_features": 12, "n_informative": 6,
         "n_classes": 3, "flip_y": 0.18, "name": "noisy_multiclass"},
    ]

    for cfg in configs:
        seed_i = int(rng.integers(0, 9999))
        X, y = make_classification(
            n_samples=cfg["n_samples"],
            n_features=cfg["n_features"],
            n_informative=cfg["n_informative"],
            n_redundant=2,
            n_classes=cfg["n_classes"],
            n_clusters_per_class=1,
            flip_y=cfg["flip_y"],
            random_state=seed_i,
        )
        sc = StandardScaler()
        X = sc.fit_transform(X)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                                    stratify=y, random_state=seed_i)
        datasets.append({"name": cfg["name"], "X_train": X_tr, "y_train": y_tr,
                          "X_test": X_te, "y_test": y_te, "seed": seed_i})
    return datasets


# ─────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────

def run_experiment(n_datasets: int = 10, seed: int = 42) -> dict:
    print("Direction 3: Trust as Training Objective")
    print("=" * 60)

    datasets = _make_datasets(seed)[:n_datasets]
    results = []
    trust_wins = 0

    for ds in datasets:
        name = ds["name"]
        Xtr, ytr = ds["X_train"], ds["y_train"]
        Xte, yte = ds["X_test"],  ds["y_test"]
        s = ds["seed"]

        print(f"\n  Dataset: {name} ({len(Xtr)} train, {len(Xte)} test)")

        m_acc   = _accuracy_model(seed=s)
        m_trust = _trust_model(seed=s)

        r_acc   = _evaluate(m_acc,   Xtr, ytr, Xte, yte, seed=s)
        r_trust = _evaluate(m_trust, Xtr, ytr, Xte, yte, seed=s)

        winner = "trust" if r_trust["risk"] < r_acc["risk"] else "accuracy"
        if winner == "trust":
            trust_wins += 1

        rec = {
            "dataset":       name,
            "acc_f1":        r_acc["test_f1"],
            "trust_f1":      r_trust["test_f1"],
            "acc_risk":      r_acc["risk"],
            "trust_risk":    r_trust["risk"],
            "acc_cal":       r_acc["cal_score"],
            "trust_cal":     r_trust["cal_score"],
            "acc_cv_std":    r_acc["cv_std"],
            "trust_cv_std":  r_trust["cv_std"],
            "acc_trust_score":   r_acc["trust"],
            "trust_trust_score": r_trust["trust"],
            "winner":        winner,
            "risk_reduction": round(r_acc["risk"] - r_trust["risk"], 6),
            "f1_delta":      round(r_trust["test_f1"] - r_acc["test_f1"], 4),
        }
        results.append(rec)

        print(f"    Accuracy-only: F1={r_acc['test_f1']:.3f}  risk={r_acc['risk']:.4f}  "
              f"cal={r_acc['cal_score']:.3f}  cv_std={r_acc['cv_std']:.4f}")
        print(f"    Trust-trained: F1={r_trust['test_f1']:.3f}  risk={r_trust['risk']:.4f}  "
              f"cal={r_trust['cal_score']:.3f}  cv_std={r_trust['cv_std']:.4f}")
        print(f"    Winner: {winner.upper()}  (risk Δ = {rec['risk_reduction']:+.4f})")

    win_rate = trust_wins / n_datasets
    mean_risk_acc   = np.mean([r["acc_risk"]   for r in results])
    mean_risk_trust = np.mean([r["trust_risk"] for r in results])
    mean_f1_acc     = np.mean([r["acc_f1"]     for r in results])
    mean_f1_trust   = np.mean([r["trust_f1"]   for r in results])
    mean_risk_reduction = np.mean([r["risk_reduction"] for r in results])

    # Wilcoxon signed-rank test on risk reduction
    from scipy import stats as _stats
    risk_diffs = [r["acc_risk"] - r["trust_risk"] for r in results]
    if len(risk_diffs) >= 5:
        stat, pval = _stats.wilcoxon(risk_diffs, alternative="greater")
    else:
        stat, pval = float("nan"), float("nan")

    summary = {
        "n_datasets":         n_datasets,
        "trust_win_rate":     round(win_rate, 4),
        "mean_risk_acc":      round(float(mean_risk_acc),   4),
        "mean_risk_trust":    round(float(mean_risk_trust), 4),
        "mean_risk_reduction":round(float(mean_risk_reduction), 4),
        "mean_f1_acc":        round(float(mean_f1_acc),    4),
        "mean_f1_trust":      round(float(mean_f1_trust),  4),
        "f1_delta":           round(float(mean_f1_trust - mean_f1_acc), 4),
        "wilcoxon_stat":      round(float(stat), 4) if not np.isnan(stat) else "nan",
        "wilcoxon_p":         round(float(pval), 4) if not np.isnan(pval) else "nan",
        "hypothesis":         "SUPPORTED" if win_rate > 0.50 else "NOT_SUPPORTED",
        "per_dataset":        results,
    }

    print("\n" + "=" * 60)
    print(f"Trust-trained wins:   {trust_wins}/{n_datasets} ({win_rate:.1%})")
    print(f"Mean risk (acc):      {mean_risk_acc:.4f}")
    print(f"Mean risk (trust):    {mean_risk_trust:.4f}")
    print(f"Mean risk reduction:  {mean_risk_reduction:+.4f}")
    print(f"Mean F1 delta:        {mean_f1_trust - mean_f1_acc:+.4f}")
    print(f"Wilcoxon p (risk):    {pval:.4f}" if not np.isnan(pval) else "Wilcoxon: too few")
    print(f"Hypothesis:           {summary['hypothesis']}")

    return summary


if __name__ == "__main__":
    OUT = ROOT / "outputs" / "research"
    OUT.mkdir(parents=True, exist_ok=True)

    result = run_experiment(n_datasets=10, seed=42)

    out_path = OUT / "direction_trust_objective.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {out_path}")
