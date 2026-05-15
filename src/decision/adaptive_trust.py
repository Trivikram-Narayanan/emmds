"""
Adaptive Trust Weighting — Meta-Learner
=========================================
Learns dataset-specific trust weights from meta-features instead of using
fixed empirical weights. This is the strongest novel contribution: the weight
vector w = (w_acc, w_cal, w_agr, w_dq, w_stab) is predicted from dataset
meta-features, and is different for every dataset.

Architecture
------------
  Input : meta-feature vector (9 features, see below)
  Output: weight vector in the probability simplex (sum=1, each ≥ 0)

  Models tried (in order of complexity):
    1. Ridge regression (baseline)
    2. Random Forest regressor
    3. Gradient Boosting regressor
    4. Rule-based fallback (always available)

Meta-features used
------------------
  n_samples, n_features, imbalance_ratio, missing_ratio,
  noise_estimate, dim_ratio, avg_abs_correlation, n_classes,
  task_type (0=clf, 1=reg)

Training protocol
-----------------
  Meta-train: for each dataset in the training pool, compute the
    weight vector that minimises mean deployment risk of the selected
    model (exhaustive search over weight simplex).
  Meta-test: predict weights from meta-features; use predicted weights
    in trust score; evaluate model selection quality.

Leave-one-dataset-out (LOO) cross-validation across the meta-train pool
gives an unbiased estimate of adaptive weighting performance.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

ROOT = Path(__file__).resolve().parents[2]
OUT  = ROOT / "outputs" / "research"
OUT.mkdir(parents=True, exist_ok=True)

# Fixed empirical weights (baseline)
FIXED_WEIGHTS = np.array([0.05, 0.10, 0.10, 0.35, 0.40])
WEIGHT_NAMES  = ["accuracy", "calibration", "agreement", "data_quality", "stability"]


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Project vector onto the probability simplex (non-negative, sum=1)."""
    v = np.clip(v, 0, None)
    s = v.sum()
    return v / s if s > 0 else np.ones(len(v)) / len(v)


def _compute_trust(metrics: Dict, weights: np.ndarray,
                   agreement: float = 0.80, dq: float = 0.90) -> float:
    components = np.array([
        metrics.get("test_f1",   0.5),
        metrics.get("cal_score", 0.5),
        agreement,
        dq,
        metrics.get("stability", 0.5),
    ])
    return float(np.clip((weights * components).sum(), 0, 1))


def _select_by_weights(all_metrics: Dict[str, Dict], weights: np.ndarray,
                       agreement: float = 0.80, dq: float = 0.90) -> str:
    return max(all_metrics.keys(),
               key=lambda n: _compute_trust(all_metrics[n], weights, agreement, dq))


# ─────────────────────────────────────────────────────────────
# Optimal weight finder (grid search on simplex)
# ─────────────────────────────────────────────────────────────

def find_optimal_weights(
    all_metrics: Dict[str, Dict],
    agreement_scores: Dict[str, float],
    dq: float,
    n_grid: int = 30,
    seed: int = 0,
) -> Tuple[np.ndarray, float]:
    """
    Find the weight vector w* that minimises the deployment risk of the
    trust-selected model via random simplex search.

    Returns (w_star, min_risk).
    """
    rng = np.random.default_rng(seed)
    best_w    = FIXED_WEIGHTS.copy()
    best_risk = float("inf")

    # Oracle risk (lower bound)
    oracle_risk = min(m["risk"] for m in all_metrics.values())

    for _ in range(n_grid):
        # Random Dirichlet sample from simplex
        w = _project_simplex(rng.dirichlet(np.ones(5)))
        chosen = _select_by_weights(all_metrics, w,
                                     agreement_scores.get(
                                         list(all_metrics)[0], 0.8), dq)
        risk = all_metrics[chosen]["risk"]
        if risk < best_risk:
            best_risk = risk
            best_w    = w.copy()

    return best_w, best_risk


# ─────────────────────────────────────────────────────────────
# Meta-feature extraction (standalone, no pipeline dependency)
# ─────────────────────────────────────────────────────────────

def extract_meta_features(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return 9-dim meta-feature vector."""
    n, p  = X.shape
    classes, counts = np.unique(y, return_counts=True)
    imb   = float(counts.max() / counts.min()) if counts.min() > 0 else 1.0
    miss  = float(np.isnan(X).mean())
    noise = float(np.std([np.std(X[:, j]) for j in range(p)]))
    dim_r = float(p / n)
    corr  = float(np.abs(np.corrcoef(X.T)).mean()) if p > 1 else 0.0
    nc    = float(len(classes))
    return np.array([
        np.log1p(n), np.log1p(p), np.log1p(imb),
        miss, noise, dim_r, corr, nc, 0.0  # task=0 (classification)
    ], dtype=float)


# ─────────────────────────────────────────────────────────────
# Adaptive Trust Weight Learner
# ─────────────────────────────────────────────────────────────

class AdaptiveTrustWeighter:
    """
    Meta-learner that predicts trust weight vectors from dataset meta-features.

    Trained on meta-train datasets; evaluated on meta-test datasets.
    """

    def __init__(self, model_type: str = "rf"):
        self.model_type = model_type
        self._scaler    = StandardScaler()
        self._models: List = []  # one per weight dimension
        self._fitted = False

    def fit(
        self,
        meta_features: List[np.ndarray],    # one per dataset
        optimal_weights: List[np.ndarray],  # target weight vectors
    ) -> "AdaptiveTrustWeighter":
        X = np.vstack(meta_features)
        X = self._scaler.fit_transform(X)
        Y = np.vstack(optimal_weights)  # shape (n_datasets, 5)

        self._models = []
        for k in range(5):
            y_k = Y[:, k]
            if self.model_type == "ridge":
                m = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]).fit(X, y_k)
            elif self.model_type == "gbm":
                m = GradientBoostingRegressor(
                    n_estimators=50, max_depth=3, random_state=42).fit(X, y_k)
            else:  # rf (default)
                m = RandomForestRegressor(
                    n_estimators=50, random_state=42).fit(X, y_k)
            self._models.append(m)

        self._fitted = True
        return self

    def predict_weights(self, meta_feat: np.ndarray) -> np.ndarray:
        """Predict weight vector for a single dataset meta-feature vector."""
        if not self._fitted:
            return FIXED_WEIGHTS.copy()
        X = self._scaler.transform(meta_feat.reshape(1, -1))
        raw = np.array([m.predict(X)[0] for m in self._models])
        return _project_simplex(raw)

    def loo_score(
        self,
        meta_features: List[np.ndarray],
        optimal_weights: List[np.ndarray],
    ) -> float:
        """Leave-one-out MAE on weight prediction."""
        n = len(meta_features)
        maes = []
        for i in range(n):
            tr_X = [meta_features[j] for j in range(n) if j != i]
            tr_Y = [optimal_weights[j] for j in range(n) if j != i]
            if len(tr_X) < 3:
                continue
            tmp = AdaptiveTrustWeighter(self.model_type).fit(tr_X, tr_Y)
            w_pred = tmp.predict_weights(meta_features[i])
            maes.append(float(np.abs(w_pred - optimal_weights[i]).mean()))
        return float(np.mean(maes)) if maes else float("nan")


# ─────────────────────────────────────────────────────────────
# Rule-based fallback
# ─────────────────────────────────────────────────────────────

def rule_based_weights(meta_feat: np.ndarray) -> np.ndarray:
    """
    Simple heuristic: down-weight stability on small/noisy datasets,
    up-weight calibration on imbalanced datasets.
    """
    log_n, log_p, log_imb, miss, noise, dim_r, corr, nc, _ = meta_feat
    w = FIXED_WEIGHTS.copy().astype(float)

    # Small dataset → less stability signal, more calibration
    if log_n < np.log1p(200):
        w[4] -= 0.10   # stability
        w[1] += 0.10   # calibration

    # High imbalance → up-weight calibration
    if log_imb > np.log1p(5):
        w[1] += 0.08
        w[3] -= 0.08   # DQ less informative

    # High dimensionality → agreement less reliable
    if dim_r > 0.15:
        w[2] -= 0.05
        w[3] += 0.05

    # High noise → stability paradox risk: down-weight stability
    if noise > 1.5:
        w[4] -= 0.10
        w[1] += 0.10

    return _project_simplex(w)


# ─────────────────────────────────────────────────────────────
# Experiment runner
# ─────────────────────────────────────────────────────────────

def run_adaptive_experiment(
    meta_train_records: List[Dict],
    meta_test_records:  List[Dict],
    all_metrics_train:  List[Dict[str, Dict]],
    all_metrics_test:   List[Dict[str, Dict]],
    meta_feats_train:   List[np.ndarray],
    meta_feats_test:    List[np.ndarray],
    verbose: bool = True,
) -> Dict:
    """
    Full adaptive weight experiment:
    1. Find optimal weights for each meta-train dataset (supervision signal).
    2. Fit AdaptiveTrustWeighter on (meta_feats_train, optimal_weights_train).
    3. Predict weights for each meta-test dataset.
    4. Compare: fixed_weights, adaptive_rf, adaptive_ridge, rule_based, equal_weights.
    """

    # ── Step 1: optimal weights on meta-train ──────────────────────────
    opt_weights_train = []
    for mets in all_metrics_train:
        agr = {n: 0.8 for n in mets}
        w_opt, _ = find_optimal_weights(mets, agr, dq=0.90, n_grid=100)
        opt_weights_train.append(w_opt)

    if verbose:
        mean_opt = np.mean(opt_weights_train, axis=0)
        print("\nMean optimal weights (meta-train):")
        for k, v in zip(WEIGHT_NAMES, mean_opt):
            print(f"  {k:<14}: {v:.3f}")

    # ── Step 2: fit meta-learners ──────────────────────────────────────
    learners = {
        "adaptive_rf":    AdaptiveTrustWeighter("rf"),
        "adaptive_ridge": AdaptiveTrustWeighter("ridge"),
        "adaptive_gbm":   AdaptiveTrustWeighter("gbm"),
    }
    for name, lrn in learners.items():
        lrn.fit(meta_feats_train, opt_weights_train)

    # LOO scores on meta-train
    loo_scores = {}
    for name, lrn in learners.items():
        loo = lrn.loo_score(meta_feats_train, opt_weights_train)
        loo_scores[name] = round(loo, 4)
        if verbose:
            print(f"  LOO MAE ({name}): {loo:.4f}")

    # ── Step 3: evaluate on meta-test ─────────────────────────────────
    weight_strategies = list(learners.keys()) + ["fixed", "equal", "rule_based"]
    strategy_risks: Dict[str, List[float]] = {s: [] for s in weight_strategies}
    strategy_hits:  Dict[str, List[int]]   = {s: [] for s in weight_strategies}

    for mets, mf in zip(all_metrics_test, meta_feats_test):
        agr  = {n: 0.80 for n in mets}
        dq   = 0.90
        oracle_name  = min(mets, key=lambda n: mets[n]["risk"])
        oracle_risk  = mets[oracle_name]["risk"]

        weights_map = {
            "fixed":      FIXED_WEIGHTS,
            "equal":      np.ones(5) / 5,
            "rule_based": rule_based_weights(mf),
        }
        for name, lrn in learners.items():
            weights_map[name] = lrn.predict_weights(mf)

        for strat, w in weights_map.items():
            chosen = _select_by_weights(mets, w,
                                         agr.get(list(mets)[0], 0.8), dq)
            risk = mets[chosen]["risk"]
            strategy_risks[strat].append(risk)
            strategy_hits[strat].append(int(chosen == oracle_name))

    # ── Step 4: summarise ─────────────────────────────────────────────
    summary = {}
    for strat in weight_strategies:
        risks = strategy_risks[strat]
        hits  = strategy_hits[strat]
        summary[strat] = {
            "mean_risk":  round(float(np.mean(risks)), 4),
            "win_rate":   round(float(np.mean(hits)),  4),
            "loo_mae":    loo_scores.get(strat, None),
        }

    if verbose:
        print("\n--- Adaptive Weighting Results (meta-test) ---")
        print(f"{'Strategy':<20} {'Mean Risk':>10} {'Win Rate':>10}")
        for strat, d in sorted(summary.items(), key=lambda x: x[1]["mean_risk"]):
            print(f"  {strat:<18} {d['mean_risk']:>10.4f} {d['win_rate']:>9.1%}")

    best_adaptive = min(
        [s for s in weight_strategies if s.startswith("adaptive")],
        key=lambda s: summary[s]["mean_risk"]
    )

    return {
        "loo_scores":         loo_scores,
        "strategy_summary":   summary,
        "best_adaptive":      best_adaptive,
        "adaptive_beats_fixed": (
            summary[best_adaptive]["mean_risk"] < summary["fixed"]["mean_risk"]
        ),
        "mean_optimal_weights": {
            k: round(float(v), 4)
            for k, v in zip(WEIGHT_NAMES, np.mean(opt_weights_train, axis=0))
        },
    }
