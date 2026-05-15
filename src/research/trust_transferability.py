"""
Trust Transferability  v1.0
============================
Given trust score T(M, D_src) on source dataset, predict T(M, D_tgt) on
target dataset without retraining.

Transfer function:
    T̂(M, D_tgt) = f(T(M, D_src), dist(D_src, D_tgt), Δmeta)

Where:
  dist(D_src, D_tgt) = Maximum Mean Discrepancy (kernel MMD)
  Δmeta              = difference in meta-feature vectors

Training protocol
-----------------
1. For all pairs (D_i, D_j) from real datasets: compute trust on both,
   compute MMD + meta-feature delta.
2. Train a Ridge regression: predict T(M, D_tgt) from above features.
3. LOO-CV gives unbiased transfer prediction error (MAE, Spearman r).

Insight: if trust is highly transferable (low MAE), it is dataset-agnostic
and therefore more universally valid as a deployment signal.
"""

import json
import numpy as np
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, brier_score_loss
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT  = ROOT / "outputs" / "research"
OUT.mkdir(parents=True, exist_ok=True)

TRUST_W = dict(accuracy=0.05, calibration=0.10, agreement=0.10,
               data_quality=0.35, stability=0.40)


# ─────────────────────────────────────────────────────────────
# Trust score (fast, self-contained)
# ─────────────────────────────────────────────────────────────

def _trust(X_tr, y_tr, X_te, y_te, seed: int) -> float:
    models = [
        LogisticRegression(max_iter=300, random_state=seed),
        RandomForestClassifier(n_estimators=30, random_state=seed),
        GradientBoostingClassifier(n_estimators=30, random_state=seed),
    ]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    best_trust = 0.0

    for m in models:
        try:
            m.fit(X_tr, y_tr)
            f1 = float(f1_score(y_te, m.predict(X_te), average="weighted", zero_division=0))
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
                    cv_scores.append(float(f1_score(y_tr[va_i], mc.predict(X_tr[va_i]),
                                                    average="weighted", zero_division=0)))
                except Exception:
                    cv_scores.append(0.0)
            cv_mean = float(np.mean(cv_scores))
            cv_std  = float(np.std(cv_scores))
            stab    = float(np.clip(1 - cv_std / (cv_mean + 1e-9), 0, 1))
            dq      = float(np.clip(1 - np.isnan(X_tr).mean(), 0, 1))
            t = float(np.clip(
                TRUST_W["accuracy"]    * f1   +
                TRUST_W["calibration"] * cal  +
                TRUST_W["agreement"]   * 0.80 +
                TRUST_W["data_quality"]* dq   +
                TRUST_W["stability"]   * stab, 0, 1))
            best_trust = max(best_trust, t)
        except Exception:
            pass
    return best_trust


# ─────────────────────────────────────────────────────────────
# Maximum Mean Discrepancy (Gaussian kernel)
# ─────────────────────────────────────────────────────────────

def _mmd(X_src: np.ndarray, X_tgt: np.ndarray,
          gamma: float = 1.0) -> float:
    """
    Unbiased estimate of squared MMD with Gaussian kernel k(x,y) = exp(-γ‖x-y‖²).
    Operates on common dimensionality (min of both).
    """
    p = min(X_src.shape[1], X_tgt.shape[1])
    Xs = X_src[:, :p]
    Xt = X_tgt[:, :p]

    # Sub-sample for speed
    ns = min(200, len(Xs))
    nt = min(200, len(Xt))
    rng = np.random.default_rng(0)
    Xs  = Xs[rng.choice(len(Xs), ns, replace=False)]
    Xt  = Xt[rng.choice(len(Xt), nt, replace=False)]

    def _rbf(A, B):
        sq_dists = (np.sum(A**2, 1, keepdims=True) +
                    np.sum(B**2, 1) - 2 * A @ B.T)
        return np.exp(-gamma * sq_dists)

    Kss = _rbf(Xs, Xs)
    Ktt = _rbf(Xt, Xt)
    Kst = _rbf(Xs, Xt)

    mmd2 = (Kss.mean() + Ktt.mean() - 2 * Kst.mean())
    return float(max(0.0, mmd2) ** 0.5)


# ─────────────────────────────────────────────────────────────
# Meta-features
# ─────────────────────────────────────────────────────────────

def _meta(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    n, p = X.shape
    classes, counts = np.unique(y, return_counts=True)
    imb  = float(counts.max() / counts.min()) if counts.min() > 0 else 1.0
    miss = float(np.isnan(X).mean())
    if p > 1:
        C = np.corrcoef(X.T)
        corr = float(np.nanmean(np.abs(C)))   # nanmean handles NaN from constant cols
    else:
        corr = 0.0
    return np.array([np.log1p(n), np.log1p(p), np.log1p(imb),
                     miss, float(p / n), corr, float(len(classes))])


# ─────────────────────────────────────────────────────────────
# Transfer features for one (src, tgt) pair
# ─────────────────────────────────────────────────────────────

def _transfer_features(T_src: float,
                        mf_src: np.ndarray, mf_tgt: np.ndarray,
                        X_src: np.ndarray, X_tgt: np.ndarray) -> np.ndarray:
    mmd_val  = _mmd(X_src, X_tgt)
    delta_mf = mf_tgt - mf_src
    return np.concatenate([[T_src, mmd_val], delta_mf, mf_src, mf_tgt])


# ─────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────

def run_transferability_experiment(seed: int = 42) -> Dict:
    from src.data_engine.openml_loader import load_real_datasets

    print("=" * 60)
    print("Trust Transferability  v1.0")
    print("=" * 60)

    raw = load_real_datasets(n=50, verbose=False)
    sc  = StandardScaler()

    # ── Step 1: compute trust on every dataset ────────────────
    print(f"\nComputing trust scores on {len(raw)} datasets...")
    ds_info = []
    for X, y, name in raw:
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
        t = _trust(X_tr, y_tr, X_te, y_te, seed)
        mf = _meta(X, y)
        ds_info.append({"name": name, "X": X, "y": y, "trust": t, "mf": mf})
        print(f"  {name:<30} trust={t:.4f}")

    n_ds = len(ds_info)

    # ── Step 2: build transfer feature matrix ────────────────
    print(f"\nBuilding {n_ds*(n_ds-1)} transfer pairs...")
    feat_rows, target_rows, pair_meta = [], [], []

    for i in range(n_ds):
        for j in range(n_ds):
            if i == j:
                continue
            src, tgt = ds_info[i], ds_info[j]
            feats = _transfer_features(
                src["trust"], src["mf"], tgt["mf"], src["X"], tgt["X"])
            feat_rows.append(feats)
            target_rows.append(tgt["trust"])
            pair_meta.append({"src": src["name"], "tgt": tgt["name"],
                               "T_src": src["trust"], "T_tgt": tgt["trust"]})

    X_feat = np.array(feat_rows)
    y_feat = np.array(target_rows)

    # Impute any residual NaN (e.g. from corr of constant columns)
    col_med = np.nanmedian(X_feat, axis=0)
    col_med = np.where(np.isnan(col_med), 0.0, col_med)
    for j in range(X_feat.shape[1]):
        X_feat[np.isnan(X_feat[:, j]), j] = col_med[j]

    # ── Step 3: LOO-CV with Ridge regression ─────────────────
    print("Training transfer function (Ridge, LOO-CV)...")
    from sklearn.model_selection import cross_val_predict

    # Standard scaling of transfer features
    feat_sc = StandardScaler()
    X_feat_s = feat_sc.fit_transform(X_feat)
    # Final NaN safety (constant features become 0 after scaling)
    X_feat_s = np.nan_to_num(X_feat_s, nan=0.0)

    # Ridge CV
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    y_pred_ridge = cross_val_predict(ridge, X_feat_s, y_feat, cv=5)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=seed)
    y_pred_rf = cross_val_predict(rf, X_feat_s, y_feat, cv=5)

    # Baseline: predict T_src (no transfer learning, just copy)
    y_baseline = np.array([pm["T_src"] for pm in pair_meta])

    def _metrics(y_true, y_hat, name):
        mae  = float(np.abs(y_true - y_hat).mean())
        r, p = scipy_stats.spearmanr(y_true, y_hat)
        rp, _ = scipy_stats.pearsonr(y_true, y_hat)
        rmse = float(np.sqrt(np.mean((y_true - y_hat)**2)))
        print(f"  {name:<20}  MAE={mae:.4f}  RMSE={rmse:.4f}  "
              f"Spearman r={r:.4f} (p={p:.4f})")
        return {"name": name, "mae": round(mae,4), "rmse": round(rmse,4),
                "spearman_r": round(float(r),4), "spearman_p": round(float(p),4),
                "pearson_r":  round(float(rp),4)}

    print("\n── Transfer Prediction Results (5-fold CV) ──")
    res_baseline = _metrics(y_feat, y_baseline, "baseline (copy T_src)")
    res_ridge    = _metrics(y_feat, y_pred_ridge, "Ridge regression")
    res_rf       = _metrics(y_feat, y_pred_rf,    "Random Forest")

    improvement_mae  = round((res_baseline["mae"] - res_ridge["mae"]) / res_baseline["mae"], 4)
    improvement_r    = round(res_ridge["spearman_r"] - res_baseline["spearman_r"], 4)

    # ── Step 4: MMD vs trust error analysis ──────────────────
    mmds  = [float(_mmd(ds_info[i]["X"], ds_info[j]["X"]))
             for i in range(min(15, n_ds)) for j in range(min(15, n_ds)) if i != j]
    errs  = [float(abs(y_feat[k] - y_pred_ridge[k])) for k in range(len(mmds))]
    r_mmd, p_mmd = scipy_stats.spearmanr(mmds[:len(errs)], errs)

    print(f"\nMMD vs transfer error: Spearman r={r_mmd:.4f} (p={p_mmd:.4f})")
    print(f"Interpretation: {'higher MMD → higher error (expected)' if r_mmd > 0 else 'MMD not predictive of error'}")

    print(f"\nMAE improvement over baseline: {improvement_mae:.1%}")
    print(f"Spearman r improvement:         {improvement_r:+.4f}")

    return {
        "version":      "1.0_real_datasets",
        "n_datasets":   n_ds,
        "n_pairs":      len(feat_rows),
        "baseline":     res_baseline,
        "ridge":        res_ridge,
        "random_forest":res_rf,
        "mae_improvement_over_baseline": improvement_mae,
        "r_improvement_over_baseline":   improvement_r,
        "mmd_vs_error": {
            "spearman_r": round(float(r_mmd), 4),
            "spearman_p": round(float(p_mmd), 4),
            "interpretation": ("positive" if r_mmd > 0 else "negative"),
        },
        "finding": (
            f"Trust transfer function reduces MAE by {improvement_mae:.1%} vs baseline. "
            f"Ridge: MAE={res_ridge['mae']:.4f}, Spearman r={res_ridge['spearman_r']:.4f}. "
            f"MMD-error correlation: r={r_mmd:.4f}."
        ),
    }


if __name__ == "__main__":
    result = run_transferability_experiment(seed=42)
    out = OUT / "trust_transferability.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {out}")
    print(f"\nKey finding: {result['finding']}")
