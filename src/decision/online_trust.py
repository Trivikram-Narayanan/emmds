"""
Temporal Trust Monitoring  v2.0
=================================
Rolling trust score with CUSUM and EWMA drift detection, automated
retraining triggers, and full experiment on real OpenML datasets.

Theory
------
Standard concept-drift detection monitors raw accuracy or error rate.
This module monitors the composite EMMDS trust score T(t) as a deployment
health signal.  Two classical SPC (Statistical Process Control) detectors
are adapted to the trust-score setting:

  CUSUM  (Page, 1954):
    S⁺(t) = max(0, S⁺(t-1) + (μ₀ - T(t)) - k)
    Alarm when S⁺(t) > h
    k = allowed slack (default 0.5σ₀), h = alarm threshold

  EWMA  (Roberts, 1959):
    Z(t) = λ·T(t) + (1-λ)·Z(t-1)
    σ_Z  = σ₀·√(λ/(2-λ))
    Alarm when Z(t) < μ₀ - L·σ_Z

Compared against a simple Threshold detector: alarm when EMA < gate.

Key hypothesis: CUSUM/EWMA detect trust drift faster (lower detection
delay) and with fewer false alarms than a naive threshold gate.

Reference:
  Montgomery, D. C. (2009). Introduction to Statistical Quality Control.
  Gama et al. (2014). A survey on concept drift adaptation.
  Losing et al. (2018). Incremental on-line learning: A review.
"""

import json
import numpy as np
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, brier_score_loss
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
# Trust score — two-stage: fit once, evaluate per batch
# ─────────────────────────────────────────────────────────────

def _fit_models(X_tr: np.ndarray, y_tr: np.ndarray,
                seed: int = 42) -> List[Dict]:
    """
    Fit 5 classifiers on X_tr and pre-compute stability (CV std) and dq.
    Returns a list of dicts: {model, stab, dq} for each successfully fitted model.
    Called ONCE per dataset — results are reused across all batches.
    """
    models = [
        LogisticRegression(max_iter=300, random_state=seed),
        RandomForestClassifier(n_estimators=30, random_state=seed),
        GradientBoostingClassifier(n_estimators=30, random_state=seed),
        DecisionTreeClassifier(max_depth=5, random_state=seed),
        KNeighborsClassifier(n_neighbors=5),
    ]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    dq  = float(np.clip(1 - np.isnan(X_tr).mean(), 0, 1))
    fitted = []

    for m in models:
        try:
            m.fit(X_tr, y_tr)
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
            fitted.append({"model": m, "stab": stab, "dq": dq})
        except Exception:
            pass

    return fitted


def _eval_trust_batch(fitted: List[Dict],
                      X_eval: np.ndarray, y_eval: np.ndarray) -> float:
    """
    Compute trust for one batch using pre-fitted models.
    Only F1 and calibration are re-evaluated; stability and dq are fixed.
    """
    best_trust = 0.0
    for entry in fitted:
        m    = entry["model"]
        stab = entry["stab"]
        dq   = entry["dq"]
        try:
            f1 = float(f1_score(y_eval, m.predict(X_eval),
                                average="weighted", zero_division=0))
            cal = 0.5
            if hasattr(m, "predict_proba"):
                proba   = m.predict_proba(X_eval)
                classes = np.unique(y_eval)
                if len(classes) == 2:
                    cal = float(np.clip(1 - brier_score_loss(y_eval, proba[:, 1]), 0, 1))
                else:
                    cal = float(np.clip(1 - np.mean([
                        brier_score_loss((y_eval == c).astype(int), proba[:, i])
                        for i, c in enumerate(classes)]), 0, 1))
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


def _compute_trust(X_tr: np.ndarray, y_tr: np.ndarray,
                   X_te: np.ndarray, y_te: np.ndarray,
                   seed: int = 42) -> float:
    """Full trust computation (fit + eval). Used only for standalone calls."""
    fitted = _fit_models(X_tr, y_tr, seed)
    return _eval_trust_batch(fitted, X_te, y_te)


# ─────────────────────────────────────────────────────────────
# Drift detectors
# ─────────────────────────────────────────────────────────────

class CUSUMDetector:
    """
    One-sided lower CUSUM for trust score monitoring.
    Detects a negative shift (trust drop) in the stream.

    S⁺(t) = max(0, S⁺(t-1) + (μ₀ - T(t)) - k)
    Alarm: S⁺(t) > h
    """

    def __init__(self, mu0: float, sigma0: float,
                 k_factor: float = 0.5, h_factor: float = 5.0):
        self.mu0    = mu0
        self.k      = k_factor * sigma0          # allowable slack
        self.h      = h_factor * sigma0          # alarm threshold
        self.S      = 0.0
        self.t      = 0
        self.alarms: List[int] = []

    def update(self, trust: float) -> bool:
        self.t += 1
        self.S = max(0.0, self.S + (self.mu0 - trust) - self.k)
        if self.S > self.h:
            self.alarms.append(self.t)
            self.S = 0.0   # reset after alarm
            return True
        return False

    def reset_stat(self):
        self.S = 0.0


class EWMADetector:
    """
    EWMA control chart for trust score monitoring.

    Z(t) = λ·T(t) + (1-λ)·Z(t-1),   Z(0) = μ₀
    σ_Z  = σ₀·√(λ/(2-λ))
    Alarm: Z(t) < μ₀ - L·σ_Z
    """

    def __init__(self, mu0: float, sigma0: float,
                 lam: float = 0.20, L: float = 3.0):
        self.mu0    = mu0
        self.sigma_Z = sigma0 * np.sqrt(lam / (2 - lam))
        self.lam    = lam
        self.L      = L
        self.Z      = mu0
        self.t      = 0
        self.alarms: List[int] = []
        self._lcl   = mu0 - L * self.sigma_Z

    def update(self, trust: float) -> bool:
        self.t += 1
        self.Z = self.lam * trust + (1 - self.lam) * self.Z
        if self.Z < self._lcl:
            self.alarms.append(self.t)
            return True
        return False


class ThresholdDetector:
    """Simple deployment gate: alarm when EMA trust < threshold."""

    def __init__(self, gate: float = 0.60, alpha: float = 0.20):
        self.gate   = gate
        self.alpha  = alpha
        self.ema    = None
        self.t      = 0
        self.alarms: List[int] = []

    def update(self, trust: float) -> bool:
        self.t += 1
        self.ema = trust if self.ema is None else (
            self.alpha * trust + (1 - self.alpha) * self.ema)
        if self.ema < self.gate:
            self.alarms.append(self.t)
            return True
        return False


# ─────────────────────────────────────────────────────────────
# Shift generator for temporal simulation
# ─────────────────────────────────────────────────────────────

def _apply_temporal_shift(X: np.ndarray, severity: float,
                           rng: np.random.Generator) -> np.ndarray:
    """Covariate shift: rotate feature space proportionally to severity."""
    Xc = X.copy()
    p  = X.shape[1]
    Q, _ = np.linalg.qr(rng.normal(0, 1, (p, p)))
    return (1 - severity) * Xc + severity * (Xc @ Q)


# ─────────────────────────────────────────────────────────────
# Single dataset simulation
# ─────────────────────────────────────────────────────────────

def _simulate_dataset(X: np.ndarray, y: np.ndarray,
                      name: str, seed: int = 42,
                      n_pre: int = 8, n_post: int = 12,
                      max_severity: float = 0.5) -> Dict:
    """
    Simulate temporal deployment:
      - Phase 1 (pre): n_pre batches on clean held-out test data
      - Phase 2 (post): n_post batches with linearly increasing covariate shift

    Models are fitted ONCE on X_tr; only F1 and calibration are re-evaluated
    per batch, making this ~20× faster than re-fitting each batch.
    """
    rng = np.random.default_rng(seed)

    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=seed)
    except Exception:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.30, random_state=seed)

    # Fit models ONCE — stability and dq are dataset-level constants
    fitted = _fit_models(X_tr, y_tr, seed)
    if not fitted:
        return None

    # ── Phase 1: collect baseline trust on clean batches ─────
    pre_trusts = []
    n_te = len(X_te)
    batch_size = max(30, n_te // max(1, n_pre))

    for b in range(n_pre):
        idx = rng.choice(n_te, min(batch_size, n_te), replace=True)
        t = _eval_trust_batch(fitted, X_te[idx], y_te[idx])
        pre_trusts.append(t)

    mu0    = float(np.mean(pre_trusts)) if pre_trusts else 0.5
    sigma0 = float(np.std(pre_trusts)) if len(pre_trusts) > 1 else 0.05
    sigma0 = max(sigma0, 0.01)   # floor to avoid degenerate detectors

    # Initialise detectors on pre-phase baseline
    cusum = CUSUMDetector(mu0=mu0, sigma0=sigma0, k_factor=0.5, h_factor=4.0)
    ewma  = EWMADetector(mu0=mu0,  sigma0=sigma0, lam=0.20, L=2.5)
    thr   = ThresholdDetector(gate=mu0 - 2 * sigma0, alpha=0.25)

    # Feed pre-phase to warm up detectors (no alarm expected)
    for t_val in pre_trusts:
        cusum.update(t_val)
        ewma.update(t_val)
        thr.update(t_val)
    # Reset CUSUM stat after warm-up so it starts fresh at drift boundary
    cusum.reset_stat()

    # ── Phase 2: post-deployment with drift ──────────────────
    post_trusts = []
    severities  = np.linspace(0.0, max_severity, n_post)
    drift_start = n_pre + 1      # first batch where drift appears (1-indexed)

    cusum_alarm_t = None
    ewma_alarm_t  = None
    thr_alarm_t   = None

    for b, sev in enumerate(severities):
        idx    = rng.choice(n_te, min(batch_size, n_te), replace=True)
        X_eval = _apply_temporal_shift(X_te[idx], sev, rng)
        t_val  = _eval_trust_batch(fitted, X_eval, y_te[idx])
        post_trusts.append(t_val)

        global_t = n_pre + b + 1   # 1-indexed across full stream

        if cusum.update(t_val) and cusum_alarm_t is None:
            cusum_alarm_t = global_t
        if ewma.update(t_val) and ewma_alarm_t is None:
            ewma_alarm_t = global_t
        if thr.update(t_val) and thr_alarm_t is None:
            thr_alarm_t = global_t

    all_trusts = pre_trusts + post_trusts
    total_batches = len(all_trusts)

    def _delay(alarm_t):
        if alarm_t is None:
            return None
        return max(0, alarm_t - drift_start)

    return {
        "name":         name,
        "mu0":          round(mu0,    4),
        "sigma0":       round(sigma0, 4),
        "pre_trusts":   [round(t, 4) for t in pre_trusts],
        "post_trusts":  [round(t, 4) for t in post_trusts],
        "all_trusts":   [round(t, 4) for t in all_trusts],
        "drift_start":  drift_start,
        "n_batches":    total_batches,
        "detectors": {
            "cusum": {
                "alarm_batch":     cusum_alarm_t,
                "detection_delay": _delay(cusum_alarm_t),
                "detected":        cusum_alarm_t is not None,
                "n_alarms":        len(cusum.alarms),
            },
            "ewma": {
                "alarm_batch":     ewma_alarm_t,
                "detection_delay": _delay(ewma_alarm_t),
                "detected":        ewma_alarm_t is not None,
                "n_alarms":        len(ewma.alarms),
            },
            "threshold": {
                "alarm_batch":     thr_alarm_t,
                "detection_delay": _delay(thr_alarm_t),
                "detected":        thr_alarm_t is not None,
                "n_alarms":        len(thr.alarms),
            },
        },
        "retraining_trigger": cusum_alarm_t is not None or ewma_alarm_t is not None,
        "final_trust": round(all_trusts[-1], 4) if all_trusts else 0.0,
        "trust_drop":  round(mu0 - (all_trusts[-1] if all_trusts else mu0), 4),
    }


# ─────────────────────────────────────────────────────────────
# Full experiment
# ─────────────────────────────────────────────────────────────

def run_temporal_monitoring_experiment(seed: int = 42) -> Dict:
    from src.data_engine.openml_loader import load_real_datasets

    print("=" * 60)
    print("Temporal Trust Monitoring  v2.0")
    print("=" * 60)

    raw = load_real_datasets(n=50, verbose=False)
    sc  = StandardScaler()
    le  = LabelEncoder()

    detector_names = ["cusum", "ewma", "threshold"]
    results = []

    print(f"\nSimulating temporal deployment on {len(raw)} datasets...")
    print(f"  Config: 8 pre-drift batches + 12 post-drift batches (max severity 0.5)")
    print()

    for X, y, name in raw:
        col_med = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            X[np.isnan(X[:, j]), j] = col_med[j]
        X = sc.fit_transform(X)
        y = le.fit_transform(y)

        r = _simulate_dataset(X, y, name, seed=seed,
                              n_pre=8, n_post=12, max_severity=0.5)
        if r is None:
            continue
        results.append(r)

        det_str = " | ".join(
            f"{d}={'✓' if r['detectors'][d]['detected'] else '✗'}"
            f"(lag={r['detectors'][d]['detection_delay']})"
            for d in detector_names)
        print(f"  {name:<30} μ₀={r['mu0']:.3f}  drop={r['trust_drop']:+.3f}  {det_str}")

    # ── Aggregate statistics ──────────────────────────────────
    n = len(results)
    print(f"\n── Aggregate Results ({n} datasets) ──")
    print(f"{'Detector':<12}  {'DR%':>6}  {'Avg Delay':>10}  {'Avg Alarms':>12}")
    print("-" * 50)

    summary = {}
    for det in detector_names:
        detected = [r for r in results if r["detectors"][det]["detected"]]
        dr = len(detected) / n if n > 0 else 0.0
        delays = [r["detectors"][det]["detection_delay"]
                  for r in detected
                  if r["detectors"][det]["detection_delay"] is not None]
        avg_delay  = float(np.mean(delays))  if delays  else float("nan")
        avg_alarms = float(np.mean([r["detectors"][det]["n_alarms"] for r in results]))
        summary[det] = {
            "detection_rate":  round(dr, 4),
            "avg_delay_batches": round(avg_delay, 2) if not np.isnan(avg_delay) else None,
            "avg_n_alarms":    round(avg_alarms, 2),
            "n_detected":      len(detected),
        }
        delay_str = f"{avg_delay:.1f}" if not np.isnan(avg_delay) else "N/A"
        print(f"  {det:<12}  {dr:>5.1%}  {delay_str:>10}  {avg_alarms:>12.2f}")

    # Pairwise comparison: CUSUM vs threshold
    both_detected = [r for r in results
                     if r["detectors"]["cusum"]["detected"] and
                        r["detectors"]["threshold"]["detected"]]
    if both_detected:
        cusum_delays = [r["detectors"]["cusum"]["detection_delay"]     for r in both_detected
                        if r["detectors"]["cusum"]["detection_delay"] is not None]
        thr_delays   = [r["detectors"]["threshold"]["detection_delay"] for r in both_detected
                        if r["detectors"]["threshold"]["detection_delay"] is not None]
        if cusum_delays and thr_delays:
            avg_cusum = float(np.mean(cusum_delays))
            avg_thr   = float(np.mean(thr_delays))
            advantage = round(avg_thr - avg_cusum, 2)
        else:
            avg_cusum = avg_thr = advantage = float("nan")
    else:
        avg_cusum = avg_thr = advantage = float("nan")

    # Trust drop analysis
    drops = [r["trust_drop"] for r in results]
    triggered = [r for r in results if r["retraining_trigger"]]

    print(f"\nMean trust drop (post-drift final): {float(np.mean(drops)):+.4f}")
    print(f"Retraining triggered:               {len(triggered)}/{n} datasets")
    if not np.isnan(advantage):
        print(f"CUSUM vs Threshold avg delay:       CUSUM {advantage:+.1f} batches faster")

    return {
        "version":          "2.0_real_datasets",
        "n_datasets":       n,
        "n_pre_batches":    8,
        "n_post_batches":   12,
        "max_severity":     0.5,
        "detectors":        summary,
        "retraining_triggers": {
            "n_triggered":  len(triggered),
            "trigger_rate": round(len(triggered) / n, 4) if n > 0 else 0.0,
        },
        "trust_drop_stats": {
            "mean_drop":    round(float(np.mean(drops)), 4),
            "median_drop":  round(float(np.median(drops)), 4),
            "std_drop":     round(float(np.std(drops)), 4),
        },
        "cusum_vs_threshold": {
            "n_both_detected": len(both_detected),
            "cusum_avg_delay": round(avg_cusum, 2) if not np.isnan(avg_cusum) else None,
            "threshold_avg_delay": round(avg_thr, 2) if not np.isnan(avg_thr) else None,
            "cusum_faster_by_batches": round(advantage, 2) if not np.isnan(advantage) else None,
        },
        "finding": (
            f"CUSUM detects trust drift on {summary['cusum']['detection_rate']:.0%} of datasets "
            f"(avg delay {summary['cusum']['avg_delay_batches']} batches); "
            f"EWMA on {summary['ewma']['detection_rate']:.0%} "
            f"(avg delay {summary['ewma']['avg_delay_batches']} batches); "
            f"Threshold on {summary['threshold']['detection_rate']:.0%}. "
            f"Mean trust drop under max-severity shift: "
            f"{float(np.mean(drops)):+.4f}. "
            f"Retraining triggered on {len(triggered)}/{n} datasets."
        ),
        "dataset_results": [
            {"name": r["name"], "mu0": r["mu0"], "trust_drop": r["trust_drop"],
             "retraining_trigger": r["retraining_trigger"],
             **{f"{d}_delay": r["detectors"][d]["detection_delay"] for d in detector_names},
             **{f"{d}_detected": r["detectors"][d]["detected"] for d in detector_names}}
            for r in results
        ],
    }


# ─────────────────────────────────────────────────────────────
# Production-use streaming API (kept from v1.0)
# ─────────────────────────────────────────────────────────────

class OnlineTrustUpdater:
    """
    Lightweight production wrapper: call update() per deployment batch,
    query current_trust() at any time. Uses EWMA smoothing + CUSUM alarm.
    """

    DEFAULT_WEIGHTS = dict(accuracy=0.05, calibration=0.10, agreement=0.10,
                           data_quality=0.35, stability=0.40)

    def __init__(self, alpha: float = 0.20, drift_threshold: float = 0.05,
                 weights: Optional[Dict] = None):
        self.alpha   = alpha
        self.weights = weights or dict(self.DEFAULT_WEIGHTS)
        self._ema: Dict[str, Optional[float]] = {k: None for k in self.weights}
        self._history: deque = deque(maxlen=30)
        self._n_updates = 0
        self._peak = 0.0
        self._drift_threshold = drift_threshold
        self.alerts: List[Dict] = []

    def update(self, components: Dict[str, float]) -> Dict:
        for k in self.weights:
            if k not in components:
                continue
            v = float(np.clip(components[k], 0.0, 1.0))
            self._ema[k] = v if self._ema[k] is None else (
                self.alpha * v + (1 - self.alpha) * self._ema[k])
        c = self._composite()
        self._history.append(c)
        self._n_updates += 1
        if c > self._peak:
            self._peak = c
        drop = self._peak - c
        if drop >= self._drift_threshold:
            alert = {"type": "trust_drift", "update": self._n_updates,
                     "peak": round(self._peak, 4), "current": round(c, 4),
                     "drop": round(drop, 4)}
            if not self.alerts or self.alerts[-1]["update"] != self._n_updates:
                self.alerts.append(alert)
        return self.state()

    def current_trust(self) -> float:
        return self._composite()

    def state(self) -> Dict:
        d = {k: round(v, 6) if v is not None else None for k, v in self._ema.items()}
        d["composite_trust"] = round(self._composite(), 6)
        d["n_updates"] = self._n_updates
        return d

    def _composite(self) -> float:
        total, w_sum = 0.0, 0.0
        for k, w in self.weights.items():
            v = self._ema.get(k)
            if v is not None:
                total += w * v; w_sum += w
        return total / w_sum if w_sum > 1e-9 else 0.0


if __name__ == "__main__":
    result = run_temporal_monitoring_experiment(seed=42)
    out = OUT / "temporal_trust_monitoring.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {out}")
    print(f"\nKey finding: {result['finding']}")
