"""
EMMDS Deployment Monitor
=========================
Connects drift detection to actionable retraining recommendations.
Turns EMMDS from a one-time model selection tool into a full
deployment lifecycle management system.

The monitor:
  1. Receives batches of new production data
  2. Computes drift signals (KS test, PSI) against training reference
  3. Monitors model performance on incoming batches
  4. Computes a Deployment Health Score (DHS)
  5. Issues one of three recommendations:
       DEPLOY    — model is healthy, continue
       MONITOR   — early warning signals, increase monitoring frequency
       RETRAIN   — drift or degradation exceeds threshold, retrain now

The Deployment Health Score uses the same trust components but
applied to live production signals rather than training-time metrics.

This directly addresses the distribution shift finding:
trust score predicts shift vulnerability at training time, and
the monitor detects when that vulnerability is being realised.
"""

import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List
from scipy import stats
from pathlib import Path

warnings.filterwarnings('ignore')


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class BatchMetrics:
    """Performance and drift metrics for one data batch."""
    batch_id:          int
    n_samples:         int
    # Performance (requires ground truth labels, may be None if delayed)
    performance_f1:    Optional[float] = None
    performance_acc:   Optional[float] = None
    performance_delta: Optional[float] = None   # vs baseline
    # Drift signals
    ks_statistic_mean: float = 0.0
    psi_mean:          float = 0.0
    drifted_features:  int   = 0
    drift_severity:    str   = "none"
    # Health
    deployment_health: float = 1.0
    recommendation:    str   = "DEPLOY"
    action_required:   bool  = False


@dataclass
class DeploymentState:
    """Full deployment history for one model."""
    model_name:         str
    baseline_f1:        float
    baseline_trust:     float
    training_timestamp: str
    batch_history:      List[BatchMetrics] = field(default_factory=list)
    retrain_count:      int = 0
    total_batches:      int = 0


# ── Main monitor class ────────────────────────────────────────────────

class DeploymentMonitor:
    """
    Monitors a deployed model across incoming data batches.

    Usage:
        monitor = DeploymentMonitor()
        monitor.fit_reference(X_train, y_train, model, baseline_f1, baseline_trust)

        # For each new batch in production:
        result = monitor.process_batch(X_new, y_new_if_available)
        if result.recommendation == "RETRAIN":
            trigger_retraining()
    """

    # Thresholds
    KS_DRIFT_THRESHOLD    = 0.10   # KS p-value < 0.05 per feature → flag
    PSI_WARNING_THRESHOLD = 0.10   # PSI 0.10-0.20 → moderate drift
    PSI_CRITICAL_THRESHOLD = 0.20  # PSI > 0.20 → severe drift
    PERF_WARNING_DELTA    = -0.05  # F1 drops 5% → warning
    PERF_CRITICAL_DELTA   = -0.10  # F1 drops 10% → retrain
    HEALTH_MONITOR_THRESH = 0.65   # DHS below this → MONITOR
    HEALTH_RETRAIN_THRESH = 0.45   # DHS below this → RETRAIN

    def __init__(self, model_name: str = "best_model"):
        self.model_name      = model_name
        self._reference_X:   Optional[np.ndarray] = None
        self._reference_mean: Optional[np.ndarray] = None
        self._reference_std:  Optional[np.ndarray] = None
        self._model          = None
        self._baseline_f1    = 0.0
        self._baseline_trust = 0.0
        self._batch_counter  = 0
        self._history:        List[BatchMetrics] = []
        self._alert_log:      List[dict] = []

    def fit_reference(
        self,
        X_train:        np.ndarray,
        y_train:        np.ndarray,
        model,
        baseline_f1:    float,
        baseline_trust: float,
        model_name:     str = None,
    ) -> None:
        """
        Store training distribution statistics as reference.
        Must be called after model training, before deployment.

        Args:
            X_train:        Training feature matrix
            y_train:        Training labels
            model:          Fitted model
            baseline_f1:    F1 score on validation set at training time
            baseline_trust: Trust score at training time
            model_name:     Override model name
        """
        self._reference_X     = X_train.copy()
        self._reference_mean  = X_train.mean(axis=0)
        self._reference_std   = X_train.std(axis=0) + 1e-8
        self._model           = model
        self._baseline_f1     = baseline_f1
        self._baseline_trust  = baseline_trust
        if model_name:
            self.model_name = model_name

    def process_batch(
        self,
        X_batch:         np.ndarray,
        y_batch:         Optional[np.ndarray] = None,
        return_details:  bool = True,
    ) -> BatchMetrics:
        """
        Process one incoming data batch.

        Args:
            X_batch:  Feature matrix for new data batch
            y_batch:  True labels if available (may be None — delayed labels)
            return_details: If True, compute full drift details

        Returns:
            BatchMetrics with recommendation
        """
        if self._reference_X is None:
            raise RuntimeError("Call fit_reference() before process_batch()")

        self._batch_counter += 1
        batch_id = self._batch_counter

        # ── Drift detection ───────────────────────────────────────────
        ks_stats, psi_scores, n_drifted = self._compute_drift(X_batch)
        ks_mean  = float(np.mean(ks_stats))
        psi_mean = float(np.mean(psi_scores))

        if   psi_mean > self.PSI_CRITICAL_THRESHOLD:  drift_severity = "severe"
        elif psi_mean > self.PSI_WARNING_THRESHOLD:   drift_severity = "moderate"
        elif ks_mean  > 0.15:                         drift_severity = "low"
        else:                                          drift_severity = "none"

        # ── Performance (if labels available) ─────────────────────────
        perf_f1     = None
        perf_acc    = None
        perf_delta  = None

        if y_batch is not None and self._model is not None:
            try:
                from sklearn.metrics import f1_score, accuracy_score
                y_pred  = self._model.predict(X_batch)
                perf_f1  = float(f1_score(y_batch, y_pred,
                                          average='weighted', zero_division=0))
                perf_acc = float(accuracy_score(y_batch, y_pred))
                perf_delta = round(perf_f1 - self._baseline_f1, 6)
            except Exception:
                pass

        # ── Deployment Health Score ───────────────────────────────────
        dhs = self._compute_dhs(
            drift_severity=drift_severity,
            psi_mean=psi_mean,
            perf_delta=perf_delta,
        )

        # ── Recommendation ────────────────────────────────────────────
        recommendation, action_required = self._make_recommendation(
            dhs=dhs,
            drift_severity=drift_severity,
            perf_delta=perf_delta,
        )

        # ── Build result ──────────────────────────────────────────────
        result = BatchMetrics(
            batch_id          = batch_id,
            n_samples         = len(X_batch),
            performance_f1    = round(perf_f1,    6) if perf_f1    is not None else None,
            performance_acc   = round(perf_acc,   6) if perf_acc   is not None else None,
            performance_delta = round(perf_delta, 6) if perf_delta is not None else None,
            ks_statistic_mean = round(ks_mean,    6),
            psi_mean          = round(psi_mean,   6),
            drifted_features  = int(n_drifted),
            drift_severity    = drift_severity,
            deployment_health = round(dhs,        4),
            recommendation    = recommendation,
            action_required   = action_required,
        )

        self._history.append(result)

        if action_required:
            self._alert_log.append({
                'batch_id':      batch_id,
                'recommendation': recommendation,
                'dhs':           dhs,
                'drift':         drift_severity,
                'perf_delta':    perf_delta,
            })

        return result

    # ── Internal methods ──────────────────────────────────────────────

    def _compute_drift(self, X_batch: np.ndarray):
        """Per-feature KS test and PSI against reference distribution."""
        n_feat    = min(X_batch.shape[1], self._reference_X.shape[1])
        ks_stats  = []
        psi_scores = []
        n_drifted  = 0

        for j in range(n_feat):
            ref = self._reference_X[:, j]
            new = X_batch[:, j]

            # KS test
            ks_stat, ks_p = stats.ks_2samp(ref, new)
            ks_stats.append(float(ks_stat))
            if ks_p < 0.05:
                n_drifted += 1

            # PSI
            psi = self._psi(ref, new)
            psi_scores.append(float(psi))

        return np.array(ks_stats), np.array(psi_scores), n_drifted

    def _psi(self, reference: np.ndarray, actual: np.ndarray,
              bins: int = 8) -> float:
        """Population Stability Index."""
        breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 2:
            return 0.0
        ref_pcts = np.histogram(reference, bins=breakpoints)[0] / len(reference)
        act_pcts = np.histogram(actual,    bins=breakpoints)[0] / len(actual)
        ref_pcts = np.clip(ref_pcts, 1e-6, 1)
        act_pcts = np.clip(act_pcts, 1e-6, 1)
        return float(np.sum((act_pcts - ref_pcts) * np.log(act_pcts / ref_pcts)))

    def _compute_dhs(
        self,
        drift_severity: str,
        psi_mean:       float,
        perf_delta:     Optional[float],
    ) -> float:
        """
        Deployment Health Score ∈ [0, 1].
        Higher = healthier deployment.

        Components:
          drift_health:   1 - normalised PSI
          perf_health:    based on F1 delta vs baseline
          stability:      rolling variance of recent DHS
        """
        # Drift health
        drift_health = float(np.clip(1.0 - psi_mean / 0.3, 0, 1))

        # Performance health
        if perf_delta is not None:
            # Map delta to [0,1]: 0 delta → 1.0, -0.2 delta → 0.0
            perf_health = float(np.clip(1.0 + perf_delta / 0.2, 0, 1))
        else:
            # No ground truth — neutral, slightly penalised for uncertainty
            perf_health = 0.7

        # Rolling stability (penalty if recent DHS is volatile)
        recent = [b.deployment_health for b in self._history[-5:]]
        if len(recent) >= 3:
            stability = float(np.clip(1.0 - np.std(recent) * 5, 0, 1))
        else:
            stability = 1.0

        dhs = 0.45 * drift_health + 0.40 * perf_health + 0.15 * stability
        return float(np.clip(dhs, 0.0, 1.0))

    def _make_recommendation(
        self,
        dhs:            float,
        drift_severity: str,
        perf_delta:     Optional[float],
    ) -> tuple:
        """Return (recommendation_string, action_required_bool)."""
        # Critical performance drop → always retrain
        if perf_delta is not None and perf_delta < self.PERF_CRITICAL_DELTA:
            return "RETRAIN", True

        # Severe drift → retrain
        if drift_severity == "severe":
            return "RETRAIN", True

        # DHS below retrain threshold
        if dhs < self.HEALTH_RETRAIN_THRESH:
            return "RETRAIN", True

        # DHS in warning zone
        if dhs < self.HEALTH_MONITOR_THRESH:
            return "MONITOR", False

        # Moderate drift → monitor
        if drift_severity in ("moderate",):
            return "MONITOR", False

        # Performance warning
        if perf_delta is not None and perf_delta < self.PERF_WARNING_DELTA:
            return "MONITOR", False

        return "DEPLOY", False

    # ── History and reporting ─────────────────────────────────────────

    def get_history_df(self) -> pd.DataFrame:
        """Return full batch history as DataFrame."""
        if not self._history:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                'batch_id':          b.batch_id,
                'n_samples':         b.n_samples,
                'f1':                b.performance_f1,
                'f1_delta':          b.performance_delta,
                'ks_mean':           b.ks_statistic_mean,
                'psi_mean':          b.psi_mean,
                'drifted_features':  b.drifted_features,
                'drift_severity':    b.drift_severity,
                'deployment_health': b.deployment_health,
                'recommendation':    b.recommendation,
                'action_required':   b.action_required,
            }
            for b in self._history
        ])

    def get_summary(self) -> dict:
        """Return deployment monitoring summary."""
        if not self._history:
            return {}
        hist_df = self.get_history_df()
        return {
            'model_name':           self.model_name,
            'baseline_f1':          self._baseline_f1,
            'baseline_trust':       self._baseline_trust,
            'total_batches':        len(self._history),
            'total_samples':        int(sum(b.n_samples for b in self._history)),
            'retrain_alerts':       int(sum(1 for b in self._history
                                           if b.recommendation == "RETRAIN")),
            'monitor_alerts':       int(sum(1 for b in self._history
                                           if b.recommendation == "MONITOR")),
            'mean_dhs':             round(float(hist_df['deployment_health'].mean()), 4),
            'min_dhs':              round(float(hist_df['deployment_health'].min()), 4),
            'mean_drift_severity':  hist_df['drift_severity'].mode()[0]
                                    if len(hist_df) > 0 else "none",
            'alert_log':            self._alert_log,
        }

    def print_status(self) -> None:
        """Print a readable deployment status."""
        s = self.get_summary()
        if not s:
            print("No batches processed yet.")
            return

        print(f"\n{'═'*55}")
        print(f"  EMMDS DEPLOYMENT MONITOR — {s['model_name']}")
        print(f"{'═'*55}")
        print(f"  Baseline F1:     {s['baseline_f1']:.4f}")
        print(f"  Baseline Trust:  {s['baseline_trust']:.4f}")
        print(f"  Batches:         {s['total_batches']} ({s['total_samples']:,} samples)")
        print(f"  Mean DHS:        {s['mean_dhs']:.4f}")
        print(f"  Min DHS:         {s['min_dhs']:.4f}")
        print(f"  RETRAIN alerts:  {s['retrain_alerts']}")
        print(f"  MONITOR alerts:  {s['monitor_alerts']}")
        if s['alert_log']:
            print(f"\n  Alert log:")
            for a in s['alert_log'][-5:]:
                print(f"    Batch {a['batch_id']:3d}: {a['recommendation']:8s} "
                      f"DHS={a['dhs']:.3f}  drift={a['drift']}")
        print(f"{'═'*55}\n")
