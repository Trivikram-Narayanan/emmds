"""
EMMDS Data Drift Detector
Detects distributional shift between a reference dataset
and a new dataset using:
  - Kolmogorov–Smirnov (KS) test  per numerical feature
  - Population Stability Index (PSI) per numerical feature
  - Chi-squared test for categorical features

Output tells the pipeline whether the new data is "safe" to use
the previously trained model on, or whether retraining is needed.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Thresholds
KS_P_THRESHOLD    = 0.05    # p < 0.05 → significant drift
PSI_WARNING       = 0.10    # 0.10–0.20 → moderate drift
PSI_CRITICAL      = 0.20    # > 0.20    → severe drift
DRIFT_FEAT_RATIO  = 0.30    # If >30% features drift → dataset-level drift


class DataDriftDetector:
    """
    Compares a reference (training) dataset to a new dataset.
    Reports per-feature drift and a global drift verdict.
    """

    def __init__(self, ks_threshold: float = KS_P_THRESHOLD,
                 psi_threshold: float = PSI_CRITICAL):
        self.ks_threshold  = ks_threshold
        self.psi_threshold = psi_threshold
        self.reference_stats: dict = {}
        self.last_report: dict     = {}

    # ── Reference fitting ──────────────────────────────────────────

    def fit(self, reference_df: pd.DataFrame, target_col: Optional[str] = None) -> None:
        """
        Store summary statistics from the reference (training) dataset.
        Call this once after training.
        """
        X = reference_df.drop(columns=[target_col]) if target_col else reference_df

        num_cols = list(X.select_dtypes(include=[np.number]).columns)
        cat_cols = list(X.select_dtypes(include=["object","category","bool"]).columns)

        self.reference_stats = {
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "num_stats": {
                col: {
                    "values": X[col].dropna().tolist(),  # Store for KS test
                    "mean":   float(X[col].mean()),
                    "std":    float(X[col].std()),
                    "min":    float(X[col].min()),
                    "max":    float(X[col].max()),
                    "q10":    float(X[col].quantile(0.10)),
                    "q90":    float(X[col].quantile(0.90)),
                }
                for col in num_cols
            },
            "cat_stats": {
                col: {
                    "value_counts": X[col].value_counts(normalize=True).to_dict()
                }
                for col in cat_cols
            },
            "n_reference": len(X),
        }
        logger.info(
            f"DriftDetector fitted on {len(X)} samples | "
            f"Numerical: {len(num_cols)} | Categorical: {len(cat_cols)}"
        )

    # ── Drift detection ────────────────────────────────────────────

    def detect(self, new_df: pd.DataFrame, target_col: Optional[str] = None) -> dict:
        """
        Compare new_df to the fitted reference.

        Returns:
            {
                "drift_detected":   bool,
                "drift_score":      float (0–1),
                "severity":         "none"|"low"|"moderate"|"severe",
                "drifted_features": [...],
                "feature_reports":  {col: {ks_stat, ks_p, psi, drifted}},
                "recommendation":   str,
            }
        """
        if not self.reference_stats:
            logger.warning("DriftDetector not fitted. Call .fit() first.")
            return {"drift_detected": False, "error": "Not fitted"}

        X_new = new_df.drop(columns=[target_col]) if target_col else new_df
        feature_reports = {}
        drifted = []

        # Numerical: KS test + PSI
        for col in self.reference_stats["num_cols"]:
            if col not in X_new.columns:
                continue
            ref_vals  = np.array(self.reference_stats["num_stats"][col]["values"])
            new_vals  = X_new[col].dropna().to_numpy()

            ks_stat, ks_p = stats.ks_2samp(ref_vals, new_vals)
            psi           = self._psi(ref_vals, new_vals)
            col_drifted   = (ks_p < self.ks_threshold) or (psi > self.psi_threshold)

            feature_reports[col] = {
                "ks_statistic": round(float(ks_stat), 4),
                "ks_p_value":   round(float(ks_p), 4),
                "psi":          round(float(psi), 4),
                "drifted":      col_drifted,
                "type":         "numerical",
            }
            if col_drifted:
                drifted.append(col)

        # Categorical: Chi-squared test
        for col in self.reference_stats["cat_cols"]:
            if col not in X_new.columns:
                continue
            ref_counts = self.reference_stats["cat_stats"][col]["value_counts"]
            new_vc     = X_new[col].value_counts(normalize=True).to_dict()

            all_cats = set(ref_counts) | set(new_vc)
            ref_arr  = np.array([ref_counts.get(c, 1e-6) for c in all_cats])
            new_arr  = np.array([new_vc.get(c, 1e-6)    for c in all_cats])

            # Normalise
            ref_arr = ref_arr / ref_arr.sum()
            new_arr = new_arr / new_arr.sum()

            chi2, chi_p = stats.chisquare(new_arr * len(X_new),
                                          f_exp=ref_arr * len(X_new))
            col_drifted = chi_p < self.ks_threshold

            feature_reports[col] = {
                "chi2_statistic": round(float(chi2), 4),
                "chi2_p_value":   round(float(chi_p), 4),
                "drifted":        col_drifted,
                "type":           "categorical",
            }
            if col_drifted:
                drifted.append(col)

        # Global verdict
        n_features = max(len(feature_reports), 1)
        drift_ratio = len(drifted) / n_features
        drift_score = round(float(drift_ratio), 4)

        if   drift_ratio == 0:     severity = "none"
        elif drift_ratio < 0.15:   severity = "low"
        elif drift_ratio < DRIFT_FEAT_RATIO: severity = "moderate"
        else:                      severity = "severe"

        drift_detected = severity in ("moderate", "severe")

        rec = {
            "none":     "✅ No significant drift detected. Current model remains valid.",
            "low":      "🟡 Minor drift detected. Monitor closely; retraining not yet required.",
            "moderate": "🟠 Moderate drift detected. Consider retraining the model.",
            "severe":   "🔴 Severe drift detected. Retraining is strongly recommended.",
        }[severity]

        self.last_report = {
            "drift_detected":   drift_detected,
            "drift_score":      drift_score,
            "drifted_features": drifted,
            "n_drifted":        len(drifted),
            "n_total":          n_features,
            "severity":         severity,
            "feature_reports":  feature_reports,
            "recommendation":   rec,
            "n_reference":      self.reference_stats.get("n_reference", 0),
            "n_new":            len(X_new),
        }

        logger.info(
            f"Drift detection: severity={severity} | "
            f"{len(drifted)}/{n_features} features drifted | score={drift_score}"
        )
        return self.last_report

    # ── Internal ───────────────────────────────────────────────────

    def _psi(self, reference: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """
        Population Stability Index.
        PSI < 0.10 → stable; 0.10–0.20 → some shift; > 0.20 → large shift.
        """
        breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)  # Remove duplicates from constant cols

        ref_pcts  = np.histogram(reference, bins=breakpoints)[0] / len(reference)
        act_pcts  = np.histogram(actual,    bins=breakpoints)[0] / len(actual)

        # Clip to avoid log(0)
        ref_pcts = np.clip(ref_pcts, 1e-6, 1)
        act_pcts = np.clip(act_pcts, 1e-6, 1)

        psi = np.sum((act_pcts - ref_pcts) * np.log(act_pcts / ref_pcts))
        return float(psi)

    def get_last_report(self) -> dict:
        return self.last_report

    def is_fitted(self) -> bool:
        return bool(self.reference_stats)
