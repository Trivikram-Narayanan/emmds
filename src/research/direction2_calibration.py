"""
EMMDS Direction 2: Trust Score Calibration
==========================================
Research Question:
  "Is the EMMDS Trust Score itself well-calibrated — i.e., when
   trust = 0.8, does actual reliability equal ~0.8?"

Methodology:
  1. Collect trust scores and actual reliability (1 - deployment_risk)
     across all model-dataset pairs
  2. Build reliability diagram: bin trust scores, plot empirical reliability
  3. Measure calibration error (ECE - Expected Calibration Error)
  4. Apply isotonic regression to calibrate the trust score
  5. Compare: raw trust ECE vs calibrated trust ECE
  6. Show calibration curve for each trust tier

This parallels the probability calibration literature (Guo et al. 2017)
but applied to AutoML trust scores rather than neural network probabilities.
Nobody has done this before for trust scores.
"""

import sys, warnings, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.model_selection import KFold
from sklearn.base import clone

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

OUT     = Path("outputs/research/direction2")
OUT_D1  = Path("outputs/research/direction1")
OUT_D1.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# RELIABILITY DIAGRAM
# ══════════════════════════════════════════════════════════════════════

def build_reliability_diagram(trust_scores, actual_reliability,
                               n_bins=8, label="Trust Score"):
    """
    Bin trust scores, compute mean actual reliability per bin.
    A perfectly calibrated score lies on the diagonal.

    Returns:
        bins_df: DataFrame with bin_center, mean_trust, mean_reliability,
                 count, calibration_gap
    """
    bins    = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    mean_trusts = []
    mean_rels   = []
    counts      = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (trust_scores >= lo) & (trust_scores < hi)
        if i == n_bins - 1:
            mask = (trust_scores >= lo) & (trust_scores <= hi)
        n = mask.sum()
        if n == 0:
            continue
        bin_centers.append(round((lo + hi) / 2, 4))
        mean_trusts.append(round(float(trust_scores[mask].mean()), 4))
        mean_rels.append(round(float(actual_reliability[mask].mean()), 4))
        counts.append(int(n))

    df = pd.DataFrame({
        'bin_center':          bin_centers,
        'mean_trust_in_bin':   mean_trusts,
        'mean_actual_reliability': mean_rels,
        'count':               counts,
        'calibration_gap':     [round(t - r, 4)
                                for t, r in zip(mean_trusts, mean_rels)],
    })
    return df


# ══════════════════════════════════════════════════════════════════════
# EXPECTED CALIBRATION ERROR
# ══════════════════════════════════════════════════════════════════════

def expected_calibration_error(trust_scores, actual_reliability,
                                n_bins=8):
    """
    ECE = Σ (n_b / n) × |avg_trust_b - avg_reliability_b|

    Lower ECE = better calibrated trust score.
    ECE = 0 is perfect calibration.
    """
    n       = len(trust_scores)
    bins    = np.linspace(0, 1, n_bins + 1)
    ece     = 0.0
    details = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (trust_scores >= lo) & (trust_scores < hi)
        if i == n_bins - 1:
            mask = (trust_scores >= lo) & (trust_scores <= hi)
        n_b = mask.sum()
        if n_b == 0:
            continue
        avg_trust = float(trust_scores[mask].mean())
        avg_rel   = float(actual_reliability[mask].mean())
        gap       = abs(avg_trust - avg_rel)
        ece      += (n_b / n) * gap
        details.append({
            'bin':        i, 'n_b': int(n_b),
            'avg_trust':  round(avg_trust, 4),
            'avg_rel':    round(avg_rel, 4),
            'gap':        round(gap, 4),
            'contribution': round((n_b / n) * gap, 6),
        })

    return round(float(ece), 6), details


# ══════════════════════════════════════════════════════════════════════
# TRUST SCORE CALIBRATION (ISOTONIC REGRESSION)
# ══════════════════════════════════════════════════════════════════════

def calibrate_trust_score(trust_scores, actual_reliability):
    """
    Fit isotonic regression to map raw trust → calibrated trust.
    Uses 5-fold CV to evaluate calibration improvement.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    raw_eces = []
    cal_eces = []

    for tr_idx, te_idx in kf.split(trust_scores):
        ts_tr = trust_scores[tr_idx]
        rel_tr = actual_reliability[tr_idx]
        ts_te  = trust_scores[te_idx]
        rel_te = actual_reliability[te_idx]

        # Fit isotonic regression on training fold
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(ts_tr, rel_tr)

        # Evaluate raw ECE on test fold
        raw_ece, _ = expected_calibration_error(ts_te, rel_te)

        # Evaluate calibrated ECE on test fold
        ts_cal = ir.predict(ts_te)
        cal_ece, _ = expected_calibration_error(
            np.array(ts_cal), rel_te)

        raw_eces.append(raw_ece)
        cal_eces.append(cal_ece)

    # Fit final calibrator on all data
    ir_final = IsotonicRegression(out_of_bounds='clip')
    ir_final.fit(trust_scores, actual_reliability)

    return ir_final, {
        'raw_ece_mean':  round(float(np.mean(raw_eces)), 6),
        'raw_ece_std':   round(float(np.std(raw_eces)),  6),
        'cal_ece_mean':  round(float(np.mean(cal_eces)), 6),
        'cal_ece_std':   round(float(np.std(cal_eces)),  6),
        'improvement':   round(float(np.mean(raw_eces) - np.mean(cal_eces)), 6),
        'improvement_pct': round(float(
            (np.mean(raw_eces) - np.mean(cal_eces)) /
            max(np.mean(raw_eces), 1e-8) * 100), 2),
    }


# ══════════════════════════════════════════════════════════════════════
# PER-TIER ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def analyse_trust_tiers(df):
    """
    For each trust tier (Very Low / Low / Moderate / High / Very High),
    measure actual reliability and deployment risk statistics.
    Validates whether the tier labels are empirically meaningful.
    """
    tiers = [
        ('Very Low',  0.00, 0.40),
        ('Low',       0.40, 0.55),
        ('Moderate',  0.55, 0.70),
        ('High',      0.70, 0.85),
        ('Very High', 0.85, 1.00),
    ]

    rows = []
    for tier_name, lo, hi in tiers:
        mask = (df['trust_score'] >= lo) & (df['trust_score'] < hi)
        if tier_name == 'Very High':
            mask = (df['trust_score'] >= lo) & (df['trust_score'] <= hi)
        sub = df[mask]
        if len(sub) == 0:
            continue

        rows.append({
            'tier':             tier_name,
            'trust_range':      f"[{lo}, {hi})",
            'n':                int(len(sub)),
            'mean_trust':       round(float(sub['trust_score'].mean()), 4),
            'mean_reliability': round(float(sub['actual_reliability'].mean()), 4),
            'mean_risk':        round(float(sub['deployment_risk'].mean()), 4),
            'mean_accuracy':    round(float(sub['test_acc'].mean()), 4),
            'calibration_gap':  round(float(
                sub['trust_score'].mean() - sub['actual_reliability'].mean()), 4),
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# OVERCONFIDENCE ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def analyse_overconfidence(df):
    """
    Where is the trust score systematically overconfident (trust > actual)?
    Mirrors Guo et al. (2017) analysis for neural networks, applied to AutoML.
    """
    df2 = df.copy()
    df2['overconfident'] = df2['trust_score'] > df2['actual_reliability']
    df2['confidence_gap'] = df2['trust_score'] - df2['actual_reliability']

    overall_overconf = float(df2['overconfident'].mean())
    mean_gap_overconf = float(df2[df2['overconfident']]['confidence_gap'].mean())
    mean_gap_underconf = float(df2[~df2['overconfident']]['confidence_gap'].mean())

    # Which datasets are most overconfident?
    per_ds = df2.groupby('dataset').agg(
        mean_gap=('confidence_gap', 'mean'),
        pct_overconf=('overconfident', 'mean'),
    ).round(4).sort_values('mean_gap', ascending=False)

    # Which models are most overconfident?
    per_model = df2.groupby('model').agg(
        mean_gap=('confidence_gap', 'mean'),
        pct_overconf=('overconfident', 'mean'),
    ).round(4).sort_values('mean_gap', ascending=False)

    return {
        'overall_overconfident_pct': round(overall_overconf * 100, 2),
        'mean_gap_when_overconfident': round(mean_gap_overconf, 4),
        'mean_gap_when_underconfident': round(mean_gap_underconf, 4),
        'most_overconfident_datasets': per_ds.head(3).to_dict('index'),
        'most_overconfident_models': per_model.head(3).to_dict('index'),
    }


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def run_direction2(raw_results_df=None):
    """
    If raw_results_df provided, use it. Otherwise load from Direction 1 output.
    """
    print("=" * 65)
    print("  DIRECTION 2: TRUST SCORE CALIBRATION")
    print("  Research: Is the EMMDS Trust Score well-calibrated?")
    print("            Can we calibrate it like probability estimates?")
    print("=" * 65)

    # Load data
    if raw_results_df is None:
        # Try direction1 measurements first, then original experiments
        d1_path = OUT_D1 / "all_model_measurements.csv"
        orig_path = Path("outputs/research/raw_results.csv")
        if d1_path.exists():
            df = pd.read_csv(d1_path)
            print(f"  Loaded {len(df)} rows from Direction 1 measurements")
        elif orig_path.exists():
            df = pd.read_csv(orig_path)
            print(f"  Loaded {len(df)} rows from original experiments")
        else:
            print("  ERROR: No data found. Run Direction 1 or experiments.py first.")
            return {}
    else:
        df = raw_results_df.copy()

    # Compute deployment risk if not present
    if 'deployment_risk' not in df.columns:
        df['overfitting_ratio'] = df['gen_gap'] / (df['test_acc'] + 1e-8)
        df['calibration_error'] = 1.0 - df['cal_score']
        df['deployment_risk'] = (
            0.40 * np.clip(df['overfitting_ratio'], 0, 1) +
            0.30 * df['calibration_error'] +
            0.30 * df['cv_std']
        )

    # Actual reliability = 1 - normalised deployment risk
    risk_max = df['deployment_risk'].max()
    risk_min = df['deployment_risk'].min()
    df['actual_reliability'] = 1.0 - (
        (df['deployment_risk'] - risk_min) /
        (risk_max - risk_min + 1e-8)
    )

    trust_scores       = df['trust_score'].values
    actual_reliability = df['actual_reliability'].values

    print(f"\n  Dataset: {len(df)} model-dataset pairs across "
          f"{df['dataset'].nunique()} datasets")

    # Step 1: Raw reliability diagram
    print("\n  Step 1/5: Building reliability diagram...")
    rel_diag = build_reliability_diagram(trust_scores, actual_reliability)
    print(f"  Reliability diagram ({len(rel_diag)} bins):")
    print(f"  {'Bin':6s} {'Mean Trust':12s} {'Mean Actual':12s} {'Gap':8s} {'N':5s}")
    print(f"  {'-'*50}")
    for _, r in rel_diag.iterrows():
        gap_sign = '↑ over' if r['calibration_gap'] > 0.02 else (
                   '↓ under' if r['calibration_gap'] < -0.02 else 'OK')
        print(f"  {r['bin_center']:.2f}   {r['mean_trust_in_bin']:.4f}       "
              f"{r['mean_actual_reliability']:.4f}       {r['calibration_gap']:+.4f}  "
              f"{r['count']:4d}  {gap_sign}")

    # Step 2: ECE computation
    print("\n  Step 2/5: Computing Expected Calibration Error (ECE)...")
    raw_ece, ece_details = expected_calibration_error(
        trust_scores, actual_reliability)
    print(f"  Raw Trust Score ECE: {raw_ece:.6f}")

    # Compare to accuracy ECE (accuracy is NOT a trust score, so ECE
    # interpretation differs, but useful for comparison)
    acc_ece, _ = expected_calibration_error(
        df['test_acc'].values, actual_reliability)
    print(f"  Accuracy ECE:        {acc_ece:.6f}")

    # Step 3: Calibration
    print("\n  Step 3/5: Calibrating trust score via isotonic regression...")
    ir_final, cal_results = calibrate_trust_score(
        trust_scores, actual_reliability)

    print(f"  Raw trust ECE  (5-fold CV): "
          f"{cal_results['raw_ece_mean']:.6f} ± {cal_results['raw_ece_std']:.6f}")
    print(f"  Calibrated ECE (5-fold CV): "
          f"{cal_results['cal_ece_mean']:.6f} ± {cal_results['cal_ece_std']:.6f}")
    print(f"  ECE improvement: {cal_results['improvement']:.6f} "
          f"({cal_results['improvement_pct']:.1f}%)")

    # Calibrated reliability diagram
    ts_calibrated = ir_final.predict(trust_scores)
    rel_diag_cal  = build_reliability_diagram(
        np.array(ts_calibrated), actual_reliability,
        label="Calibrated Trust")

    # Step 4: Tier analysis
    print("\n  Step 4/5: Per-tier reliability analysis...")
    tier_df = analyse_trust_tiers(df)
    print(f"\n  {'Tier':10s}  {'N':5s}  {'Mean Trust':12s}  "
          f"{'Actual Rel':12s}  {'Gap':8s}")
    print(f"  {'-'*60}")
    for _, r in tier_df.iterrows():
        print(f"  {r['tier']:10s}  {r['n']:5d}  {r['mean_trust']:.4f}       "
              f"{r['mean_reliability']:.4f}       {r['calibration_gap']:+.4f}")

    # Step 5: Overconfidence analysis
    print("\n  Step 5/5: Overconfidence analysis...")
    overconf = analyse_overconfidence(df)
    print(f"  Overall overconfident: {overconf['overall_overconfident_pct']:.1f}%")
    print(f"  Mean gap (overconf):   {overconf['mean_gap_when_overconfident']:.4f}")
    print(f"  Mean gap (underconf):  {overconf['mean_gap_when_underconfident']:.4f}")

    # Save everything
    rel_diag.to_csv(OUT / "reliability_diagram_raw.csv", index=False)
    rel_diag_cal.to_csv(OUT / "reliability_diagram_calibrated.csv", index=False)
    tier_df.to_csv(OUT / "tier_analysis.csv", index=False)
    df[['dataset','model','trust_score','actual_reliability',
        'deployment_risk','test_acc']].to_csv(
        OUT / "trust_reliability_pairs.csv", index=False)

    def _j(o):
        if isinstance(o, (np.bool_,)):    return bool(o)
        if isinstance(o, (np.integer,)):  return int(o)
        if isinstance(o, (np.floating,)):
            return None if (np.isnan(o) or np.isinf(o)) else float(o)
        if isinstance(o, np.ndarray):     return o.tolist()
        return str(o)

    results = {
        'n_pairs':          int(len(df)),
        'n_datasets':       int(df['dataset'].nunique()),
        'raw_ece':          raw_ece,
        'accuracy_ece':     acc_ece,
        'calibration': {
            **cal_results,
            'interpretation': (
                f"Isotonic calibration reduces ECE by "
                f"{cal_results['improvement_pct']:.1f}%, from "
                f"{cal_results['raw_ece_mean']:.4f} to "
                f"{cal_results['cal_ece_mean']:.4f}"
            ),
        },
        'reliability_diagram': rel_diag.to_dict('records'),
        'reliability_diagram_calibrated': rel_diag_cal.to_dict('records'),
        'tier_analysis':    tier_df.to_dict('records'),
        'overconfidence':   overconf,
        'key_finding': (
            f"The raw EMMDS Trust Score has ECE={raw_ece:.4f}. "
            f"Isotonic calibration reduces this by "
            f"{cal_results['improvement_pct']:.1f}% (ECE="
            f"{cal_results['cal_ece_mean']:.4f}). "
            f"{overconf['overall_overconfident_pct']:.1f}% of "
            f"model-dataset pairs are overconfident (trust > actual reliability)."
        ),
    }

    with open(OUT / "direction2_results.json", "w") as f:
        json.dump(results, f, indent=2, default=_j)

    print(f"\n  Results saved → {OUT}/")
    print(f"\n  KEY FINDING:")
    print(f"  {results['key_finding']}")

    return results


if __name__ == "__main__":
    run_direction2()
