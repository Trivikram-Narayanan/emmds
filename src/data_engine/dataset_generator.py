"""
EMMDS Expanded Dataset Generator
==================================
Generates 50+ synthetic datasets that systematically cover the
property space of real-world classification problems.

When OpenML is unavailable (no network), this provides a rigorous
controlled experiment collection with known ground-truth properties.

Dataset categories:
  1. Imbalance series      (8 datasets) — ratio 1:1 to 20:1
  2. Noise series          (7 datasets) — 0% to 25% label flip
  3. Dimensionality series (7 datasets) — 5 to 200 features
  4. Sample size series    (7 datasets) — 100 to 5000 samples
  5. Class count series    (5 datasets) — 2 to 8 classes
  6. Mixed hard cases      (8 datasets) — combined challenges
  7. Real-mimicking        (10 datasets)— properties from real domains
"""

import numpy as np
import pandas as pd
from sklearn.datasets import (
    make_classification, make_circles, make_moons,
    load_breast_cancer, load_iris, load_wine, load_digits
)
from typing import List, Tuple

RANDOM_STATE = 42


def _make_df(X, y, prefix="f"):
    df = pd.DataFrame(X, columns=[f"{prefix}{i}" for i in range(X.shape[1])])
    df["target"] = y
    return df


def build_full_dataset_collection() -> List[Tuple[pd.DataFrame, str, str]]:
    """
    Returns list of (dataframe, target_col, name) for all datasets.
    """
    datasets = []

    # ── Real sklearn built-ins ────────────────────────────────────────
    for name, loader in [
        ("breast_cancer", load_breast_cancer),
        ("wine",          load_wine),
        ("iris",          load_iris),
        ("digits",        load_digits),
    ]:
        d = loader(as_frame=True)
        df = d.frame.copy(); df["target"] = d.target
        datasets.append((df, "target", f"real_{name}"))

    # ── Series 1: Imbalance (8 datasets) ─────────────────────────────
    for ratio_label, weights in [
        ("1_1",  None),
        ("2_1",  [0.667, 0.333]),
        ("3_1",  [0.750, 0.250]),
        ("5_1",  [0.833, 0.167]),
        ("7_1",  [0.875, 0.125]),
        ("10_1", [0.909, 0.091]),
        ("15_1", [0.938, 0.063]),
        ("20_1", [0.952, 0.048]),
    ]:
        kw = dict(n_samples=800, n_features=20, n_informative=12,
                  n_redundant=4, n_classes=2, class_sep=1.0,
                  flip_y=0.02, random_state=RANDOM_STATE,
                  n_clusters_per_class=1)
        if weights:
            kw["weights"] = weights
        X, y = make_classification(**kw)
        datasets.append((_make_df(X, y), "target", f"imbal_{ratio_label}"))

    # ── Series 2: Noise (7 datasets) ──────────────────────────────────
    for noise_pct in [0, 2, 5, 10, 15, 20, 25]:
        X, y = make_classification(
            n_samples=800, n_features=20, n_informative=12,
            n_redundant=4, n_classes=2, class_sep=1.0,
            flip_y=noise_pct/100, random_state=RANDOM_STATE,
            n_clusters_per_class=1
        )
        datasets.append((_make_df(X, y), "target", f"noise_{noise_pct}pct"))

    # ── Series 3: Dimensionality (7 datasets) ─────────────────────────
    for n_feat, n_inf in [
        (5,  3), (10, 6), (20, 12), (40, 20),
        (60, 25), (100, 30), (150, 40)
    ]:
        n_red = min(4, n_feat - n_inf - 2)
        X, y = make_classification(
            n_samples=600, n_features=n_feat, n_informative=n_inf,
            n_redundant=max(1, min(n_red, n_feat-n_inf-1)), n_classes=2, class_sep=1.0,
            flip_y=0.02, random_state=RANDOM_STATE,
            n_clusters_per_class=1
        )
        datasets.append((_make_df(X, y), "target", f"dim_{n_feat}f"))

    # ── Series 4: Sample size (7 datasets) ───────────────────────────
    for n_samples in [100, 200, 400, 800, 1500, 3000, 5000]:
        X, y = make_classification(
            n_samples=n_samples, n_features=20, n_informative=12,
            n_redundant=4, n_classes=2, class_sep=1.0,
            flip_y=0.02, random_state=RANDOM_STATE,
            n_clusters_per_class=1
        )
        datasets.append((_make_df(X, y), "target", f"n_{n_samples}"))

    # ── Series 5: Class count (5 datasets) ───────────────────────────
    for nc in [2, 3, 4, 6, 8]:
        n_feat = max(20, nc * 4)
        n_inf  = min(nc * 3, n_feat - 3)
        X, y = make_classification(
            n_samples=800, n_features=n_feat, n_informative=n_inf,
            n_redundant=2, n_classes=nc, class_sep=1.0,
            flip_y=0.02, random_state=RANDOM_STATE,
            n_clusters_per_class=1
        )
        datasets.append((_make_df(X, y), "target", f"classes_{nc}"))

    # ── Series 6: Mixed hard cases (8 datasets) ───────────────────────
    hard_cases = [
        ("hard_noisy_imbal",    dict(n_samples=600, n_features=20, n_informative=10,
                                     n_redundant=5, flip_y=0.15, weights=[0.80,0.20],
                                     class_sep=0.7, n_clusters_per_class=1)),
        ("hard_highdim_small",  dict(n_samples=200, n_features=60, n_informative=20,
                                     n_redundant=15, flip_y=0.03, class_sep=1.0,
                                     n_clusters_per_class=1)),
        ("hard_many_classes",   dict(n_samples=1000, n_features=30, n_informative=20,
                                     n_redundant=5, n_classes=6, flip_y=0.08,
                                     class_sep=0.8, n_clusters_per_class=1)),
        ("hard_low_signal",     dict(n_samples=800, n_features=30, n_informative=3,
                                     n_redundant=20, flip_y=0.10, class_sep=0.5,
                                     n_clusters_per_class=1)),
        ("hard_tiny_minority",  dict(n_samples=1000, n_features=20, n_informative=12,
                                     n_redundant=4, weights=[0.97,0.03], flip_y=0.02,
                                     class_sep=1.0, n_clusters_per_class=1)),
        ("hard_high_noise_mc",  dict(n_samples=600, n_features=20, n_informative=12,
                                     n_redundant=4, n_classes=4, flip_y=0.20,
                                     class_sep=0.6, n_clusters_per_class=1)),
        ("hard_correlated",     dict(n_samples=700, n_features=20, n_informative=8,
                                     n_redundant=12, flip_y=0.05, class_sep=1.0,
                                     n_clusters_per_class=1)),
        ("hard_small_noisy",    dict(n_samples=150, n_features=15, n_informative=6,
                                     n_redundant=5, flip_y=0.12, class_sep=0.8,
                                     n_clusters_per_class=1)),
    ]
    for name, kw in hard_cases:
        kw["random_state"] = RANDOM_STATE
        X, y = make_classification(**kw)
        datasets.append((_make_df(X, y), "target", name))

    # ── Series 7: Real-domain mimicking (10 datasets) ─────────────────
    # Properties match documented characteristics of real datasets
    domain_specs = [
        # Medical: moderate imbalance, meaningful features
        ("medical_cardiac",  dict(n_samples=400,  n_features=13, n_informative=8,
                                   n_redundant=3, weights=[0.55,0.45], flip_y=0.03,
                                   class_sep=1.2, n_clusters_per_class=1)),
        # Credit scoring: imbalance, many features
        ("financial_credit", dict(n_samples=1000, n_features=25, n_informative=15,
                                   n_redundant=6, weights=[0.70,0.30], flip_y=0.02,
                                   class_sep=0.9, n_clusters_per_class=1)),
        # Spam detection: many features, good separation
        ("text_spam",        dict(n_samples=1500, n_features=57, n_informative=30,
                                   n_redundant=15, weights=[0.61,0.39], flip_y=0.01,
                                   class_sep=1.5, n_clusters_per_class=1)),
        # Sensor/IoT: high noise, many redundant features
        ("iot_sensor",       dict(n_samples=2000, n_features=40, n_informative=10,
                                   n_redundant=25, flip_y=0.08, class_sep=0.8,
                                   n_clusters_per_class=1)),
        # Genomics: high dim, small n
        ("genomics_small",   dict(n_samples=200,  n_features=80, n_informative=20,
                                   n_redundant=30, flip_y=0.05, class_sep=1.0,
                                   n_clusters_per_class=1)),
        # Customer churn: imbalanced, noisy
        ("churn_prediction", dict(n_samples=3000, n_features=18, n_informative=10,
                                   n_redundant=5, weights=[0.85,0.15], flip_y=0.04,
                                   class_sep=0.9, n_clusters_per_class=1)),
        # Fault detection: severe imbalance
        ("fault_detection",  dict(n_samples=1000, n_features=30, n_informative=12,
                                   n_redundant=8, weights=[0.95,0.05], flip_y=0.01,
                                   class_sep=1.3, n_clusters_per_class=1)),
        # Image classification (features): many classes
        ("image_features",   dict(n_samples=2000, n_features=64, n_informative=40,
                                   n_redundant=15, n_classes=5, flip_y=0.03,
                                   class_sep=1.0, n_clusters_per_class=1)),
        # Network intrusion: very imbalanced
        ("network_intrusion", dict(n_samples=2000, n_features=35, n_informative=20,
                                    n_redundant=8, weights=[0.90,0.10], flip_y=0.01,
                                    class_sep=1.2, n_clusters_per_class=1)),
        # Social media: noisy labels, many features
        ("social_behaviour",  dict(n_samples=1200, n_features=45, n_informative=18,
                                    n_redundant=15, flip_y=0.12, class_sep=0.7,
                                    n_clusters_per_class=1)),
    ]
    for name, kw in domain_specs:
        kw["random_state"] = RANDOM_STATE
        X, y = make_classification(**kw)
        datasets.append((_make_df(X, y), "target", name))

    return datasets


def get_collection_summary(
    datasets: List[Tuple[pd.DataFrame, str, str]]
) -> pd.DataFrame:
    rows = []
    for df, target, name in datasets:
        y   = df[target]
        vc  = y.value_counts()
        ir  = round(float(vc.iloc[0]/vc.iloc[-1]), 2) if len(vc)>1 else 1.0
        rows.append({
            "name":           name,
            "n_samples":      len(df),
            "n_features":     df.shape[1]-1,
            "n_classes":      int(y.nunique()),
            "imbalance":      ir,
            "source":         "real" if name.startswith("real_") else "synthetic",
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    datasets = build_full_dataset_collection()
    summary  = get_collection_summary(datasets)
    print(f"Total datasets: {len(datasets)}")
    print(f"\nBy source:")
    print(summary["source"].value_counts().to_string())
    print(f"\nProperty ranges:")
    print(f"  n_samples:  {summary['n_samples'].min()} — {summary['n_samples'].max()}")
    print(f"  n_features: {summary['n_features'].min()} — {summary['n_features'].max()}")
    print(f"  n_classes:  {summary['n_classes'].min()} — {summary['n_classes'].max()}")
    print(f"  imbalance:  {summary['imbalance'].min()} — {summary['imbalance'].max()}")
