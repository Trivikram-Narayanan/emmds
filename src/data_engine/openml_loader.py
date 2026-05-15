"""
EMMDS Real Dataset Loader
==========================
Fetches real classification datasets from OpenML CC18 benchmark suite
plus sklearn built-ins. Caches everything locally so experiments are
reproducible without repeated network calls.

Usage
-----
    from src.data_engine.openml_loader import load_real_datasets
    datasets = load_real_datasets(n=50)   # list of (X, y, name) tuples
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "openml_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Curated OpenML dataset IDs (CC18 + well-known UCI mirrors)
# Criteria: 100–15000 rows, ≤200 features, ≤20 classes, binary preferred
# ─────────────────────────────────────────────────────────────
OPENML_IDS = [
    # id    name
    (31,    "credit-g"),
    (37,    "diabetes"),
    (40,    "sonar"),
    (44,    "spambase"),
    (54,    "vehicle"),
    (15,    "breast-w"),
    (29,    "credit-approval"),
    (38,    "sick"),
    (50,    "tic-tac-toe"),
    (181,   "yeast"),
    (182,   "satimage"),
    (188,   "eucalyptus"),
    (307,   "vowel"),
    (333,   "monks-1"),
    (334,   "monks-2"),
    (335,   "monks-3"),
    (458,   "authorship"),
    (469,   "analcatdata-dmft"),
    (1462,  "banknote"),
    (1464,  "blood-transfusion"),
    (1480,  "ilpd"),
    (1487,  "ozone-level"),
    (1488,  "wall-robot"),
    (1494,  "blood-transfusion2"),
    (1510,  "wdbc"),
    (3,     "kr-vs-kp"),
    (11,    "balance-scale"),
    (18,    "mfeat-morphological"),
    (22,    "mfeat-zernike"),
    (36,    "segment"),
    (46,    "splice"),
    (1068,  "pc1"),
    (1475,  "first-order-theorem"),
    (40966, "MiceProtein"),
    (40975, "car"),
    (40982, "steel-plates-fault"),
    (40983, "wilt"),
    (40984, "segment2"),
    (23,    "contraceptive"),
    (1018,  "ipums-small"),
]


# ─────────────────────────────────────────────────────────────
# sklearn built-in datasets (always available, no network needed)
# ─────────────────────────────────────────────────────────────

def _sklearn_builtins() -> List[Tuple[np.ndarray, np.ndarray, str]]:
    from sklearn import datasets as skds
    out = []

    ds = skds.load_iris()
    out.append((ds.data, ds.target, "iris"))

    ds = skds.load_breast_cancer()
    out.append((ds.data, ds.target, "breast_cancer"))

    ds = skds.load_wine()
    out.append((ds.data, ds.target, "wine"))

    ds = skds.load_digits()
    out.append((ds.data, ds.target, "digits"))

    # Make a binary version of digits (odd vs even)
    out.append((ds.data, (ds.target % 2).astype(int), "digits_parity"))

    return out


# ─────────────────────────────────────────────────────────────
# OpenML fetcher with disk cache
# ─────────────────────────────────────────────────────────────

def _fetch_openml(dataset_id: int, name: str,
                  max_rows: int = 15000,
                  max_features: int = 200,
                  max_classes: int = 20
                  ) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    cache_file = CACHE_DIR / f"{dataset_id}_{name}.npz"

    # ── serve from cache ──────────────────────────────────────
    if cache_file.exists():
        try:
            d = np.load(cache_file, allow_pickle=True)
            return d["X"], d["y"], name
        except Exception:
            cache_file.unlink(missing_ok=True)

    # ── download ──────────────────────────────────────────────
    try:
        import openml
        openml.config.apikey      = ""       # public API, no key needed
        openml.config.cache_directory = str(CACHE_DIR / "openml_lib")

        ds = openml.datasets.get_dataset(
            dataset_id,
            download_data=True,
            download_qualities=False,
            download_features_meta_data=False,
        )
        X_df, y_series, _, _ = ds.get_data(target=ds.default_target_attribute,
                                             dataset_format="dataframe")

        if X_df is None or y_series is None:
            return None

        # size / class filters
        n, p = X_df.shape
        if n > max_rows or n < 80:
            return None
        if p > max_features:
            return None

        n_classes = y_series.nunique()
        if n_classes < 2 or n_classes > max_classes:
            return None

        # encode categoricals + drop high-cardinality
        from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
        cat_cols = X_df.select_dtypes(include=["object", "category", "bool"]).columns
        for col in cat_cols:
            if X_df[col].nunique() > 50:
                X_df = X_df.drop(columns=[col])
            else:
                X_df[col] = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                ).fit_transform(X_df[[col]])

        X_df = X_df.select_dtypes(include=[np.number]).fillna(X_df.median(numeric_only=True))
        if X_df.shape[1] == 0:
            return None

        X = X_df.values.astype(float)
        y = LabelEncoder().fit_transform(y_series.astype(str))

        np.savez_compressed(cache_file, X=X, y=y)
        print(f"  ✅  {name:<30} n={n:>5}  p={p:>3}  classes={n_classes}")
        return X, y, name

    except Exception as e:
        print(f"  ⚠️   {name:<30} skipped ({type(e).__name__})")
        return None


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def load_real_datasets(
    n: int = 50,
    max_rows: int = 15000,
    max_features: int = 200,
    max_classes: int = 20,
    seed: int = 42,
    verbose: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """
    Return up to `n` real datasets as (X, y, name) tuples.

    Draws from:
      1. sklearn built-in datasets (always available)
      2. OpenML CC18 curated list (downloaded + cached)

    Args:
        n: max number of datasets to return
        max_rows: skip datasets with more rows
        max_features: skip datasets with more features
        max_classes: skip datasets with more classes
        seed: for shuffling the order
        verbose: print progress

    Returns:
        List of (X: np.ndarray, y: np.ndarray, name: str)
    """
    if verbose:
        print(f"Loading real datasets (target: {n}) ...")

    datasets = []

    # 1. sklearn built-ins (guaranteed)
    for X, y, name in _sklearn_builtins():
        datasets.append((X, y, name))
        if verbose:
            n_cls = len(np.unique(y))
            print(f"  ✅  {name:<30} n={len(X):>5}  p={X.shape[1]:>3}  classes={n_cls}")

    # 2. OpenML
    for did, name in OPENML_IDS:
        if len(datasets) >= n:
            break
        result = _fetch_openml(did, name, max_rows, max_features, max_classes)
        if result is not None:
            # avoid duplicates (some OpenML IDs alias the same data as builtins)
            existing_names = {d[2] for d in datasets}
            clean_name = name
            if clean_name in existing_names:
                clean_name = f"{name}_{did}"
            datasets.append((result[0], result[1], clean_name))

    # shuffle deterministically so meta-train/test split is random but reproducible
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(datasets))
    datasets = [datasets[i] for i in idx]

    if verbose:
        print(f"\nLoaded {len(datasets)} real datasets.")

    return datasets[:n]


def get_dataset_stats(datasets: List[Tuple]) -> pd.DataFrame:
    """Return a summary DataFrame of loaded datasets."""
    rows = []
    for X, y, name in datasets:
        classes, counts = np.unique(y, return_counts=True)
        rows.append({
            "name":      name,
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_classes": len(classes),
            "imbalance": round(float(counts.max() / counts.min()), 2),
            "missing":   round(float(np.isnan(X).mean()), 4),
        })
    return pd.DataFrame(rows)
