"""
EMMDS OpenML Dataset Loader
============================
Downloads real-world classification datasets from OpenML.
Caches locally so experiments can be reproduced without network.

When network is unavailable, falls back to sklearn built-ins
and locally cached CSV files.

Targets 80+ diverse datasets covering:
  - Medical: heart disease, diabetes, hepatitis, breast tissue
  - Financial: credit approval, adult income, bank marketing
  - Biological: iris, wine, mushroom, splice
  - Social: car evaluation, nursery, tic-tac-toe
  - Engineering: ionosphere, sonar, glass
  - General: covertype (subset), KDD cup (subset)
"""

import sys
import json
import hashlib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple

warnings.filterwarnings('ignore')

CACHE_DIR = Path("data/sample_datasets")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Curated OpenML dataset IDs ────────────────────────────────────────
# Selected for: diversity, citation count, varied properties
# Format: {openml_id: (name, expected_task, notes)}
OPENML_DATASET_IDS = {
    # Medical / health
    31:   ("credit_g",        "classification", "German credit risk"),
    37:   ("diabetes",        "classification", "Pima Indians diabetes"),
    53:   ("heart_disease",   "classification", "Heart disease Cleveland"),
    40:   ("sonar",           "classification", "Sonar rock vs mine"),
    188:  ("eucalyptus",      "classification", "Eucalyptus species"),
    1067: ("kc1",             "classification", "Software defect prediction"),
    1068: ("pc1",             "classification", "NASA software defects"),
    # Biological
    61:   ("iris",            "classification", "Iris flowers (3-class)"),
    187:  ("wine_quality_red","classification", "Wine quality red"),
    40691:("wine",            "classification", "Wine recognition"),
    # Financial / social
    1590: ("adult",           "classification", "Adult income >50K"),
    1461: ("bank_marketing",  "classification", "Bank telemarketing"),
    29:   ("credit_approval", "classification", "Credit card approval"),
    # Engineering / physics
    59:   ("ionosphere",      "classification", "Radar ionosphere"),
    1515: ("breast_tissue",   "classification", "Breast tissue type"),
    1510: ("wdbc",            "classification", "Wisconsin breast cancer"),
    # Classic benchmarks
    3:    ("kr_vs_kp",        "classification", "Chess end-game"),
    12:   ("mfeat_factors",   "classification", "Handwritten digits"),
    14:   ("mfeat_fourier",   "classification", "Digit features fourier"),
    16:   ("mfeat_karhunen",  "classification", "Digit features karhunen"),
    18:   ("mfeat_morphological","classification","Digit morphological"),
    22:   ("mfeat_pixel",     "classification", "Digit pixel features"),
    23:   ("cmc",             "classification", "Contraceptive method choice"),
    24:   ("mushroom",        "classification", "Mushroom edibility"),
    26:   ("nursery",         "classification", "Nursery school admissions"),
    28:   ("optdigits",       "classification", "Optical digit recognition"),
    30:   ("page_blocks",     "classification", "Document page blocks"),
    32:   ("pendigits",       "classification", "Pen-based digit recognition"),
    36:   ("segment",         "classification", "Image segmentation"),
    38:   ("sick",            "classification", "Hypothyroid sick"),
    44:   ("spambase",        "classification", "Email spam"),
    46:   ("splice",          "classification", "DNA splice junctions"),
    60:   ("waveform_5000",   "classification", "Waveform generation"),
    300:  ("isolet",          "classification", "Isolated letter speech"),
    1053: ("jm1",             "classification", "Software defects jm1"),
    1063: ("kc2",             "classification", "Software defects kc2"),
    1071: ("pc4",             "classification", "Software defects pc4"),
    1120: ("MiceProtein",     "classification", "Mice protein expression"),
    4134: ("bioresponse",     "classification", "Biological response"),
    4534: ("PhishingWebsites","classification", "Phishing URL features"),
    40923:("Satellite",       "classification", "Landsat satellite"),
    40981:("Australian",      "classification", "Australian credit"),
    40982:("car",             "classification", "Car evaluation"),
    40983:("wilt",            "classification", "Remote sensing wilt"),
    40984:("segment_2",       "classification", "Image segment 2"),
    41138:("APSFailure",      "classification", "Air pressure system failure"),
}


def _cache_path(dataset_id: int) -> Path:
    return CACHE_DIR / f"openml_{dataset_id}.csv"


def _meta_path(dataset_id: int) -> Path:
    return CACHE_DIR / f"openml_{dataset_id}_meta.json"


def load_openml_dataset(dataset_id: int) -> Optional[Tuple[pd.DataFrame, str, str]]:
    """
    Load a single OpenML dataset by ID.
    Returns (dataframe, target_column, dataset_name) or None on failure.

    Checks local cache first. Downloads if not cached.
    """
    cache = _cache_path(dataset_id)
    meta  = _meta_path(dataset_id)

    # Load from cache
    if cache.exists() and meta.exists():
        try:
            df   = pd.read_csv(cache)
            info = json.loads(meta.read_text())
            return df, info["target"], info["name"]
        except Exception:
            pass

    # Try OpenML download
    try:
        import openml
        openml.config.apikey = ''

        dataset = openml.datasets.get_dataset(
            dataset_id,
            download_data=True,
            download_qualities=True,
            download_features_meta_data=False,
        )
        X, y, _, attr_names = dataset.get_data(
            dataset_format='dataframe',
            target=dataset.default_target_attribute
        )

        if X is None or y is None:
            return None

        df = X.copy()
        df["target"] = y

        # Save to cache
        df.to_csv(cache, index=False)
        json.dumps({"target": "target", "name": dataset.name,
                    "openml_id": dataset_id}).encode()
        meta.write_text(json.dumps({
            "target": "target",
            "name":   dataset.name,
            "openml_id": dataset_id,
            "n_samples": len(df),
            "n_features": X.shape[1],
        }))

        return df, "target", dataset.name

    except Exception as e:
        return None


def load_sklearn_builtins() -> List[Tuple[pd.DataFrame, str, str]]:
    """Load all sklearn built-in datasets as (df, target, name) tuples."""
    from sklearn.datasets import (
        load_breast_cancer, load_iris, load_wine, load_digits
    )
    results = []
    for name, loader in [
        ("breast_cancer", load_breast_cancer),
        ("wine",          load_wine),
        ("iris",          load_iris),
        ("digits",        load_digits),
    ]:
        d  = loader(as_frame=True)
        df = d.frame.copy()
        df["target"] = d.target
        results.append((df, "target", name))
    return results


def load_dataset_collection(
    max_datasets: int = 100,
    max_samples:  int = 10000,
    max_features: int = 200,
    min_samples:  int = 100,
    verbose:      bool = True,
) -> List[Tuple[pd.DataFrame, str, str]]:
    """
    Load a full collection of real datasets for experiments.

    Strategy:
    1. Always include sklearn built-ins (4 datasets)
    2. Try OpenML IDs in order — stop when max_datasets reached
    3. Skip datasets that are too large, too small, or fail to load

    Returns:
        List of (dataframe, target_column, dataset_name)
    """
    collection = []

    # Step 1: sklearn built-ins (always reliable)
    builtins = load_sklearn_builtins()
    collection.extend(builtins)
    if verbose:
        print(f"  Loaded {len(builtins)} sklearn built-in datasets")

    # Step 2: OpenML datasets
    n_attempted = 0
    n_loaded    = 0
    n_failed    = 0

    for ds_id, (ds_name, task, notes) in OPENML_DATASET_IDS.items():
        if len(collection) >= max_datasets:
            break
        if task != "classification":
            continue

        n_attempted += 1
        result = load_openml_dataset(ds_id)

        if result is None:
            n_failed += 1
            if verbose:
                print(f"    ✗ {ds_name} (id={ds_id}) — failed to load")
            continue

        df, target, name = result

        # Size filters
        if len(df) < min_samples:
            if verbose:
                print(f"    ✗ {name} — too few samples ({len(df)})")
            continue
        if len(df) > max_samples:
            # Subsample rather than skip
            df = df.sample(max_samples, random_state=42).reset_index(drop=True)
        if df.shape[1] > max_features + 1:
            if verbose:
                print(f"    ✗ {name} — too many features ({df.shape[1]-1})")
            continue

        # Target sanity check
        y = df[target]
        if y.nunique() < 2 or y.isnull().any():
            continue

        collection.append((df, target, name))
        n_loaded += 1
        if verbose:
            print(f"    ✓ {name:35s} {len(df):6d} rows  {df.shape[1]-1:4d} feat  "
                  f"{int(y.nunique())} classes")

    if verbose:
        print(f"\n  Collection: {len(collection)} datasets loaded "
              f"({len(builtins)} sklearn + {n_loaded} OpenML, {n_failed} failed)")

    return collection


def get_dataset_properties(
    collection: List[Tuple[pd.DataFrame, str, str]]
) -> pd.DataFrame:
    """
    Return a summary DataFrame of all loaded datasets.
    Useful for inspecting diversity of the collection.
    """
    rows = []
    for df, target, name in collection:
        y    = df[target]
        n, p = df.shape[0], df.shape[1] - 1
        vc   = y.value_counts()
        rows.append({
            "name":           name,
            "n_samples":      n,
            "n_features":     p,
            "n_classes":      int(y.nunique()),
            "imbalance_ratio": round(float(vc.iloc[0]/vc.iloc[-1]), 2) if len(vc)>1 else 1.0,
            "missing_pct":    round(float(df.isnull().mean().mean()*100), 1),
            "n_numeric":      int(df.select_dtypes(include=[np.number]).shape[1] - 1),
        })
    return pd.DataFrame(rows).sort_values("n_samples")


if __name__ == "__main__":
    print("Testing dataset loader...")
    collection = load_dataset_collection(max_datasets=10, verbose=True)
    props = get_dataset_properties(collection)
    print("\nDataset summary:")
    print(props.to_string(index=False))
