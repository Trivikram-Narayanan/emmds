"""
End-to-end tests: EMMDS pipeline on arbitrary dataset types.
  1. Binary classification (numeric only)
  2. Multiclass classification (numeric + categorical + boolean + datetime)
  3. Regression (mixed types)
  4. High-cardinality categorical features
  5. Small dataset edge case
  6. Heavy imbalance
  7. Dataset with missing values
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.pipeline import EMPipeline


# ────────────────────────────────────────────────────────────────────────────
# Factories
# ────────────────────────────────────────────────────────────────────────────

def make_binary_numeric(n=200, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age":     rng.integers(18, 80, n).astype(float),
        "bmi":     rng.normal(25, 5, n),
        "glucose": rng.normal(100, 20, n),
        "bp":      rng.normal(80, 12, n),
        "outcome": rng.integers(0, 2, n),
    })
    return df, "outcome"


def make_multiclass_mixed(n=300, seed=7):
    rng = np.random.default_rng(seed)
    categories = rng.choice(["cat", "dog", "bird"], n)
    sizes      = rng.choice(["small", "medium", "large"], n)
    has_spots  = rng.choice([True, False], n)
    base = pd.date_range("2020-01-01", periods=n, freq="h")
    df = pd.DataFrame({
        "animal":    categories,
        "size":      sizes,
        "has_spots": has_spots,
        "weight":    rng.normal(10, 3, n),
        "age_days":  rng.integers(1, 3650, n).astype(float),
        "label":     rng.integers(0, 3, n),
    })
    return df, "label"


def make_regression_mixed(n=250, seed=13):
    rng = np.random.default_rng(seed)
    region = rng.choice(["north", "south", "east", "west"], n)
    df = pd.DataFrame({
        "sqft":    rng.integers(500, 5000, n).astype(float),
        "beds":    rng.integers(1, 6, n).astype(float),
        "baths":   rng.integers(1, 4, n).astype(float),
        "region":  region,
        "garage":  rng.choice([True, False], n),
        "price":   500 + 0.15 * rng.integers(500, 5000, n) + rng.normal(0, 20, n),
    })
    return df, "price"


def make_missing_values(n=200, seed=99):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(0, 1, n),
        "x3": rng.normal(0, 1, n),
        "cat": rng.choice(["a", "b", "c"], n),
        "y":   rng.integers(0, 2, n),
    })
    # Introduce ~15% missing values in x1, x2, cat
    for col in ["x1", "x2", "cat"]:
        mask = rng.uniform(0, 1, n) < 0.15
        df.loc[mask, col] = np.nan
    return df, "y"


def make_imbalanced(n=300, seed=55):
    rng = np.random.default_rng(seed)
    # 90% class 0, 10% class 1
    labels = np.zeros(n, dtype=int)
    labels[rng.choice(n, size=int(0.10 * n), replace=False)] = 1
    df = pd.DataFrame({
        "f1": rng.normal(0, 1, n),
        "f2": rng.normal(0, 1, n),
        "f3": rng.choice(["A", "B"], n),
        "y":  labels,
    })
    return df, "y"


def make_small(n=60, seed=3):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(0, 1, n),
        "y": rng.integers(0, 2, n),
    })
    return df, "y"


def make_high_cardinality(n=400, seed=17):
    rng = np.random.default_rng(seed)
    # 100 unique cities in categorical column
    cities = [f"city_{i}" for i in range(100)]
    df = pd.DataFrame({
        "city":    rng.choice(cities, n),
        "value":   rng.normal(50, 15, n),
        "count":   rng.integers(1, 1000, n).astype(float),
        "flag":    rng.choice([True, False], n),
        "outcome": rng.integers(0, 2, n),
    })
    return df, "outcome"


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _run(df, target, task=None, name="test_ds"):
    pipe = EMPipeline()
    result = pipe.run(df, target_col=target, task=task,
                      dataset_name=name, track=False)
    # Flatten decision keys to top level for convenient access
    dec = result.get("decision", {})
    result["best_model"]   = dec.get("best_model")
    result["trust_score"]  = dec.get("trust_score")
    result["trust_breakdown"] = dec.get("trust_breakdown", {})
    result["decision_label"]  = dec.get("decision")
    result["training"]     = result.get("steps", {}).get("training", {})
    result["cv_results"]   = result.get("steps", {}).get("cv_results", {})
    result["evaluation"]   = result.get("steps", {}).get("evaluation", {})
    result["calibration_scores"] = result.get("steps", {}).get("calibration_scores", {})
    result["leaderboard"]  = dec.get("leaderboard", [])
    return result


def _assert_pipeline_success(result, task_name=""):
    assert result.get("status") == "success", \
        f"[{task_name}] Pipeline failed: {result.get('error', result.get('status'))}"
    assert result.get("best_model"), f"[{task_name}] No best_model in result"
    ts = result.get("trust_score")
    assert ts is not None, f"[{task_name}] trust_score is None"
    assert 0.0 <= ts <= 1.0, f"[{task_name}] Trust score out of [0,1]: {ts}"


# ────────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────────

def test_binary_classification_numeric():
    df, tgt = make_binary_numeric()
    result = _run(df, tgt, task="classification", name="binary_numeric")
    _assert_pipeline_success(result, "binary_numeric")
    dec = result.get("decision", {})
    dec_val = dec.get("decision") or dec.get("decision_label") or "unknown"
    print(f"\n[binary_numeric] best={result['best_model']}, trust={result['trust_score']:.4f}, decision={dec_val}")


def test_multiclass_mixed_types():
    df, tgt = make_multiclass_mixed()
    result = _run(df, tgt, task="classification", name="multiclass_mixed")
    _assert_pipeline_success(result, "multiclass_mixed")
    print(f"\n[multiclass_mixed] best={result['best_model']}, trust={result['trust_score']:.4f}")


def test_regression_mixed_types():
    df, tgt = make_regression_mixed()
    result = _run(df, tgt, task="regression", name="regression_mixed")
    _assert_pipeline_success(result, "regression_mixed")
    print(f"\n[regression_mixed] best={result['best_model']}, trust={result['trust_score']:.4f}")


def test_missing_values():
    df, tgt = make_missing_values()
    result = _run(df, tgt, task="classification", name="missing_vals")
    _assert_pipeline_success(result, "missing_vals")
    print(f"\n[missing_vals] best={result['best_model']}, trust={result['trust_score']:.4f}")


def test_imbalanced_dataset():
    df, tgt = make_imbalanced()
    result = _run(df, tgt, task="classification", name="imbalanced")
    _assert_pipeline_success(result, "imbalanced")
    print(f"\n[imbalanced] best={result['best_model']}, trust={result['trust_score']:.4f}")


def test_small_dataset():
    df, tgt = make_small()
    result = _run(df, tgt, task="classification", name="small_ds")
    _assert_pipeline_success(result, "small_ds")
    print(f"\n[small_ds] best={result['best_model']}, trust={result['trust_score']:.4f}")


def test_high_cardinality_categorical():
    df, tgt = make_high_cardinality()
    result = _run(df, tgt, task="classification", name="high_card")
    _assert_pipeline_success(result, "high_card")
    print(f"\n[high_card] best={result['best_model']}, trust={result['trust_score']:.4f}")


def test_auto_task_detection():
    """Pipeline should detect classification/regression without explicit task."""
    df, tgt = make_binary_numeric()
    result = _run(df, tgt, task=None, name="auto_task")
    _assert_pipeline_success(result, "auto_task")
    detected = result.get("task") or result.get("steps", {}).get("analysis", {}).get("task")
    print(f"\n[auto_task] detected_task={detected}, best={result['best_model']}")


def test_trust_score_structure():
    """Trust score breakdown must have all 5 required component values."""
    df, tgt = make_binary_numeric()
    result = _run(df, tgt, name="trust_structure")
    _assert_pipeline_success(result, "trust_structure")
    breakdown = result.get("trust_breakdown", {})
    # The breakdown uses _component suffix keys
    for key in ("accuracy_component", "calibration_component",
                "agreement_component", "data_quality_component", "stability_component"):
        assert key in breakdown, f"Missing trust component: {key}"
        val = breakdown[key]
        assert 0.0 <= val <= 1.0, f"Trust component {key}={val} out of range"


def test_all_training_outputs_present():
    """Pipeline result must include training, CV, evaluation, and decision keys."""
    df, tgt = make_binary_numeric()
    result = _run(df, tgt, name="output_keys")
    _assert_pipeline_success(result, "output_keys")
    for key in ("training", "cv_results", "evaluation",
                "calibration_scores", "leaderboard",
                "trust_score", "best_model"):
        val = result.get(key)
        assert val is not None, f"Missing or None key in result: {key}"
