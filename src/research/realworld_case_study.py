"""
Real-world temporal case study using UCI Electricity dataset.

Downloads OpenML dataset 151 (Electricity, 45,312 samples, temporal ordering).
Splits by time: train on first 60%, calibrate on next 20%, test on last 20%.
Compares EMMDS trust-selected model vs accuracy-only on progressive temporal slices.

Shows that trust selection degrades less gracefully under covariate shift
compared to pure accuracy selection.

Outputs: outputs/research/realworld_case_study.json
"""
import sys
import json
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from scipy import stats

MODELS = {
    "lr": LogisticRegression(max_iter=500, random_state=42),
    "lda": LinearDiscriminantAnalysis(),
    "rf": RandomForestClassifier(n_estimators=80, random_state=42),
    "gb": GradientBoostingClassifier(n_estimators=80, random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=7),
}


def compute_trust_score(cv_mean, cal, agreement, dq, stability):
    return (0.05 * cv_mean + 0.10 * cal + 0.10 * agreement +
            0.35 * dq + 0.40 * stability)


def compute_deployment_risk(overfit, cal_err, cv_std):
    return 0.40 * max(0, overfit) + 0.30 * cal_err + 0.30 * cv_std


def _ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def train_and_select(X_train, y_train, X_test, y_test):
    """
    Train all candidate models, compute trust and risk, return:
    - trust_winner (model with max trust)
    - acc_winner (model with max CV accuracy)
    - per-model metrics
    """
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    imb = max(np.bincount(y_train.astype(int))) / len(y_train)
    dq = float(np.clip(1.0 - 0.5 * (imb - 0.5), 0, 1))
    cv = StratifiedKFold(n_splits=3, shuffle=False)  # no shuffle: temporal

    records = {}
    for name, clf in MODELS.items():
        clf_copy = type(clf)(**clf.get_params())
        scores = cross_val_score(clf_copy, X_tr, y_train, cv=cv, scoring="f1_weighted")
        cv_mean, cv_std = float(scores.mean()), float(scores.std())

        clf_copy.fit(X_tr, y_train)
        train_f1 = f1_score(y_train, clf_copy.predict(X_tr), average="weighted")
        test_f1 = f1_score(y_test, clf_copy.predict(X_te), average="weighted")
        overfit = max(0, train_f1 - test_f1)

        if hasattr(clf_copy, "predict_proba"):
            y_prob = clf_copy.predict_proba(X_te)[:, 1]
            cal_err = _ece(y_test, y_prob)
        else:
            cal_err = 0.20

        cal = 1.0 - cal_err
        stability = max(0.0, 1.0 - cv_std)
        trust = compute_trust_score(cv_mean, cal, 1.0, dq, stability)
        risk = compute_deployment_risk(overfit, cal_err, cv_std)

        records[name] = dict(
            cv_mean=cv_mean, cv_std=cv_std, test_f1=test_f1,
            overfit=overfit, cal_err=cal_err, trust=trust, risk=risk,
        )

    trust_winner = max(records, key=lambda k: records[k]["trust"])
    acc_winner = max(records, key=lambda k: records[k]["cv_mean"])
    return trust_winner, acc_winner, records


def load_electricity():
    """Load the Electricity dataset via OpenML, with graceful synthetic fallback."""
    try:
        import openml
        ds = openml.datasets.get_dataset(151, download_data=True)
        X, y, _, _ = ds.get_data(target=ds.default_target_attribute)
        X = X.values if hasattr(X, 'values') else np.array(X)
        y = (y.values if hasattr(y, 'values') else np.array(y))
        # Binarise
        unique = np.unique(y)
        y = (y == unique[-1]).astype(int)
        print(f"  Loaded Electricity: n={len(X)}, features={X.shape[1]}")
        return X.astype(float), y.astype(int), "electricity_openml"
    except Exception as e:
        print(f"  OpenML load failed ({e}), generating synthetic temporal substitute")
        return _synthetic_temporal()


def _synthetic_temporal():
    """
    Synthetic dataset with temporal structure: feature distributions shift
    progressively over time, simulating electricity price patterns.
    """
    from sklearn.datasets import make_classification
    rng = np.random.default_rng(42)
    n = 10000
    X, y = make_classification(
        n_samples=n, n_features=8, n_informative=5, n_redundant=2,
        flip_y=0.02, random_state=42
    )
    # Add progressive covariate shift
    time_idx = np.arange(n)
    shift = (time_idx / n) * 1.5  # grows from 0 to 1.5 over time
    X = X + rng.normal(0, shift.reshape(-1, 1) * 0.3, X.shape)
    return X.astype(float), y.astype(int), "electricity_synthetic"


def main():
    out_dir = ROOT / "outputs" / "research"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Electricity dataset...")
    X, y, dataset_name = load_electricity()
    n = len(X)

    # Temporal splits: 60/20/20
    t1 = int(0.60 * n)
    t2 = int(0.80 * n)
    X_train, y_train = X[:t1], y[:t1]
    X_cal,   y_cal   = X[t1:t2], y[t1:t2]
    X_test,  y_test  = X[t2:], y[t2:]

    print(f"  Train: {len(X_train)}, Cal: {len(X_cal)}, Test: {len(X_test)}")

    # Select model on training set, evaluate on calibration and test
    print("Selecting models on training split...")
    trust_winner, acc_winner, train_records = train_and_select(X_train, y_train, X_cal, y_cal)

    # Retrain selected models on train+cal, evaluate on test (temporal forward)
    results_by_slice = []
    slices = [("near_term", X_cal, y_cal), ("far_term", X_test, y_test)]
    for slice_name, X_eval, y_eval in slices:
        scaler = StandardScaler()
        X_tr_full = scaler.fit_transform(np.vstack([X_train, X_cal]))
        y_tr_full = np.concatenate([y_train, y_cal])
        X_ev = scaler.transform(X_eval)

        slice_rec = {"slice": slice_name, "n_eval": len(X_eval)}
        for selector, winner in [("trust", trust_winner), ("accuracy", acc_winner)]:
            clf = type(MODELS[winner])(**MODELS[winner].get_params())
            clf.fit(X_tr_full, y_tr_full)
            f1 = float(f1_score(y_eval, clf.predict(X_ev), average="weighted"))
            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(X_ev)[:, 1]
                cal_err = _ece(y_eval, y_prob)
            else:
                cal_err = 0.20
            overfit = max(0, train_records[winner]["cv_mean"] - f1)
            cv_std = train_records[winner]["cv_std"]
            risk = compute_deployment_risk(overfit, cal_err, cv_std)
            slice_rec[f"{selector}_winner"] = winner
            slice_rec[f"{selector}_f1"] = round(f1, 4)
            slice_rec[f"{selector}_risk"] = round(risk, 5)
        slice_rec["trust_beats_acc_on_risk"] = (
            slice_rec["trust_risk"] <= slice_rec["accuracy_risk"]
        )
        results_by_slice.append(slice_rec)
        print(f"  {slice_name}: trust={slice_rec['trust_winner']} "
              f"risk={slice_rec['trust_risk']:.4f} vs "
              f"acc={slice_rec['accuracy_winner']} risk={slice_rec['accuracy_risk']:.4f}")

    # Progressive degradation: test risk on 5 sub-windows of the test set
    degradation = []
    chunk_size = max(50, len(X_test) // 5)
    scaler_prog = StandardScaler()
    X_tr_prog = scaler_prog.fit_transform(np.vstack([X_train, X_cal]))
    y_tr_prog = np.concatenate([y_train, y_cal])

    for i in range(5):
        lo, hi = i * chunk_size, min((i + 1) * chunk_size, len(X_test))
        if lo >= hi:
            break
        X_chunk = scaler_prog.transform(X_test[lo:hi])
        y_chunk = y_test[lo:hi]

        chunk_rec = {"chunk": i + 1, "time_fraction": round((t2 + lo) / n, 3)}
        for selector, winner in [("trust", trust_winner), ("accuracy", acc_winner)]:
            clf = type(MODELS[winner])(**MODELS[winner].get_params())
            clf.fit(X_tr_prog, y_tr_prog)
            f1 = float(f1_score(y_chunk, clf.predict(X_chunk), average="weighted",
                                zero_division=0))
            overfit = max(0, train_records[winner]["cv_mean"] - f1)
            cv_std = train_records[winner]["cv_std"]
            risk = compute_deployment_risk(overfit, 0.15, cv_std)
            chunk_rec[f"{selector}_f1"] = round(f1, 4)
            chunk_rec[f"{selector}_risk"] = round(risk, 5)
        degradation.append(chunk_rec)

    # Wilcoxon test: trust risks vs acc risks across chunks
    t_risks = [c["trust_risk"] for c in degradation]
    a_risks = [c["accuracy_risk"] for c in degradation]
    if len(t_risks) >= 3:
        try:
            stat, pval = stats.wilcoxon(t_risks, a_risks)
        except Exception:
            stat, pval = float("nan"), float("nan")
    else:
        stat, pval = float("nan"), float("nan")

    result = {
        "dataset": dataset_name,
        "n_total": n,
        "train_size": t1,
        "cal_size": t2 - t1,
        "test_size": n - t2,
        "selected_trust_winner": trust_winner,
        "selected_acc_winner": acc_winner,
        "temporal_slices": results_by_slice,
        "progressive_degradation": degradation,
        "statistical_test": {
            "test": "wilcoxon_trust_risk_vs_acc_risk_on_chunks",
            "statistic": stat if not np.isnan(stat) else None,
            "p_value": pval if not np.isnan(pval) else None,
            "verdict": (
                "TRUST_BETTER" if (not np.isnan(pval) and pval < 0.05
                                   and np.mean(t_risks) < np.mean(a_risks))
                else "NO_SIGNIFICANT_DIFFERENCE"
            ),
        },
        "trust_winner_train_trust": round(train_records[trust_winner]["trust"], 4),
        "acc_winner_train_trust": round(train_records[acc_winner]["trust"], 4),
    }

    out_path = out_dir / "realworld_case_study.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    pval_str = f"{pval:.4f}" if not np.isnan(pval) else "nan"
    print(f"\nWilcoxon p={pval_str} → {result['statistical_test']['verdict']}")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
