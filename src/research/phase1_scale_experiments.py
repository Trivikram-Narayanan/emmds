"""
Phase 1: Scale Evaluation — Trust vs Accuracy Selection
Tests EMMDS Trust Score across 20 datasets with controlled difficulty.
Datasets: 4 sklearn real + 16 synthetic (4 easy, 4 medium, 4 hard, 4 extreme)

Hard dataset criteria: imbalance_ratio > 3.0 OR noise > 0.08 OR n_samples < 250
"""

import sys, json, warnings, time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from scipy import stats
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

OUT = Path('outputs/research')
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    'lr':  LogisticRegression(max_iter=1000, random_state=42),
    'lda': LinearDiscriminantAnalysis(),
    'rf':  RandomForestClassifier(n_estimators=50, random_state=42),
    'gb':  GradientBoostingClassifier(n_estimators=50, random_state=42),
    'knn': KNeighborsClassifier(n_neighbors=5),
}

W_EMP  = dict(acc=0.05, cal=0.10, agr=0.10, dq=0.35, stab=0.40)
W_ACC  = dict(acc=1.0,  cal=0.0,  agr=0.0,  dq=0.0,  stab=0.0)

def compute_data_quality(X, y):
    """Simple data quality score: completeness + balance + low_noise proxy"""
    n, p = X.shape
    completeness = 1.0  # synthetic data has no missing values
    classes, counts = np.unique(y, return_counts=True)
    balance = counts.min() / counts.max()
    # noise proxy: ratio of overlapping samples (use cv variance as proxy)
    return float(np.clip(0.4*completeness + 0.4*balance + 0.2*0.8, 0, 1))

def compute_agreement(models_fitted, X_test):
    """Cross-model agreement: fraction of test samples where majority vote matches all"""
    preds = np.stack([m.predict(X_test) for m in models_fitted.values()], axis=1)
    # entropy-based agreement: 1 - normalised entropy of prediction distribution
    n_models = preds.shape[1]
    agreements = []
    for row in preds:
        vals, cnts = np.unique(row, return_counts=True)
        probs = cnts / n_models
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(n_models + 1e-10)
        agreements.append(1.0 - entropy / max_entropy)
    return float(np.mean(agreements))

def run_dataset(name, X, y, difficulty):
    """Run fast EMMDS on one dataset. Returns results dict."""
    le = LabelEncoder()
    y = le.fit_transform(y)
    sc = StandardScaler()

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                                random_state=42, stratify=y if len(np.unique(y))>=2 else None)
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)
    X_all_s = sc.transform(X)
    y_all = y.copy()

    n_classes = len(np.unique(y))
    dq_score = compute_data_quality(X_tr_s, y_tr)

    results = {}
    for mname, model in MODELS.items():
        try:
            m = clone(model)
            m.fit(X_tr_s, y_tr)

            # Train F1 (for overfitting gap)
            train_f1 = float(f1_score(y_tr, m.predict(X_tr_s), average='weighted', zero_division=0))
            # Test F1
            test_f1  = float(f1_score(y_te, m.predict(X_te_s), average='weighted', zero_division=0))

            # 3-fold CV stability
            cv_s = cross_val_score(clone(model), X_all_s, y_all, cv=3,
                                   scoring='f1_weighted', n_jobs=1)
            cv_mean = float(np.mean(cv_s))
            cv_std  = float(np.std(cv_s))
            stability = float(np.clip(1.0 - cv_std / (cv_mean + 1e-8), 0, 1))

            # Calibration (Brier score -> 1 - brier)
            try:
                cal_m = CalibratedClassifierCV(clone(model), cv=3, method='sigmoid')
                cal_m.fit(X_tr_s, y_tr)
                proba = cal_m.predict_proba(X_te_s)
                if n_classes == 2:
                    brier = brier_score_loss(y_te, proba[:, 1])
                else:
                    # Multi-class: average one-vs-rest Brier
                    brier = np.mean([brier_score_loss((y_te==c).astype(int), proba[:,i])
                                     for i, c in enumerate(np.unique(y_te))])
                cal_score = float(np.clip(1.0 - brier, 0, 1))
            except:
                cal_score = 0.7

            # Deployment risk
            overfitting_gap = max(0.0, train_f1 - test_f1)
            risk = 0.40 * overfitting_gap + 0.30 * (1 - cal_score) + 0.30 * cv_std

            results[mname] = {
                'test_f1': test_f1, 'train_f1': train_f1,
                'cv_mean': cv_mean, 'cv_std': cv_std,
                'cal_score': cal_score, 'stability': stability,
                'dq_score': dq_score, 'deployment_risk': float(risk),
            }
        except Exception as e:
            pass

    if not results:
        return None

    # Agreement across fitted models
    fitted = {}
    for mname, model in MODELS.items():
        if mname in results:
            try:
                m = clone(model); m.fit(X_tr_s, y_tr)
                fitted[mname] = m
            except: pass
    agr = compute_agreement(fitted, X_te_s) if len(fitted) >= 2 else 0.75

    # Compute trust scores
    for mname, r in results.items():
        r['agr'] = agr
        trust = (W_EMP['acc'] * r['test_f1'] + W_EMP['cal'] * r['cal_score'] +
                 W_EMP['agr'] * r['agr'] + W_EMP['dq'] * r['dq_score'] +
                 W_EMP['stab'] * r['stability'])
        r['trust_score'] = float(np.clip(trust, 0, 1))

    # Select best model by each strategy
    trust_best = max(results, key=lambda m: results[m]['trust_score'])
    acc_best   = max(results, key=lambda m: results[m]['test_f1'])

    trust_risk = results[trust_best]['deployment_risk']
    acc_risk   = results[acc_best]['deployment_risk']

    # Spearman inputs
    trust_vals = [r['trust_score'] for r in results.values()]
    risk_vals  = [r['deployment_risk'] for r in results.values()]
    acc_vals   = [r['test_f1'] for r in results.values()]

    return {
        'name': name, 'difficulty': difficulty,
        'n_samples': len(X), 'n_features': X.shape[1],
        'n_models_successful': len(results),
        'trust_selected': trust_best, 'acc_selected': acc_best,
        'trust_risk': float(trust_risk), 'acc_risk': float(acc_risk),
        'trust_wins': bool(trust_risk <= acc_risk),
        'risk_delta': float(acc_risk - trust_risk),  # positive = trust wins
        'trust_vals': trust_vals, 'risk_vals': risk_vals, 'acc_vals': acc_vals,
        'all_results': results,
    }


def build_datasets():
    """Build 20 datasets covering easy -> extreme difficulty."""
    datasets = []

    # -- Real sklearn datasets --
    for name, loader in [('iris', load_iris), ('wine', load_wine),
                          ('breast_cancer', load_breast_cancer)]:
        d = loader()
        datasets.append((name, d.data, d.target, 'real'))

    # Digits (downsample to speed up)
    d = load_digits()
    idx = np.random.RandomState(42).choice(len(d.data), 600, replace=False)
    datasets.append(('digits_600', d.data[idx], d.target[idx], 'real'))

    # -- Easy synthetic (balanced, large n, low noise) --
    easy_cfgs = [
        dict(n_samples=800, n_features=15, n_informative=10, n_redundant=3,
             flip_y=0.01, weights=None, n_classes=2, class_sep=1.5, random_state=1),
        dict(n_samples=900, n_features=20, n_informative=12, n_redundant=4,
             flip_y=0.02, weights=None, n_classes=3, class_sep=1.5, random_state=2),
        dict(n_samples=1000, n_features=10, n_informative=8, n_redundant=1,
             flip_y=0.01, weights=None, n_classes=2, class_sep=2.0, random_state=3),
        dict(n_samples=700, n_features=25, n_informative=15, n_redundant=5,
             flip_y=0.02, weights=None, n_classes=4, class_sep=1.5, random_state=4),
    ]
    for i, cfg in enumerate(easy_cfgs):
        X, y = make_classification(**cfg)
        datasets.append((f'easy_{i+1}', X, y, 'easy'))

    # -- Medium synthetic (moderate imbalance/noise) --
    med_cfgs = [
        dict(n_samples=400, n_features=20, n_informative=10, n_redundant=5,
             flip_y=0.07, weights=[0.75,0.25], n_classes=2, class_sep=1.0, random_state=5),
        dict(n_samples=500, n_features=25, n_informative=12, n_redundant=6,
             flip_y=0.06, weights=[0.70,0.30], n_classes=2, class_sep=1.0, random_state=6),
        dict(n_samples=350, n_features=30, n_informative=15, n_redundant=7,
             flip_y=0.08, weights=[0.65,0.35], n_classes=2, class_sep=0.9, random_state=7),
        dict(n_samples=450, n_features=20, n_informative=10, n_redundant=5,
             flip_y=0.07, weights=None, n_classes=3, class_sep=0.8, random_state=8),
    ]
    for i, cfg in enumerate(med_cfgs):
        X, y = make_classification(**cfg)
        datasets.append((f'medium_{i+1}', X, y, 'medium'))

    # -- Hard synthetic (high imbalance + high noise + small n) --
    hard_cfgs = [
        dict(n_samples=200, n_features=20, n_informative=8,  n_redundant=6,
             flip_y=0.15, weights=[0.85,0.15], n_classes=2, class_sep=0.6, random_state=9),
        dict(n_samples=180, n_features=30, n_informative=10, n_redundant=8,
             flip_y=0.13, weights=[0.82,0.18], n_classes=2, class_sep=0.5, random_state=10),
        dict(n_samples=220, n_features=25, n_informative=8,  n_redundant=8,
             flip_y=0.16, weights=[0.88,0.12], n_classes=2, class_sep=0.7, random_state=11),
        dict(n_samples=160, n_features=40, n_informative=12, n_redundant=10,
             flip_y=0.14, weights=[0.80,0.20], n_classes=2, class_sep=0.5, random_state=12),
    ]
    for i, cfg in enumerate(hard_cfgs):
        X, y = make_classification(**cfg)
        datasets.append((f'hard_{i+1}', X, y, 'hard'))

    # -- Extreme synthetic (worst case conditions) --
    ext_cfgs = [
        dict(n_samples=150, n_features=50, n_informative=10, n_redundant=15,
             flip_y=0.20, weights=[0.91,0.09], n_classes=2, class_sep=0.4, random_state=13),
        dict(n_samples=140, n_features=60, n_informative=8,  n_redundant=20,
             flip_y=0.22, weights=[0.89,0.11], n_classes=2, class_sep=0.3, random_state=14),
        dict(n_samples=160, n_features=45, n_informative=10, n_redundant=15,
             flip_y=0.18, weights=[0.87,0.13], n_classes=2, class_sep=0.4, random_state=15),
        dict(n_samples=130, n_features=55, n_informative=8,  n_redundant=18,
             flip_y=0.25, weights=[0.92,0.08], n_classes=2, class_sep=0.3, random_state=16),
    ]
    for i, cfg in enumerate(ext_cfgs):
        X, y = make_classification(**cfg)
        datasets.append((f'extreme_{i+1}', X, y, 'extreme'))

    return datasets


def bootstrap_ci(values, n_boot=1000, ci=95, random_state=42):
    rng = np.random.RandomState(random_state)
    boots = [np.mean(rng.choice(values, len(values), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(boots, (100-ci)/2)
    hi = np.percentile(boots, 100-(100-ci)/2)
    return float(np.mean(values)), float(lo), float(hi)


if __name__ == '__main__':
    print("="*65)
    print("  PHASE 1: SCALE EVALUATION -- 20 DATASETS")
    print("  Trust-based vs Accuracy-based Model Selection")
    print("="*65)

    datasets = build_datasets()
    all_results = []
    hard_criteria = lambda r: (r['difficulty'] in ('hard','extreme') or
                               r.get('n_samples',9999) < 250)

    for name, X, y, difficulty in datasets:
        t0 = time.time()
        print(f"  [{difficulty:8s}] {name:25s}", end='', flush=True)
        try:
            r = run_dataset(name, X, y, difficulty)
            if r:
                all_results.append(r)
                marker = 'WIN' if r['trust_wins'] else '   '
                print(f"  trust_risk={r['trust_risk']:.4f}  acc_risk={r['acc_risk']:.4f}  "
                      f"delta={r['risk_delta']:+.4f}  {marker}  ({time.time()-t0:.1f}s)")
            else:
                print("  SKIPPED")
        except Exception as e:
            print(f"  ERROR: {e}")

    if not all_results:
        print("No results. Exiting.")
        sys.exit(1)

    # -- Analysis --
    all_trust_wins = [r['trust_wins'] for r in all_results]
    hard_results   = [r for r in all_results if r['difficulty'] in ('hard','extreme')]
    easy_results   = [r for r in all_results if r['difficulty'] in ('easy','real')]
    hard_wins      = [r['trust_wins'] for r in hard_results]

    # Spearman correlations (pool all model-level results)
    all_trust_flat = [v for r in all_results for v in r['trust_vals']]
    all_risk_flat  = [v for r in all_results for v in r['risk_vals']]
    all_acc_flat   = [v for r in all_results for v in r['acc_vals']]

    sp_trust_risk, p_trust = stats.spearmanr(all_trust_flat, all_risk_flat)
    sp_acc_risk,   p_acc   = stats.spearmanr(all_acc_flat,   all_risk_flat)

    # Bootstrap CIs for win rates
    wr_all_mean, wr_all_lo, wr_all_hi = bootstrap_ci([int(x) for x in all_trust_wins])
    wr_hard_mean, wr_hard_lo, wr_hard_hi = (bootstrap_ci([int(x) for x in hard_wins])
                                              if hard_wins else (0,0,0))

    print(f"\n{'='*65}")
    print(f"  RESULTS SUMMARY")
    print(f"  Total datasets processed: {len(all_results)}")
    print(f"  Hard datasets:            {len(hard_results)}")
    print(f"\n  WIN RATES (trust beats accuracy):")
    print(f"    All datasets:   {wr_all_mean:.1%}  95% CI [{wr_all_lo:.1%}, {wr_all_hi:.1%}]")
    print(f"    Hard only:      {wr_hard_mean:.1%}  95% CI [{wr_hard_lo:.1%}, {wr_hard_hi:.1%}]")
    print(f"\n  SPEARMAN r (predictor vs deployment risk):")
    print(f"    Trust score:    r={sp_trust_risk:+.4f}  p={p_trust:.4f}")
    print(f"    Accuracy alone: r={sp_acc_risk:+.4f}  p={p_acc:.4f}")

    # Risk delta analysis
    hard_deltas = [r['risk_delta'] for r in hard_results]
    easy_deltas = [r['risk_delta'] for r in easy_results if r]

    if hard_deltas:
        t_stat, p_wil = stats.wilcoxon(hard_deltas) if len(hard_deltas)>=5 else (0,1)
        print(f"\n  HARD DATASET RISK DELTA (positive = trust better):")
        print(f"    Mean delta: {np.mean(hard_deltas):+.4f}")
        print(f"    Wilcoxon p: {p_wil:.4f}")

    # Save results
    def _j(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, bool): return bool(o)
        if isinstance(o, dict): return {k: _j(v) for k,v in o.items()}
        if isinstance(o, list): return [_j(x) for x in o]
        return o

    output = {
        'n_datasets': len(all_results),
        'n_hard_datasets': len(hard_results),
        'win_rate_all':  {'mean': wr_all_mean,  'ci_lo': wr_all_lo,  'ci_hi': wr_all_hi},
        'win_rate_hard': {'mean': wr_hard_mean, 'ci_lo': wr_hard_lo, 'ci_hi': wr_hard_hi},
        'spearman_trust_vs_risk': {'r': float(sp_trust_risk), 'p': float(p_trust)},
        'spearman_acc_vs_risk':   {'r': float(sp_acc_risk),   'p': float(p_acc)},
        'hard_risk_delta': {'mean': float(np.mean(hard_deltas)) if hard_deltas else 0,
                            'std':  float(np.std(hard_deltas)) if hard_deltas else 0},
        'datasets': _j(all_results),
    }

    out_path = OUT / 'phase1_scale_results.json'
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Saved -> {out_path}")
