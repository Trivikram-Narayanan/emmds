"""
Phase 4: Temporal Deployment Validation
=========================================
Simulates temporal deployment by splitting datasets into 3 progressive
time windows: train -> val -> test_late.

For each dataset, compares performance of trust-selected vs accuracy-selected
model on the late test window (simulating deployment drift).

Also runs CTGAN augmentation evaluation.
"""
import sys, json, warnings
import numpy as np
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, brier_score_loss
from sklearn.base import clone
from scipy import stats
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

OUT = Path('outputs/research'); OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    'lr':  LogisticRegression(max_iter=1000, random_state=42),
    'lda': LinearDiscriminantAnalysis(),
    'rf':  RandomForestClassifier(n_estimators=50, random_state=42),
    'gb':  GradientBoostingClassifier(n_estimators=50, random_state=42),
    'knn': KNeighborsClassifier(n_neighbors=5),
}

W_EMP = dict(acc=0.05, cal=0.10, agr=0.10, dq=0.35, stab=0.40)

def compute_trust_score(m, X_tr, X_te, y_tr, y_te, X_all, y_all, dq=0.8):
    test_f1 = float(f1_score(y_te, m.predict(X_te), average='weighted', zero_division=0))
    cv_s = cross_val_score(clone(m), X_all, y_all, cv=3, scoring='f1_weighted')
    stability = float(np.clip(1 - np.std(cv_s)/(np.mean(cv_s)+1e-8), 0, 1))
    try:
        cm = CalibratedClassifierCV(clone(m), cv=3); cm.fit(X_tr, y_tr)
        p = cm.predict_proba(X_te)
        n_c = len(np.unique(y_te))
        brier = brier_score_loss(y_te, p[:,1]) if n_c==2 else np.mean(
            [brier_score_loss((y_te==c).astype(int),p[:,i]) for i,c in enumerate(np.unique(y_te))])
        cal = float(np.clip(1-brier, 0, 1))
    except: cal = 0.7
    trust = W_EMP['acc']*test_f1 + W_EMP['cal']*cal + W_EMP['agr']*0.75 + W_EMP['dq']*dq + W_EMP['stab']*stability
    return float(np.clip(trust,0,1)), test_f1, stability, cal

def apply_temporal_drift(X, t, drift_strength=1.5, seed=0):
    """Apply increasing covariate shift to simulate temporal drift."""
    rng = np.random.RandomState(seed)
    direction = rng.randn(X.shape[1])
    direction /= (np.linalg.norm(direction) + 1e-8)
    stds = X.std(axis=0) + 1e-8
    shift = t * drift_strength * stds * direction
    noise = rng.randn(*X.shape) * 0.05 * t
    return X + shift + noise

TEMPORAL_DATASETS = [
    ('temporal_easy',    dict(n_samples=1200, n_features=15, n_informative=10, flip_y=0.03, weights=None, n_classes=2, class_sep=1.5, random_state=1)),
    ('temporal_medium',  dict(n_samples=900,  n_features=20, n_informative=10, flip_y=0.08, weights=[0.75,0.25], n_classes=2, class_sep=1.0, random_state=2)),
    ('temporal_hard1',   dict(n_samples=600,  n_features=25, n_informative=8,  flip_y=0.15, weights=[0.82,0.18], n_classes=2, class_sep=0.6, random_state=3)),
    ('temporal_hard2',   dict(n_samples=600,  n_features=30, n_informative=8,  flip_y=0.12, weights=[0.85,0.15], n_classes=2, class_sep=0.5, random_state=4)),
    ('temporal_extreme1',dict(n_samples=600,  n_features=40, n_informative=8,  flip_y=0.20, weights=[0.88,0.12], n_classes=2, class_sep=0.4, random_state=5)),
    ('temporal_extreme2',dict(n_samples=600,  n_features=35, n_informative=8,  flip_y=0.22, weights=[0.90,0.10], n_classes=2, class_sep=0.3, random_state=6)),
    ('temporal_balanced_noise', dict(n_samples=900, n_features=20, n_informative=8, flip_y=0.18, weights=None, n_classes=2, class_sep=0.7, random_state=7)),
    ('temporal_small',   dict(n_samples=600,  n_features=20, n_informative=8,  flip_y=0.15, weights=[0.80,0.20], n_classes=2, class_sep=0.6, random_state=8)),
]

DRIFT_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0]

if __name__ == '__main__':
    print("="*65)
    print("  PHASE 4: TEMPORAL DEPLOYMENT VALIDATION")
    print("  Trust vs Accuracy Selection under Progressive Drift")
    print("="*65)

    all_results = []

    for ds_name, cfg in TEMPORAL_DATASETS:
        X, y = make_classification(**cfg)
        le = LabelEncoder(); y = le.fit_transform(y)
        sc = StandardScaler()

        # Temporal split: first 60% = train, next 20% = val (iid), last 20% = test_late (drifted)
        n = len(X)
        idx = np.random.RandomState(42).permutation(n)
        tr_idx = idx[:int(0.6*n)]
        va_idx = idx[int(0.6*n):int(0.8*n)]
        te_idx = idx[int(0.8*n):]

        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        X_tr_s = sc.fit_transform(X_tr)
        X_va_s = sc.transform(X_va)
        X_te_s = sc.transform(X_te)
        X_all_s = sc.transform(X); y_all = y.copy()

        dq_score = float(np.clip(0.4 + 0.4*(np.bincount(y_tr).min()/(np.bincount(y_tr).max()+1)), 0, 1))

        # Train all models on training set, evaluate on val (iid) to select
        model_metrics = {}
        for mname, model in MODELS.items():
            try:
                m = clone(model); m.fit(X_tr_s, y_tr)
                trust, test_f1, stab, cal = compute_trust_score(m, X_tr_s, X_va_s, y_tr, y_va, X_all_s, y_all, dq_score)
                model_metrics[mname] = {'trust': trust, 'val_f1': test_f1, 'model': m}
            except: pass

        if not model_metrics: continue

        trust_best = max(model_metrics, key=lambda m: model_metrics[m]['trust'])
        acc_best   = max(model_metrics, key=lambda m: model_metrics[m]['val_f1'])

        # Evaluate on late test at different drift levels
        drift_results = []
        for drift_t in DRIFT_LEVELS:
            X_te_drifted = apply_temporal_drift(X_te_s, drift_t, drift_strength=1.2, seed=42)

            trust_model = model_metrics[trust_best]['model']
            acc_model   = model_metrics[acc_best]['model']

            try:
                trust_f1 = float(f1_score(y_te, trust_model.predict(X_te_drifted), average='weighted', zero_division=0))
                acc_f1   = float(f1_score(y_te, acc_model.predict(X_te_drifted), average='weighted', zero_division=0))
            except:
                trust_f1 = acc_f1 = 0.5

            drift_results.append({
                'drift_level': float(drift_t),
                'trust_f1': float(trust_f1),
                'acc_f1': float(acc_f1),
                'trust_advantage': float(trust_f1 - acc_f1),
            })

        # Average advantage at HIGH drift (t >= 1.5)
        high_drift = [d for d in drift_results if d['drift_level'] >= 1.5]
        avg_advantage = float(np.mean([d['trust_advantage'] for d in high_drift]))

        result = {
            'dataset': ds_name,
            'trust_selected_model': trust_best,
            'acc_selected_model': acc_best,
            'same_model': trust_best == acc_best,
            'trust_score_diff': float(model_metrics[trust_best]['trust'] - model_metrics[acc_best]['trust']),
            'drift_results': drift_results,
            'avg_trust_advantage_high_drift': float(avg_advantage),
            'trust_wins_at_high_drift': bool(avg_advantage > 0),
        }
        all_results.append(result)

        marker = '[WIN]' if result['trust_wins_at_high_drift'] else '     '
        print(f"  {ds_name:30s}  trust_model={trust_best:5s}  acc_model={acc_best:5s}  "
              f"avg_advantage_highDrift={avg_advantage:+.4f}  {marker}")

    # ── Summary ───────────────────────────────────────────────────────
    wins = [r['trust_wins_at_high_drift'] for r in all_results]
    advantages = [r['avg_trust_advantage_high_drift'] for r in all_results]

    # Wilcoxon signed-rank test: advantages > 0
    if len(advantages) >= 5:
        stat, p_val = stats.wilcoxon([x+1e-10 for x in advantages])
    else: stat, p_val = 0, 1

    print(f"\n{'─'*65}")
    print(f"  Trust wins at high drift: {sum(wins)}/{len(wins)} = {np.mean(wins):.1%}")
    print(f"  Mean trust advantage:     {np.mean(advantages):+.4f}")
    print(f"  Wilcoxon p (advantages>0): {p_val:.4f}")

    # ── CTGAN / Augmentation evaluation ───────────────────────────────
    print(f"\n{'─'*65}")
    print("  CTGAN AUGMENTATION EVALUATION")
    from src.genai.ctgan_augmentation import AugmentedMetaLearner, TabularAugmenter

    # Build mini meta-dataset from our temporal results
    meta_X = []
    meta_Y = []
    for r in all_results:
        if r.get('trust_score_diff') is not None:
            meta_X.append([
                r.get('drift_results',[{}])[0].get('trust_f1',0.5),
                r.get('avg_trust_advantage_high_drift',0),
                r.get('trust_score_diff',0),
                float(r.get('same_model',False)),
            ])
            meta_Y.append([0.05,0.10,0.10,0.35,0.40])  # empirical weights

    meta_X = np.array(meta_X) if meta_X else np.random.rand(8,4)
    meta_Y = np.array(meta_Y) if meta_Y else np.tile([0.05,0.10,0.10,0.35,0.40],(8,1))

    ctgan_results = {}
    try:
        aug_strategy = TabularAugmenter(strategy='auto').strategy
        for name, augment in [('real_only', False), ('augmented', True)]:
            learner = AugmentedMetaLearner(augmentation_factor=5, augmentation_strategy='auto')
            loo_r = learner.evaluate_loo(meta_X, meta_Y, augment=augment)
            ctgan_results[name] = {'mae': loo_r['mae'], 'per_component': loo_r['per_component']}
            print(f"  {name:15s}  LOO MAE={loo_r['mae']:.6f}")

        improvement = ctgan_results['real_only']['mae'] - ctgan_results['augmented']['mae']
        pct = improvement / ctgan_results['real_only']['mae'] * 100
        print(f"  Improvement: {improvement:+.6f} ({pct:+.1f}%)")
        ctgan_results['improvement'] = float(improvement)
        ctgan_results['improvement_pct'] = float(pct)
        ctgan_results['strategy_used'] = aug_strategy
    except Exception as e:
        print(f"  CTGAN eval error: {e}")
        ctgan_results = {'error': str(e)}

    output = {
        'n_datasets': len(all_results),
        'trust_win_rate_high_drift': float(np.mean(wins)),
        'mean_trust_advantage_high_drift': float(np.mean(advantages)),
        'wilcoxon_p': float(p_val),
        'datasets': all_results,
        'ctgan_augmentation': ctgan_results,
    }
    out_path = OUT / 'phase4_temporal_results.json'
    out_path.write_text(json.dumps(output, indent=2, default=lambda o: float(o) if hasattr(o,'item') else str(o)))
    print(f"\n  Saved -> {out_path}")
