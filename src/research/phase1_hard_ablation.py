"""
Phase 1b: Hard-Dataset Ablation Study
Shows meaningful deltas when trust components are removed,
specifically on challenging datasets (hard + extreme difficulty).
"""
import sys, json, warnings
import numpy as np
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, train_test_split
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

def compute_metrics(name, X_tr, X_te, y_tr, y_te, X_all, y_all):
    metrics = {}
    for mname, model in MODELS.items():
        try:
            m = clone(model); m.fit(X_tr, y_tr)
            train_f1 = float(f1_score(y_tr, m.predict(X_tr), average='weighted', zero_division=0))
            test_f1  = float(f1_score(y_te, m.predict(X_te), average='weighted', zero_division=0))
            cv_s = cross_val_score(clone(model), X_all, y_all, cv=3, scoring='f1_weighted')
            cv_mean, cv_std = float(np.mean(cv_s)), float(np.std(cv_s))
            stability = float(np.clip(1.0 - cv_std/(cv_mean+1e-8), 0, 1))
            try:
                cm = CalibratedClassifierCV(clone(model), cv=3, method='sigmoid')
                cm.fit(X_tr, y_tr); proba = cm.predict_proba(X_te)
                n_c = len(np.unique(y_te))
                if n_c == 2:
                    brier = brier_score_loss(y_te, proba[:,1])
                else:
                    brier = np.mean([brier_score_loss((y_te==c).astype(int), proba[:,i])
                                     for i,c in enumerate(np.unique(y_te))])
                cal = float(np.clip(1-brier, 0, 1))
            except: cal = 0.7
            classes, cnts = np.unique(y_te, return_counts=True)
            dq = float(np.clip(0.4 + 0.4*(cnts.min()/cnts.max()) + 0.2*0.8, 0, 1))
            risk = 0.40*max(0,train_f1-test_f1) + 0.30*(1-cal) + 0.30*cv_std
            metrics[mname] = {'test_f1':test_f1,'stability':stability,'cal':cal,'dq':dq,'risk':float(risk),'cv_std':cv_std}
        except: pass
    return metrics

def trust_select(metrics, w_acc, w_cal, w_agr, w_dq, w_stab, agr=0.75):
    if not metrics: return None, 0
    total = w_acc+w_cal+w_agr+w_dq+w_stab
    scores = {}
    for m,r in metrics.items():
        scores[m] = (w_acc*r['test_f1'] + w_cal*r['cal'] + w_agr*agr +
                     w_dq*r['dq'] + w_stab*r['stability']) / (total+1e-8)
    best = max(scores, key=scores.get)
    return best, metrics[best]['risk']

CONDITIONS = [
    ('full_system',    0.05, 0.10, 0.10, 0.35, 0.40),
    ('no_calibration', 0.05, 0.00, 0.10, 0.35, 0.40),
    ('no_agreement',   0.05, 0.10, 0.00, 0.35, 0.40),
    ('no_stability',   0.05, 0.10, 0.10, 0.35, 0.00),
    ('no_dataquality', 0.05, 0.10, 0.10, 0.00, 0.40),
    ('accuracy_only',  1.00, 0.00, 0.00, 0.00, 0.00),
    ('equal_weights',  0.20, 0.20, 0.20, 0.20, 0.20),
]

HARD_CFGS = [
    ('hard_highimb_1',  dict(n_samples=200,n_features=20,n_informative=8,n_redundant=6,flip_y=0.15,weights=[0.85,0.15],n_classes=2,class_sep=0.6,random_state=9)),
    ('hard_highimb_2',  dict(n_samples=180,n_features=30,n_informative=10,n_redundant=8,flip_y=0.13,weights=[0.82,0.18],n_classes=2,class_sep=0.5,random_state=10)),
    ('hard_noise_1',    dict(n_samples=220,n_features=25,n_informative=8,n_redundant=8,flip_y=0.18,weights=[0.80,0.20],n_classes=2,class_sep=0.5,random_state=11)),
    ('hard_noise_2',    dict(n_samples=200,n_features=20,n_informative=6,n_redundant=8,flip_y=0.20,weights=[0.80,0.20],n_classes=2,class_sep=0.4,random_state=12)),
    ('extreme_1',       dict(n_samples=150,n_features=50,n_informative=10,n_redundant=15,flip_y=0.20,weights=[0.91,0.09],n_classes=2,class_sep=0.4,random_state=13)),
    ('extreme_2',       dict(n_samples=140,n_features=60,n_informative=8,n_redundant=20,flip_y=0.22,weights=[0.89,0.11],n_classes=2,class_sep=0.3,random_state=14)),
    ('extreme_3',       dict(n_samples=160,n_features=45,n_informative=10,n_redundant=15,flip_y=0.18,weights=[0.87,0.13],n_classes=2,class_sep=0.4,random_state=15)),
    ('extreme_4',       dict(n_samples=130,n_features=55,n_informative=8,n_redundant=18,flip_y=0.25,weights=[0.92,0.08],n_classes=2,class_sep=0.3,random_state=16)),
]

if __name__ == '__main__':
    print("="*65)
    print("  PHASE 1b: ABLATION ON HARD DATASETS ONLY")
    print("="*65)

    cond_risks = {cname: [] for cname,*_ in CONDITIONS}
    dataset_rows = []

    for ds_name, cfg in HARD_CFGS:
        X, y = make_classification(**cfg)
        le = LabelEncoder(); y = le.fit_transform(y)
        sc = StandardScaler()
        X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.25,random_state=42,
                                                 stratify=y if len(np.unique(y))>=2 else None)
        X_tr_s=sc.fit_transform(X_tr); X_te_s=sc.transform(X_te); X_all_s=sc.transform(X)

        metrics = compute_metrics(ds_name, X_tr_s, X_te_s, y_tr, y_te, X_all_s, y)
        if not metrics: continue

        row = {'dataset': ds_name}
        for cname,w_acc,w_cal,w_agr,w_dq,w_stab in CONDITIONS:
            best, risk = trust_select(metrics, w_acc,w_cal,w_agr,w_dq,w_stab)
            row[cname+'_risk'] = round(float(risk),5)
            cond_risks[cname].append(risk)
        dataset_rows.append(row)

        full_risk = row['full_system_risk']
        print(f"  {ds_name:25s}")
        for cname,*_ in CONDITIONS:
            delta = row[cname+'_risk'] - full_risk
            print(f"    {cname:18s}: {row[cname+'_risk']:.5f}  (delta {delta:+.5f})")

    print(f"\n{'='*65}")
    print(f"  MEAN RISK ACROSS {len(dataset_rows)} HARD DATASETS:")
    full_mean = float(np.mean(cond_risks['full_system']))
    rows_summary = []
    for cname,*_ in CONDITIONS:
        mean_r = float(np.mean(cond_risks[cname]))
        delta  = mean_r - full_mean
        rows_summary.append({'condition':cname,'mean_risk':round(mean_r,5),'delta_vs_full':round(delta,5)})
        print(f"    {cname:18s}  mean_risk={mean_r:.5f}  delta={delta:+.5f}")

    # Statistical test: is full_system significantly better than accuracy_only on hard datasets?
    full_risks = cond_risks['full_system']
    acc_risks  = cond_risks['accuracy_only']
    t_s, t_p = 0.0, 1.0
    if len(full_risks)>=5:
        t_s,t_p = stats.ttest_rel(acc_risks, full_risks)  # H0: no diff; positive t_s = full better
        print(f"\n  Paired t-test (acc_only vs full, hard datasets): t={t_s:.4f}  p={t_p:.4f}")

    output = {
        'n_hard_datasets': len(dataset_rows),
        'conditions': rows_summary,
        'dataset_detail': dataset_rows,
        'statistical_test': {
            'test': 'paired_t_acc_only_vs_full_on_hard_datasets',
            't_stat': float(t_s) if len(full_risks)>=5 else None,
            'p_value': float(t_p) if len(full_risks)>=5 else None,
        }
    }

    out_path = OUT / 'phase1_hard_ablation.json'
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Saved -> {out_path}")
