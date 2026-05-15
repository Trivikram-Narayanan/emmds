"""
Phase 2: Formal RL Deployment Evaluation
=========================================
Evaluates the DQL agent for deployment timing across:
  - Multiple datasets (6 synthetic, 2 real-ish)
  - 3 drift schedules: gradual, sudden, cyclic
  - 4 baselines: always_continue, periodic_retrain, threshold_f1, random

KEY HYPOTHESIS: The learned Q(retrain | state) increases faster as a function
of training-time trust score. Low-trust models should be retrained sooner.
"""
import sys, json, warnings
import numpy as np
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from scipy import stats
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

OUT = Path('outputs/research'); OUT.mkdir(parents=True, exist_ok=True)

# ── Import existing DQL components ────────────────────────────────────
from src.rl.dql_agent import DQLAgent
from src.rl.deployment_env import DeploymentEnvironment


def compute_trust(model, X_tr, X_te, y_tr, y_te, X_all, y_all, dq=0.75, agr=0.75):
    """Quick trust score for a trained model."""
    try:
        test_f1 = float(f1_score(y_te, model.predict(X_te), average='weighted', zero_division=0))
        cv_s = cross_val_score(clone(model), X_all, y_all, cv=3, scoring='f1_weighted')
        cv_mean, cv_std = float(np.mean(cv_s)), float(np.std(cv_s))
        stability = float(np.clip(1 - cv_std/(cv_mean+1e-8), 0, 1))
        try:
            cm = CalibratedClassifierCV(clone(model), cv=3)
            cm.fit(X_tr, y_tr)
            p = cm.predict_proba(X_te)
            n_c = len(np.unique(y_te))
            if n_c == 2:
                brier = brier_score_loss(y_te, p[:,1])
            else:
                brier = np.mean([brier_score_loss((y_te==c).astype(int),p[:,i])
                                 for i,c in enumerate(np.unique(y_te))])
            cal = float(np.clip(1-brier, 0, 1))
        except: cal = 0.7
        trust = 0.05*test_f1 + 0.10*cal + 0.10*agr + 0.35*dq + 0.40*stability
        return float(np.clip(trust, 0, 1)), stability, test_f1
    except: return 0.5, 0.5, 0.5


def train_dql_agent(env, n_episodes=80, random_state=42):
    """Train a DQL agent on a given environment."""
    agent = DQLAgent(state_dim=7, n_actions=3, hidden_dim=64, lr=0.001,
                     gamma=0.95, epsilon_start=1.0, epsilon_min=0.05,
                     epsilon_decay=0.97, batch_size=32, random_state=random_state)
    episode_rewards = []
    for ep in range(n_episodes):
        state = env.reset()
        total_r = 0.0
        done = False
        steps = 0
        while not done and steps < 25:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.push_experience(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            total_r += reward
            steps += 1
        episode_rewards.append(total_r)
        agent.reward_history.append(total_r)
    return agent, episode_rewards


def evaluate_policy(env, policy_fn, n_eval=20):
    """Evaluate a policy function over n_eval episodes. Returns mean total reward."""
    rewards = []
    retrain_counts = []
    for _ in range(n_eval):
        state = env.reset()
        total_r = 0.0; done = False; steps = 0
        while not done and steps < 25:
            action = policy_fn(state, steps)
            state, reward, done, info = env.step(action)
            total_r += reward; steps += 1
        rewards.append(total_r)
        retrain_counts.append(env._retrain_count)
    return float(np.mean(rewards)), float(np.std(rewards)), float(np.mean(retrain_counts))


# Baseline policies
def always_continue(state, step): return 0
def periodic_retrain(state, step, period=7): return 1 if step > 0 and step % period == 0 else 0
def threshold_f1(state, step, threshold=0.75): return 1 if state[0] < threshold else 0
def random_policy(state, step): return np.random.randint(3)


DATASET_CONFIGS = [
    # (name, n, n_feat, flip_y, weights, trust_expected)
    ('clean_balanced',  600, 15, 0.02, None,           'high'),
    ('noisy_balanced',  400, 20, 0.15, None,           'medium'),
    ('imbalanced',      400, 20, 0.05, [0.80,0.20],    'medium'),
    ('small_noisy',     200, 25, 0.18, [0.82,0.18],    'low'),
    ('extreme_hard',    150, 40, 0.22, [0.89,0.11],    'very_low'),
    ('moderate',        500, 20, 0.08, [0.70,0.30],    'medium'),
]

DRIFT_SCHEDULES = ['gradual', 'sudden', 'cyclic']


if __name__ == '__main__':
    print("="*65)
    print("  PHASE 2: FORMAL RL DEPLOYMENT EVALUATION")
    print("  DQL Agent vs 4 Baselines across 6 Datasets x 3 Drift Schedules")
    print("="*65)

    all_results = []
    trust_qvalue_data = []  # For trust-dependency analysis

    for ds_name, n_samp, n_feat, flip_y, weights, trust_expected in DATASET_CONFIGS:
        X, y = make_classification(n_samples=n_samp+200, n_features=n_feat,
                                   n_informative=max(5,n_feat//3), n_redundant=n_feat//5,
                                   flip_y=flip_y, weights=weights, n_classes=2,
                                   class_sep=0.8, random_state=42)
        le = LabelEncoder(); y = le.fit_transform(y)
        sc = StandardScaler()
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=200, random_state=42)
        X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)
        X_all_s = sc.transform(X); y_all = y.copy()

        base_model   = RandomForestClassifier(n_estimators=30, random_state=42)
        backup_model = LogisticRegression(max_iter=500, random_state=42)
        base_model.fit(X_tr_s, y_tr); backup_model.fit(X_tr_s, y_tr)

        baseline_f1 = float(f1_score(y_te, base_model.predict(X_te_s), average='weighted', zero_division=0))
        trust_score, stability, test_f1 = compute_trust(base_model, X_tr_s, X_te_s, y_tr, y_te, X_all_s, y_all)

        print(f"\n  Dataset: {ds_name:20s}  trust={trust_score:.3f}  f1={test_f1:.3f}  [{trust_expected}]")

        for drift in DRIFT_SCHEDULES:
            env = DeploymentEnvironment(retraining_cost=0.15, max_batches=20,
                                        batch_size=80, drift_schedule=drift, random_state=42)
            env.setup(X_tr_s, X_te_s, y_tr, y_te, base_model, backup_model,
                      trust_score=trust_score, baseline_f1=baseline_f1, scaler=sc)

            # Train DQL agent
            agent, train_rewards = train_dql_agent(env, n_episodes=80)

            # Define DQL policy using trained agent
            def dql_policy(state, step, _agent=agent):
                return _agent.get_policy_action(np.array(state, dtype=np.float32))

            # Evaluate all policies
            dql_r,  dql_std,  dql_rc  = evaluate_policy(env, dql_policy, n_eval=20)
            cont_r, cont_std, cont_rc = evaluate_policy(env, always_continue, n_eval=20)
            per_r,  per_std,  per_rc  = evaluate_policy(env, periodic_retrain, n_eval=20)
            thr_r,  thr_std,  thr_rc  = evaluate_policy(env, threshold_f1, n_eval=20)
            rnd_r,  rnd_std,  rnd_rc  = evaluate_policy(env, random_policy, n_eval=20)

            # Q-value analysis: retrain Q-value as function of trust at step 5
            env.reset()
            state_5 = np.array([baseline_f1*0.9, -0.1, 0.3, 0.2,
                                 5/20, trust_score, 0.1], dtype=np.float32)
            q_vals = agent.get_q_values(state_5)
            q_continue, q_retrain = float(q_vals[0]), float(q_vals[1])

            result = {
                'dataset': ds_name, 'drift': drift,
                'trust_score': float(trust_score), 'trust_label': trust_expected,
                'baseline_f1': float(baseline_f1),
                'rewards': {
                    'dql': float(dql_r), 'always_continue': float(cont_r),
                    'periodic': float(per_r), 'threshold_f1': float(thr_r), 'random': float(rnd_r),
                },
                'dql_vs_best_baseline': float(dql_r - max(cont_r, per_r, thr_r)),
                'dql_wins': bool(dql_r >= max(cont_r, per_r, thr_r, rnd_r)),
                'q_retrain_minus_continue': float(q_retrain - q_continue),
                'retrain_counts': {'dql': float(dql_rc), 'periodic': float(per_rc)},
            }
            all_results.append(result)

            trust_qvalue_data.append({
                'trust': float(trust_score),
                'q_retrain_advantage': float(q_retrain - q_continue),
            })

            marker = '[WIN]' if result['dql_wins'] else '     '
            print(f"    [{drift:8s}]  DQL={dql_r:.3f}  cont={cont_r:.3f}  "
                  f"per={per_r:.3f}  thr={thr_r:.3f}  D={result['dql_vs_best_baseline']:+.3f}  {marker}")

    # ── Analysis ──────────────────────────────────────────────────────
    dql_wins = [r['dql_wins'] for r in all_results]
    dql_deltas = [r['dql_vs_best_baseline'] for r in all_results]

    # Trust-dependency: does higher trust -> lower Q(retrain) advantage?
    trust_vals = [d['trust'] for d in trust_qvalue_data]
    q_adv_vals = [d['q_retrain_advantage'] for d in trust_qvalue_data]
    sp_r, sp_p = stats.spearmanr(trust_vals, q_adv_vals)

    print(f"\n{'─'*65}")
    print(f"  SUMMARY:")
    print(f"  DQL win rate:        {np.mean(dql_wins):.1%}  ({sum(dql_wins)}/{len(dql_wins)})")
    print(f"  Mean reward delta:   {np.mean(dql_deltas):+.4f}")
    print(f"  Trust-Q correlation: Spearman r={sp_r:.4f}  p={sp_p:.4f}")
    print(f"  (negative r = lower trust -> higher Q(retrain) advantage)")

    output = {
        'n_scenarios': len(all_results),
        'dql_win_rate': float(np.mean(dql_wins)),
        'mean_reward_delta_vs_best_baseline': float(np.mean(dql_deltas)),
        'trust_retrain_spearman': {'r': float(sp_r), 'p': float(sp_p)},
        'results': all_results,
        'trust_qvalue_data': trust_qvalue_data,
    }
    out_path = OUT / 'phase2_rl_results.json'
    out_path.write_text(json.dumps(output, indent=2, default=lambda o: float(o) if hasattr(o,'item') else str(o)))
    print(f"  Saved -> {out_path}")
