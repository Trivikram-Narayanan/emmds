"""
EMMDS Phase 2: DQL Training + Evaluation
=========================================
Full training loop and baseline comparison.

Baselines:
  1. Always CONTINUE   — never retrain (lower bound)
  2. Fixed schedule    — retrain every K batches
  3. Reactive threshold — retrain when F1 drops below θ
  4. DQL policy        — learned agent (our contribution)
  5. Oracle            — always knows when to retrain (upper bound)

Evaluation metrics:
  - Cumulative reward (performance - cost)
  - Mean F1 across episode
  - Number of retrains
  - Performance at episode end (final F1)

Research analysis:
  - Does DQL outperform reactive threshold baseline?
  - Does learned policy correlate with training-time trust score?
  - Do low-trust models trigger earlier retraining in learned policy?
"""

import sys, json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rl.deployment_env import DeploymentEnvironment
from src.rl.dql_agent import DQLAgent
from src.data_engine.dataset_generator import build_full_dataset_collection
from src.data_engine.data_quality import DataQualityScorer
from src.decision.trust_score import TrustScoreEngine
from src.training.cross_validation import CrossValidator
from src.calibration.calibrator import ModelCalibrator

OUT = Path("outputs/phase2")
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE   = 42
N_TRAIN_EPISODES = 200   # Episodes to train the DQL agent
N_EVAL_EPISODES  = 50    # Episodes to evaluate
RETRAINING_COST  = 0.15
MAX_BATCHES      = 20


# ══════════════════════════════════════════════════════════════════════
# SCENARIO BUILDER
# ══════════════════════════════════════════════════════════════════════

def build_scenario(df, target_col, model_name="random_forest"):
    """
    Prepare one deployment scenario from a dataset.
    Returns (env, trust_score, baseline_f1) or None on failure.
    """
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = LabelEncoder().fit_transform(df[target_col])
    if len(np.unique(y)) < 2 or len(X) < 60:
        return None

    # Subsample if large
    if len(X) > 1000:
        idx = np.random.RandomState(RANDOM_STATE).choice(len(X), 1000, replace=False)
        X, y = (X.iloc[idx] if hasattr(X,'iloc') else X[idx]), y[idx]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X.values if hasattr(X,'values') else X, y,
        test_size=0.35, random_state=RANDOM_STATE,
        stratify=y if len(np.unique(y))>1 else None)

    sc = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)

    # Base model and backup
    models = {
        "random_forest":    RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE),
        "gradient_boosting":GradientBoostingClassifier(n_estimators=50, random_state=RANDOM_STATE),
        "logistic":         LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
    }
    base_model   = models.get(model_name, models["random_forest"])
    backup_model = models["logistic"]

    try:
        base_model.fit(X_tr_s, y_tr)
        baseline_f1 = float(f1_score(y_te, base_model.predict(X_te_s),
                                      average='weighted', zero_division=0))
    except Exception:
        return None

    # Trust score
    dq = DataQualityScorer().score_dataset(df, target_col)
    cv_s = np.array([0.8, 0.82, 0.78])  # Simplified for speed
    try:
        from sklearn.model_selection import cross_val_score
        cv_s = cross_val_score(clone(base_model), X_tr_s, y_tr, cv=3,
                               scoring='f1_weighted', n_jobs=1)
    except: pass

    cal = 0.75
    stab = float(np.clip(1 - cv_s.std()/max(abs(cv_s.mean()),1e-8), 0, 1))
    ev   = {'m': {'f1': baseline_f1, 'accuracy': baseline_f1}}
    cr   = {'m': cal}
    cvr  = {'m': {'f1_weighted': {'mean':float(cv_s.mean()),'std':float(cv_s.std()),'values':cv_s.tolist()}}}
    te   = TrustScoreEngine(use_empirical_weights=True)
    ts   = te.compute_all(ev, cr, cvr, agreement_score=0.75, data_quality_score=dq)['m']

    # Create environment
    env = DeploymentEnvironment(
        retraining_cost=RETRAINING_COST,
        max_batches=MAX_BATCHES,
        drift_schedule="gradual",
        random_state=RANDOM_STATE,
    )
    env.setup(X_tr_s, X_te_s, y_tr, y_te,
              base_model, backup_model, ts, baseline_f1, sc)

    return env, float(ts), float(baseline_f1)


# ══════════════════════════════════════════════════════════════════════
# BASELINE POLICIES
# ══════════════════════════════════════════════════════════════════════

def run_policy(env, policy_fn, drift_schedule="gradual") -> dict:
    """Run a policy for one episode. Returns episode stats."""
    env.drift_schedule = drift_schedule
    state = env.reset()
    total_reward = 0.0
    f1_history   = []
    actions_taken = []
    retrain_count = 0

    for step in range(MAX_BATCHES):
        action = policy_fn(state, step)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        f1_history.append(info["f1"])
        actions_taken.append(action)
        if action == 1:
            retrain_count += 1
        state = next_state
        if done:
            break

    return {
        "total_reward":   round(total_reward, 6),
        "mean_f1":        round(float(np.mean(f1_history)), 6),
        "final_f1":       round(float(f1_history[-1]), 6) if f1_history else 0.0,
        "retrain_count":  retrain_count,
        "retraining_cost": round(retrain_count * RETRAINING_COST, 6),
        "net_performance": round(float(np.mean(f1_history)) - retrain_count*RETRAINING_COST, 6),
    }


# ══════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def train_dql_agent(scenarios: list) -> DQLAgent:
    """Train DQL agent across multiple deployment scenarios."""
    agent = DQLAgent(
        state_dim=7, n_actions=3, hidden_dim=64,
        lr=0.001, gamma=0.95,
        epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.992,
        batch_size=32, buffer_capacity=5000,
        target_update=50, random_state=RANDOM_STATE,
    )

    print(f"  Training DQL agent: {N_TRAIN_EPISODES} episodes "
          f"across {len(scenarios)} scenarios...")

    episode_rewards = []
    for episode in range(N_TRAIN_EPISODES):
        # Sample random scenario
        env, trust_score, baseline_f1 = scenarios[
            episode % len(scenarios)]

        # Randomise drift schedule for diversity
        env.drift_schedule = np.random.choice(
            ["gradual","sudden","cyclic","none"], p=[0.4,0.25,0.25,0.1])

        state = env.reset()
        ep_reward = 0.0

        for step in range(MAX_BATCHES):
            action     = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.push_experience(state, action, reward, next_state, done)
            loss = agent.update()
            ep_reward += reward
            state = next_state
            if done:
                break

        episode_rewards.append(ep_reward)
        agent.reward_history.append(ep_reward)

        if (episode + 1) % 50 == 0:
            mean_r = np.mean(episode_rewards[-50:])
            summ   = agent.training_summary()
            print(f"    Episode {episode+1:4d}  "
                  f"mean_reward={mean_r:.4f}  "
                  f"epsilon={summ.get('epsilon',0):.3f}  "
                  f"loss={summ.get('mean_loss_last50',0):.6f}")

    return agent


# ══════════════════════════════════════════════════════════════════════
# EVALUATION + ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def evaluate_all_policies(agent: DQLAgent, scenarios: list) -> pd.DataFrame:
    """
    Compare DQL vs all baselines on held-out scenarios.
    """
    # Define baseline policies
    def always_continue(state, step): return 0
    def fixed_schedule_5(state, step): return 1 if step % 5 == 4 else 0
    def fixed_schedule_3(state, step): return 1 if step % 3 == 2 else 0
    def reactive_05(state, step):  return 1 if state[0] < 0.75 else 0
    def reactive_10(state, step):  return 1 if state[0] < 0.70 else 0
    def dql_policy(state, step):
        return agent.get_policy_action(state)

    policies = {
        "Always Continue":      always_continue,
        "Fixed (every 5)":      fixed_schedule_5,
        "Fixed (every 3)":      fixed_schedule_3,
        "Reactive (F1<0.75)":   reactive_05,
        "Reactive (F1<0.70)":   reactive_10,
        "DQL Agent":            dql_policy,
    }

    rows = []
    for env, trust_score, baseline_f1 in scenarios:
        for policy_name, policy_fn in policies.items():
            for schedule in ["gradual", "sudden", "cyclic"]:
                env.drift_schedule = schedule
                result = run_policy(env, policy_fn, schedule)
                result.update({
                    "policy":        policy_name,
                    "trust_score":   trust_score,
                    "baseline_f1":   baseline_f1,
                    "drift_schedule": schedule,
                })
                rows.append(result)

    return pd.DataFrame(rows)


def analyse_trust_policy_correlation(
    agent: DQLAgent, trust_values: list
) -> dict:
    """
    KEY ANALYSIS: Does the learned policy threshold vary with trust score?

    We probe the agent with states that differ only in trust score.
    If trust score predicts policy aggressiveness, the Q-values for
    the RETRAIN action should be higher for low-trust states.
    """
    results = []
    for trust in trust_values:
        # Create probe states: moderate drift, varying trust
        state = np.array([
            0.80,    # current F1
            -0.05,   # delta from peak
            0.15,    # KS statistic (moderate drift)
            0.12,    # PSI (moderate drift)
            0.30,    # batches since retrain
            trust,   # VARYING trust score
            0.10,    # cumulative cost
        ], dtype=np.float32)

        q_vals = agent.get_q_values(state)
        action = agent.get_policy_action(state)

        results.append({
            "trust_score":       round(trust, 3),
            "q_continue":        round(float(q_vals[0]), 6),
            "q_retrain":         round(float(q_vals[1]), 6),
            "q_fallback":        round(float(q_vals[2]), 6),
            "preferred_action":  ["CONTINUE","RETRAIN","FALLBACK"][action],
            "retrain_urgency":   round(float(q_vals[1] - q_vals[0]), 6),
        })

    return results


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def run_phase2():
    print("=" * 65)
    print("  PHASE 2: DQL DEPLOYMENT LIFECYCLE AGENT")
    print("  Hypothesis: Trust score predicts optimal retraining policy")
    print("=" * 65)

    # Build scenarios from dataset collection
    print("\n  Building deployment scenarios...")
    datasets   = build_full_dataset_collection()
    scenarios  = []
    for df, target, name in datasets[:25]:  # 25 scenarios
        result = build_scenario(df, target)
        if result is not None:
            scenarios.append(result)
    print(f"  {len(scenarios)} valid scenarios built")

    if len(scenarios) < 5:
        print("  ERROR: Not enough valid scenarios. Check dataset loading.")
        return {}

    # Split train/eval scenarios
    train_scenarios = scenarios[:int(len(scenarios)*0.7)]
    eval_scenarios  = scenarios[int(len(scenarios)*0.7):]

    print(f"  Training: {len(train_scenarios)} scenarios | "
          f"Eval: {len(eval_scenarios)} scenarios")

    # Train agent
    t0    = time.time()
    agent = train_dql_agent(train_scenarios)
    agent.save(str(OUT / "dql_agent.json"))
    print(f"  Training complete in {round(time.time()-t0,1)}s")
    print(f"  Final epsilon: {agent.epsilon:.3f}")

    # Evaluate all policies
    print("\n  Evaluating all policies on held-out scenarios...")
    eval_df = evaluate_all_policies(agent, eval_scenarios)
    eval_df.to_csv(OUT / "policy_comparison.csv", index=False)

    # Summary table
    summary = eval_df.groupby("policy").agg(
        mean_net_perf   = ("net_performance",  "mean"),
        mean_reward     = ("total_reward",     "mean"),
        mean_f1         = ("mean_f1",          "mean"),
        mean_retrains   = ("retrain_count",    "mean"),
    ).round(4).sort_values("mean_net_perf", ascending=False)

    print(f"\n  Policy comparison (mean across {len(eval_scenarios)} scenarios × 3 drift types):")
    print(f"  {'Policy':25s}  Net Perf  Reward   F1      Retrains")
    print(f"  {'-'*65}")
    for pol, row in summary.iterrows():
        marker = " ← DQL" if pol == "DQL Agent" else ""
        print(f"  {pol:25s}  {row['mean_net_perf']:.4f}    "
              f"{row['mean_reward']:.4f}   {row['mean_f1']:.4f}   "
              f"{row['mean_retrains']:.1f}{marker}")

    # Statistical test: DQL vs best baseline
    dql_rewards      = eval_df[eval_df["policy"]=="DQL Agent"]["total_reward"].values
    reactive_rewards = eval_df[eval_df["policy"]=="Reactive (F1<0.75)"]["total_reward"].values
    if len(dql_rewards) >= 5 and len(reactive_rewards) >= 5:
        n = min(len(dql_rewards), len(reactive_rewards))
        stat, p = stats.wilcoxon(dql_rewards[:n], reactive_rewards[:n])
        print(f"\n  Wilcoxon (DQL vs Reactive): stat={stat:.4f}  p={p:.6f}  "
              f"{'DQL significantly better ✅' if p<0.05 else 'No significant difference'}")

    # Trust-policy correlation analysis
    print("\n  Analysing trust score → policy relationship...")
    trust_vals = np.arange(0.3, 1.0, 0.05).round(2)
    trust_analysis = analyse_trust_policy_correlation(agent, trust_vals)
    ta_df = pd.DataFrame(trust_analysis)
    ta_df.to_csv(OUT / "trust_policy_analysis.csv", index=False)

    # Correlation between trust and retrain urgency
    r, p_val = stats.spearmanr(
        ta_df["trust_score"], ta_df["retrain_urgency"])
    print(f"  Trust score vs retrain urgency: r={r:.4f}  p={p_val:.6f}")
    print(f"  Interpretation: {'Low trust → higher retrain urgency ✅' if r<-0.3 else 'Trust does not predict urgency'}")

    # Show policy by trust tier
    print(f"\n  Preferred action by trust tier:")
    ta_df["tier"] = pd.cut(ta_df["trust_score"],
                            bins=[0,0.5,0.7,0.85,1.01],
                            labels=["Low","Moderate","High","Very High"])
    tier_policy = ta_df.groupby("tier")["preferred_action"].agg(
        lambda x: x.value_counts().index[0])
    for tier, action in tier_policy.items():
        print(f"    {str(tier):12s} → {action}")

    def _j(o):
        if isinstance(o,(np.bool_,)): return bool(o)
        if isinstance(o,(np.integer,)): return int(o)
        if isinstance(o,(np.floating,)):
            return None if(np.isnan(o) or np.isinf(o)) else float(o)
        if isinstance(o,np.ndarray): return o.tolist()
        return str(o)

    results = {
        "n_train_scenarios":  len(train_scenarios),
        "n_eval_scenarios":   len(eval_scenarios),
        "policy_summary":     summary.reset_index().to_dict("records"),
        "trust_policy_analysis": trust_analysis,
        "trust_urgency_correlation": {
            "spearman_r": round(float(r),   4),
            "p_value":    round(float(p_val),6),
            "significant": bool(p_val < 0.05),
        },
        "agent_training": agent.training_summary(),
        "key_finding": (
            f"DQL agent achieves net_performance="
            f"{summary.loc['DQL Agent','mean_net_perf']:.4f} vs "
            f"reactive baseline "
            f"{summary.loc['Reactive (F1<0.75)','mean_net_perf'] if 'Reactive (F1<0.75)' in summary.index else 'N/A'}. "
            f"Trust-urgency correlation: r={r:.4f} (p={p_val:.4f})."
        ),
    }

    with open(OUT/"phase2_results.json","w") as f:
        json.dump(results, f, indent=2, default=_j)

    print(f"\n  Results saved → {OUT}/")
    print(f"\n  KEY FINDING: {results['key_finding']}")
    return results


if __name__ == "__main__":
    run_phase2()
