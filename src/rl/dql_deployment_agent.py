"""
EMMDS Phase 2: DQL Deployment Lifecycle Agent
==============================================
Research Question:
  "Is the optimal retraining threshold predictable from training-time
   trust scores, and does a DQL agent that incorporates trust signals
   learn better retraining policies than baselines?"

Design:
  Environment:  simulates a deployed model receiving data batches with
                controlled covariate shift injected over time
  State (7-dim): [current_f1, rolling_f1_delta, ks_mean, psi_mean,
                  batches_since_retrain, trust_at_training, degradation_from_peak]
  Actions:       0=CONTINUE  1=RETRAIN  2=SWITCH_TO_BACKUP
  Reward:        current_f1 - λ*retrain_cost - μ*switch_cost

  DQL: tabular Q-learning with discretised state (no neural net needed
       for 7-dim state — tabular Q-table is more interpretable and
       produces clearer policy analysis)

  Key novelty: trust_at_training is in the state vector.
  We measure whether Q-values for RETRAIN are systematically higher
  for states where trust_at_training is low — this would confirm
  that trust score is a deployment lifecycle parameter, not just
  a selection criterion.

Baselines compared:
  B1: Fixed schedule (retrain every N batches)
  B2: Reactive threshold (retrain when F1 drops by X%)
  B3: DQL without trust in state
  B4: DQL with trust in state (our proposal)
  Oracle: retrain at perfect timing (knows future performance)
"""

import sys, warnings, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from scipy import stats

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

OUT = Path("outputs/phase2")
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE  = 42
N_BATCHES     = 30          # batches per deployment episode
BATCH_SIZE    = 80          # samples per batch
RETRAIN_COST  = 0.05        # λ: F1 units equivalent
SWITCH_COST   = 0.02        # μ: F1 units equivalent
SHIFT_START   = 8           # batch at which drift begins
SHIFT_RATE    = 0.08        # σ per batch after SHIFT_START


# ══════════════════════════════════════════════════════════════════════
# DEPLOYMENT ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════

class DeploymentEnv:
    """
    Simulates a deployed ML model receiving streaming data with drift.

    Episode:
      - Model is trained on clean in-distribution data
      - Data arrives in batches; after SHIFT_START, covariate shift
        is injected at SHIFT_RATE σ per batch
      - Agent observes state, takes action, receives reward
      - Episode ends after N_BATCHES

    Trust score at training time is passed in as a fixed state feature.
    This is the key design decision: the agent can learn to use
    training-time trust as a prior for deployment decisions.
    """

    def __init__(
        self,
        X_train:       np.ndarray,
        y_train:       np.ndarray,
        X_test_pool:   np.ndarray,
        y_test_pool:   np.ndarray,
        base_model,
        trust_at_training: float,
        backup_model       = None,
        shift_start:   int   = SHIFT_START,
        shift_rate:    float = SHIFT_RATE,
        n_batches:     int   = N_BATCHES,
        batch_size:    int   = BATCH_SIZE,
        seed:          int   = RANDOM_STATE,
    ):
        self.X_train          = X_train
        self.y_train          = y_train
        self.X_pool           = X_test_pool
        self.y_pool           = y_test_pool
        self.base_model_class = clone(base_model)
        self.backup_model     = clone(backup_model) if backup_model else clone(base_model)
        self.trust            = trust_at_training
        self.shift_start      = shift_start
        self.shift_rate       = shift_rate
        self.n_batches        = n_batches
        self.batch_size       = batch_size
        self.rng              = np.random.RandomState(seed)

        # Fitted scaler from training data
        self.scaler           = StandardScaler().fit(X_train)

        # State variables (reset per episode)
        self._reset_state()

    def reset(self) -> np.ndarray:
        self._reset_state()
        return self._get_state()

    def _reset_state(self):
        # Fit initial model
        self.current_model = clone(self.base_model_class)
        self.current_model.fit(self.scaler.transform(self.X_train), self.y_train)

        self.backup_model_fitted = clone(self.backup_model)
        self.backup_model_fitted.fit(self.scaler.transform(self.X_train), self.y_train)

        # Track peak F1
        init_f1 = self._eval_model(self.current_model, self.X_pool, self.y_pool, shift=0.0)
        self.baseline_f1       = init_f1
        self.peak_f1           = init_f1
        self.current_f1        = init_f1
        self.f1_history        = [init_f1]
        self.batch_num         = 0
        self.batches_since_retrain = 0
        self.total_reward      = 0.0
        self.retrain_count     = 0
        self.done              = False
        self.history           = []

    def step(self, action: int) -> tuple:
        """
        Take action, advance one batch.
        Returns: (next_state, reward, done, info)

        Actions: 0=CONTINUE, 1=RETRAIN, 2=SWITCH_TO_BACKUP
        """
        if self.done:
            raise RuntimeError("Episode finished. Call reset().")

        self.batch_num += 1
        self.batches_since_retrain += 1

        # Current shift magnitude
        shift_mag = max(0.0, (self.batch_num - self.shift_start) * self.shift_rate)

        # Evaluate current model on shifted batch
        f1 = self._eval_model(self.current_model, self.X_pool, self.y_pool, shift_mag)
        self.peak_f1 = max(self.peak_f1, f1)
        self.current_f1 = f1
        self.f1_history.append(f1)

        # Compute reward
        reward = f1  # base reward = current performance

        if action == 1:  # RETRAIN
            # Retrain on recent (slightly shifted) data
            X_recent = self._get_shifted_batch(self.X_train, shift_mag * 0.5)
            self.current_model = clone(self.base_model_class)
            self.current_model.fit(self.scaler.transform(X_recent), self.y_train)
            self.batches_since_retrain = 0
            self.retrain_count += 1
            reward -= RETRAIN_COST

        elif action == 2:  # SWITCH TO BACKUP
            self.current_model = self.backup_model_fitted
            reward -= SWITCH_COST

        self.total_reward += reward
        self.done = self.batch_num >= self.n_batches

        info = {
            'batch':        self.batch_num,
            'f1':           f1,
            'shift_mag':    shift_mag,
            'action':       action,
            'reward':       reward,
            'retrain_count': self.retrain_count,
        }
        self.history.append(info)

        return self._get_state(), reward, self.done, info

    def _eval_model(self, model, X_pool, y_pool, shift: float) -> float:
        """Evaluate model on a shifted sample."""
        idx   = self.rng.choice(len(X_pool), min(self.batch_size, len(X_pool)), replace=False)
        X_b   = X_pool[idx].copy()
        y_b   = y_pool[idx]
        X_shifted = self._get_shifted_batch(X_b, shift)
        try:
            preds = model.predict(self.scaler.transform(X_shifted))
            return float(f1_score(y_b, preds, average='weighted', zero_division=0))
        except Exception:
            return 0.0

    def _get_shifted_batch(self, X: np.ndarray, shift_mag: float) -> np.ndarray:
        """Apply covariate shift to a batch."""
        if shift_mag == 0.0:
            return X.copy()
        feature_stds  = self.X_train.std(axis=0)
        shift_dir     = self.rng.choice([-1, 1], size=X.shape[1])
        return X + shift_mag * feature_stds * shift_dir

    def _get_state(self) -> np.ndarray:
        """
        7-dimensional state vector:
          [current_f1, rolling_delta, ks_mean, psi_mean,
           batches_since_retrain, trust_at_training, degradation]

        trust_at_training is constant throughout the episode —
        it is the training-time signal. The agent can learn to
        use it as a prior for how aggressive its retraining policy should be.
        """
        rolling_delta = 0.0
        if len(self.f1_history) >= 3:
            rolling_delta = self.f1_history[-1] - np.mean(self.f1_history[-4:-1])

        degradation = self.peak_f1 - self.current_f1

        # Simplified drift signal (KS and PSI approximation from F1 trend)
        if len(self.f1_history) >= 5:
            recent_trend = np.polyfit(range(5), self.f1_history[-5:], 1)[0]
            ks_approx    = float(np.clip(abs(recent_trend) * 10, 0, 1))
            psi_approx   = float(np.clip(degradation * 2, 0, 1))
        else:
            ks_approx = psi_approx = 0.0

        return np.array([
            self.current_f1,
            rolling_delta,
            ks_approx,
            psi_approx,
            min(self.batches_since_retrain / self.n_batches, 1.0),
            self.trust,
            degradation,
        ], dtype=np.float32)

    def get_episode_summary(self) -> dict:
        return {
            'total_reward':      round(self.total_reward, 6),
            'mean_f1':           round(float(np.mean([h['f1'] for h in self.history])), 6),
            'final_f1':          round(self.current_f1, 6),
            'retrain_count':     self.retrain_count,
            'trust_at_training': round(self.trust, 6),
            'baseline_f1':       round(self.baseline_f1, 6),
        }


# ══════════════════════════════════════════════════════════════════════
# TABULAR Q-LEARNING AGENT
# ══════════════════════════════════════════════════════════════════════

class DQLAgent:
    """
    Tabular Q-learning with discretised state.

    Discretisation bins per dimension:
      F1 and delta: 5 bins each
      KS, PSI, degradation: 4 bins each
      Batches_since_retrain: 4 bins
      Trust: 3 bins (low/medium/high)

    Total states: 5×5×4×4×4×3×4 = 19,200 (manageable tabular)

    Why tabular and not neural?
      - 7-dim state is small enough for tabular
      - Tabular Q-values are directly interpretable
      - We can examine Q(s,RETRAIN) as a function of trust — this is
        the key analysis for our research question
      - Neural DQL would be appropriate for higher-dimensional states
    """

    N_ACTIONS = 3  # CONTINUE, RETRAIN, SWITCH

    def __init__(
        self,
        use_trust:   bool  = True,
        alpha:       float = 0.1,
        gamma:       float = 0.95,
        epsilon:     float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay:float = 0.995,
    ):
        self.use_trust     = use_trust
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q             = defaultdict(lambda: np.zeros(self.N_ACTIONS))
        self.steps         = 0
        self.episode_rewards = []

    def _discretise(self, state: np.ndarray) -> tuple:
        """Bin each state dimension into discrete levels."""
        f1, delta, ks, psi, batches, trust, degradation = state

        b_f1    = int(np.clip(f1 * 5, 0, 4))
        b_delta = int(np.clip((delta + 0.2) / 0.08, 0, 4))
        b_ks    = int(np.clip(ks * 4, 0, 3))
        b_psi   = int(np.clip(psi * 4, 0, 3))
        b_bat   = int(np.clip(batches * 4, 0, 3))
        b_degrad= int(np.clip(degradation * 8, 0, 3))

        if self.use_trust:
            b_trust = 0 if trust < 0.60 else (1 if trust < 0.80 else 2)
            return (b_f1, b_delta, b_ks, b_psi, b_bat, b_trust, b_degrad)
        else:
            return (b_f1, b_delta, b_ks, b_psi, b_bat, b_degrad)

    def act(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.N_ACTIONS)
        s = self._discretise(state)
        return int(np.argmax(self.Q[s]))

    def update(self, state, action, reward, next_state, done):
        """Bellman update."""
        s  = self._discretise(state)
        s_ = self._discretise(next_state)

        target = reward + (0 if done else self.gamma * np.max(self.Q[s_]))
        self.Q[s][action] += self.alpha * (target - self.Q[s][action])

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.steps += 1

    def get_retrain_q_by_trust(self) -> dict:
        """
        Extract mean Q(s, RETRAIN) grouped by trust tier.
        This is the key analysis: if trust predicts Q(RETRAIN),
        trust score is a deployment lifecycle parameter.
        """
        trust_bins = {0: 'low (<0.60)', 1: 'medium (0.60-0.80)', 2: 'high (>0.80)'}
        q_by_trust = defaultdict(list)

        if self.use_trust:
            for state_key, q_vals in self.Q.items():
                trust_bin = state_key[5]
                q_by_trust[trust_bins[trust_bin]].append(q_vals[1])  # Q(RETRAIN)

        return {
            k: {'mean_q_retrain': round(float(np.mean(v)), 4),
                'n_states':       len(v)}
            for k, v in q_by_trust.items() if v
        }


# ══════════════════════════════════════════════════════════════════════
# BASELINE POLICIES
# ══════════════════════════════════════════════════════════════════════

def run_fixed_schedule(env: DeploymentEnv, retrain_every: int = 8) -> dict:
    """Baseline 1: Retrain every N batches regardless of drift."""
    state = env.reset()
    while not env.done:
        action = 1 if (env.batch_num > 0 and env.batch_num % retrain_every == 0) else 0
        state, _, done, _ = env.step(action)
    return env.get_episode_summary()


def run_reactive(env: DeploymentEnv, threshold: float = 0.08) -> dict:
    """Baseline 2: Retrain when F1 drops by threshold from peak."""
    state = env.reset()
    while not env.done:
        degradation = env.peak_f1 - env.current_f1
        action = 1 if degradation > threshold else 0
        state, _, done, _ = env.step(action)
    return env.get_episode_summary()


def run_dql(env: DeploymentEnv, agent: DQLAgent,
            training: bool = False) -> dict:
    """Run DQL agent for one episode (training or evaluation)."""
    state = env.reset()
    ep_reward = 0.0

    while not env.done:
        action       = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        if training:
            agent.update(state, action, reward, next_state, done)
        ep_reward += reward
        state = next_state

    if training:
        agent.episode_rewards.append(ep_reward)

    return env.get_episode_summary()


def run_oracle(env: DeploymentEnv) -> dict:
    """
    Oracle baseline: retrain whenever F1 has dropped and will continue dropping.
    Requires knowledge of future — used as upper bound.
    """
    state = env.reset()
    # Simulate forward to know when to retrain
    # Simple oracle: retrain at peak shift onset
    while not env.done:
        action = 1 if env.batch_num == env.shift_start else 0
        state, _, done, _ = env.step(action)
    return env.get_episode_summary()


# ══════════════════════════════════════════════════════════════════════
# TRAINING AND EVALUATION
# ══════════════════════════════════════════════════════════════════════

def build_envs(datasets_with_trust: list) -> list:
    """
    Build deployment environments from (df, target, trust_score) triples.
    Each dataset becomes one environment.
    """
    envs = []
    base_model   = RandomForestClassifier(n_estimators=50, random_state=42)
    backup_model = LogisticRegression(max_iter=500, random_state=42)

    for df, target, trust_score in datasets_with_trust:
        X = df.drop(columns=[target]).select_dtypes(include=[float, int])
        y = LabelEncoder().fit_transform(df[target])

        if len(np.unique(y)) < 2 or len(X) < 100:
            continue

        # Subsample for speed
        if len(X) > 1000:
            idx = np.random.RandomState(42).choice(len(X), 1000, replace=False)
            X, y = X.iloc[idx].values, y[idx]
        else:
            X = X.values

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.4, random_state=42,
            stratify=y if len(np.unique(y)) > 1 else None
        )

        env = DeploymentEnv(
            X_train=X_tr, y_train=y_tr,
            X_test_pool=X_te, y_test_pool=y_te,
            base_model=base_model,
            trust_at_training=trust_score,
            backup_model=backup_model,
        )
        envs.append(env)

    return envs


def train_agent(
    agent:     DQLAgent,
    train_envs: list,
    n_episodes: int = 200,
    verbose:    bool = True,
) -> DQLAgent:
    """Train DQL agent across all training environments."""
    ep = 0
    while ep < n_episodes:
        for env in train_envs:
            run_dql(env, agent, training=True)
            ep += 1
            if ep >= n_episodes:
                break

        if verbose and ep % 50 == 0:
            recent_reward = np.mean(agent.episode_rewards[-20:]) if len(agent.episode_rewards) >= 20 else 0
            print(f"    Episode {ep:4d}  ε={agent.epsilon:.3f}  "
                  f"recent_reward={recent_reward:.4f}  "
                  f"Q_states={len(agent.Q)}")

    return agent


def evaluate_all_policies(
    test_envs:      list,
    agent_with_trust: DQLAgent,
    agent_no_trust:   DQLAgent,
    n_eval_runs:    int = 5,
) -> pd.DataFrame:
    """
    Evaluate all policies across test environments.
    Each environment is run n_eval_runs times (different random seeds).
    """
    results = []

    for env_idx, env in enumerate(test_envs):
        for run in range(n_eval_runs):
            # Different seed per run
            env.rng = np.random.RandomState(env_idx * 100 + run)

            # All policies on same environment
            for policy_name, policy_fn in [
                ('Fixed_N8',        lambda e: run_fixed_schedule(e, retrain_every=8)),
                ('Fixed_N5',        lambda e: run_fixed_schedule(e, retrain_every=5)),
                ('Reactive_5pct',   lambda e: run_reactive(e, threshold=0.05)),
                ('Reactive_10pct',  lambda e: run_reactive(e, threshold=0.10)),
                ('DQL_no_trust',    lambda e: run_dql(e, agent_no_trust, training=False)),
                ('DQL_with_trust',  lambda e: run_dql(e, agent_with_trust, training=False)),
                ('Oracle',          lambda e: run_oracle(e)),
            ]:
                summary = policy_fn(env)
                results.append({
                    'env_idx':           env_idx,
                    'run':               run,
                    'policy':            policy_name,
                    'mean_f1':           summary['mean_f1'],
                    'final_f1':          summary['final_f1'],
                    'total_reward':      summary['total_reward'],
                    'retrain_count':     summary['retrain_count'],
                    'trust_at_training': summary['trust_at_training'],
                    'baseline_f1':       summary['baseline_f1'],
                })

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════
# MAIN PHASE 2 RUNNER
# ══════════════════════════════════════════════════════════════════════

def run_phase2(datasets_with_trust: list) -> dict:
    """
    Full Phase 2 experiment.

    Args:
        datasets_with_trust: list of (df, target_col, trust_score)

    Returns:
        results dict
    """
    print("=" * 65)
    print("  PHASE 2: DQL DEPLOYMENT LIFECYCLE AGENT")
    print("  Hypothesis: trust score predicts optimal retraining policy")
    print("=" * 65)

    # Build environments
    print("\n  Building deployment environments...")
    all_envs = build_envs(datasets_with_trust)
    if len(all_envs) < 4:
        print("  Not enough datasets. Need at least 4.")
        return {}

    # Train/test split on environments
    n_train = max(2, int(len(all_envs) * 0.7))
    train_envs = all_envs[:n_train]
    test_envs  = all_envs[n_train:]
    print(f"  {len(all_envs)} environments: {len(train_envs)} train, {len(test_envs)} test")

    # Train two agents: one with trust in state, one without
    print("\n  Training DQL agent WITH trust in state...")
    agent_with_trust = DQLAgent(use_trust=True, epsilon=1.0)
    train_agent(agent_with_trust, train_envs, n_episodes=300, verbose=True)

    print("\n  Training DQL agent WITHOUT trust in state...")
    agent_no_trust = DQLAgent(use_trust=False, epsilon=1.0)
    train_agent(agent_no_trust, train_envs, n_episodes=300, verbose=True)

    # Evaluate
    print("\n  Evaluating all policies on test environments...")
    results_df = evaluate_all_policies(test_envs, agent_with_trust, agent_no_trust)
    results_df.to_csv(OUT / "policy_comparison.csv", index=False)

    # Analysis 1: Policy comparison
    print("\n  Policy comparison (mean F1 across test episodes):")
    print(f"  {'Policy':20s}  {'Mean F1':8s}  {'Retrains':8s}  {'Reward':8s}")
    print(f"  {'-'*55}")
    policy_summary = results_df.groupby('policy').agg(
        mean_f1=('mean_f1', 'mean'),
        mean_retrains=('retrain_count', 'mean'),
        mean_reward=('total_reward', 'mean'),
    ).round(4)

    for policy, row in policy_summary.sort_values('mean_reward', ascending=False).iterrows():
        print(f"  {policy:20s}  {row['mean_f1']:.4f}   {row['mean_retrains']:.1f}     {row['mean_reward']:.4f}")

    # Analysis 2: Does trust predict Q(RETRAIN)?
    print("\n  Q-value analysis: does trust predict Q(RETRAIN)?")
    q_analysis = agent_with_trust.get_retrain_q_by_trust()
    for tier, vals in q_analysis.items():
        print(f"  Trust tier {tier}: mean Q(RETRAIN)={vals['mean_q_retrain']:.4f}  "
              f"(n={vals['n_states']} states)")

    if q_analysis:
        tiers = list(q_analysis.keys())
        q_vals = [q_analysis[t]['mean_q_retrain'] for t in tiers]
        if len(q_vals) >= 2:
            # Is Q(RETRAIN) higher for low trust?
            low_trust_higher = q_vals[0] > q_vals[-1] if len(q_vals) >= 2 else None
            print(f"\n  Low-trust states have higher Q(RETRAIN): "
                  f"{'✅ YES — trust predicts policy' if low_trust_higher else '❌ NO'}")

    # Analysis 3: Correlation between trust and optimal retrain frequency
    trust_retrain_corr = stats.spearmanr(
        results_df[results_df['policy'] == 'DQL_with_trust']['trust_at_training'],
        results_df[results_df['policy'] == 'DQL_with_trust']['retrain_count']
    )
    print(f"\n  Trust ↔ retrain frequency (DQL policy):")
    print(f"  Spearman r={trust_retrain_corr.correlation:.4f}  "
          f"p={trust_retrain_corr.pvalue:.4f}  "
          f"{'✅ significant' if trust_retrain_corr.pvalue < 0.05 else '—'}")
    print("  (Negative r = low trust → more retrains = correct direction)")

    # Statistical test: DQL_with_trust vs baselines
    dql_trust_rewards = results_df[results_df['policy'] == 'DQL_with_trust']['total_reward'].values
    reactive_rewards  = results_df[results_df['policy'] == 'Reactive_10pct']['total_reward'].values

    if len(dql_trust_rewards) >= 5 and len(reactive_rewards) >= 5:
        t_stat, t_p = stats.ttest_ind(dql_trust_rewards, reactive_rewards)
        print(f"\n  DQL_with_trust vs Reactive (t-test): "
              f"t={t_stat:.4f}  p={t_p:.4f}  "
              f"{'✅ significant' if t_p < 0.05 else '—'}")

    def _j(o):
        if isinstance(o, (bool, np.bool_)): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating):
            return None if (np.isnan(o) or np.isinf(o)) else float(o)
        return str(o)

    results = {
        'n_train_envs':     len(train_envs),
        'n_test_envs':      len(test_envs),
        'policy_summary':   policy_summary.to_dict(),
        'q_analysis_by_trust': q_analysis,
        'trust_retrain_spearman_r':   round(float(trust_retrain_corr.correlation), 4),
        'trust_retrain_spearman_p':   round(float(trust_retrain_corr.pvalue), 4),
        'trust_predicts_policy':      bool(trust_retrain_corr.pvalue < 0.05),
        'key_finding': (
            f"DQL agent with trust in state achieves "
            f"mean reward "
            f"{policy_summary.loc['DQL_with_trust','mean_reward']:.4f} vs "
            f"{policy_summary.loc['Reactive_10pct','mean_reward']:.4f} (reactive). "
            f"Trust ↔ retrain frequency: r={trust_retrain_corr.correlation:.4f} "
            f"(p={trust_retrain_corr.pvalue:.4f})."
        ) if 'DQL_with_trust' in policy_summary.index and 'Reactive_10pct' in policy_summary.index else "Insufficient data"
    }

    with open(OUT / "phase2_results.json", "w") as f:
        json.dump(results, f, indent=2, default=_j)

    print(f"\n  Results saved → {OUT}/")
    return results


if __name__ == "__main__":
    # Self-test with generated datasets
    from sklearn.datasets import make_classification

    print("Self-testing Phase 2...")
    test_sets = []
    trust_levels = [0.45, 0.55, 0.65, 0.72, 0.78, 0.85, 0.91, 0.95]
    for i, trust in enumerate(trust_levels):
        X, y = make_classification(n_samples=400, n_features=10,
                                   n_informative=6, random_state=i)
        df = pd.DataFrame(X, columns=[f"f{j}" for j in range(10)])
        df["target"] = y
        test_sets.append((df, "target", trust))

    results = run_phase2(test_sets)
    print("\nPhase 2 complete.")
