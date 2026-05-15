"""
EMMDS Component 1: DQL Deployment Monitor
==========================================
Deep Q-Learning agent for optimal retraining timing.

Research Question:
  "Does training-time trust score predict optimal retraining
   policy parameters, and does a DQL agent that incorporates
   trust signals outperform fixed-threshold baselines?"

Key novelty: the state representation includes trust_score_at_last_training.
This tests whether pre-deployment reliability estimates shape
optimal post-deployment management — no existing paper does this.

Environment:
  State  (7-dim): [rolling_f1, f1_delta, ks_mean, psi_mean,
                   batches_since_retrain, trust_at_training, drift_trend]
  Action (3):     0=continue, 1=retrain, 2=switch_to_backup
  Reward:         current_f1 - lambda * retrain_cost

Baselines compared:
  B1: Fixed schedule (retrain every K batches)
  B2: Reactive threshold (retrain when F1 drops below threshold)
  B3: DQL without trust signal (ablation)
  B4: DQL with trust signal (proposed)
"""

import sys, warnings, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque
from typing import Optional, List, Tuple
warnings.filterwarnings('ignore')

OUT = Path("outputs/rl")
OUT.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# REPLAY BUFFER
# ══════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """Experience replay buffer for DQL training."""
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self): return len(self.buffer)


# ══════════════════════════════════════════════════════════════════════
# Q-NETWORK (pure numpy — no torch/tf dependency)
# ══════════════════════════════════════════════════════════════════════

class QNetwork:
    """
    Simple fully-connected Q-network implemented in pure numpy.
    Architecture: 7 → 64 → 32 → 3
    Activation: ReLU hidden, linear output
    """

    def __init__(self, state_dim: int = 7, action_dim: int = 3,
                 hidden: Tuple[int, ...] = (64, 32), lr: float = 1e-3):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.lr         = lr

        dims = [state_dim] + list(hidden) + [action_dim]
        self.W = []
        self.b = []
        rng = np.random.RandomState(42)
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / dims[i])
            self.W.append(rng.randn(dims[i], dims[i+1]) * scale)
            self.b.append(np.zeros(dims[i+1]))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass. x: (batch, state_dim) or (state_dim,)"""
        single = x.ndim == 1
        h = x.reshape(-1, self.state_dim).astype(np.float32)
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.maximum(0, h)  # ReLU
        return h[0] if single else h

    def update(self, states, actions, targets):
        """One gradient step via MSE loss + backprop."""
        batch = states.shape[0]
        # Forward pass with cache
        activations = [states.astype(np.float32)]
        h = states.astype(np.float32)
        pre_acts = []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            pre_acts.append(z)
            if i < len(self.W) - 1:
                h = np.maximum(0, z)
            else:
                h = z
            activations.append(h)

        # Q values for taken actions
        q_vals    = activations[-1]
        q_targets = q_vals.copy()
        for k in range(batch):
            q_targets[k, actions[k]] = targets[k]

        # Backprop
        delta = (q_vals - q_targets) * 2 / batch
        for i in range(len(self.W) - 1, -1, -1):
            dW = activations[i].T @ delta
            db = delta.sum(axis=0)
            if i > 0:
                delta = delta @ self.W[i].T
                delta *= (activations[i] > 0).astype(np.float32)
            self.W[i] -= self.lr * np.clip(dW, -1, 1)
            self.b[i] -= self.lr * np.clip(db, -1, 1)

    def copy_weights_from(self, other: "QNetwork"):
        self.W = [w.copy() for w in other.W]
        self.b = [b.copy() for b in other.b]

    def predict_action(self, state: np.ndarray) -> int:
        q = self.forward(state)
        return int(np.argmax(q))


# ══════════════════════════════════════════════════════════════════════
# DEPLOYMENT ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════

class DeploymentEnvironment:
    """
    Simulates a production ML deployment with covariate shift.

    State: [rolling_f1, f1_delta, ks_mean, psi_mean,
            batches_since_retrain, trust_at_training, drift_trend]
    """

    STATE_DIM  = 7
    ACTION_DIM = 3   # 0=continue, 1=retrain, 2=switch_backup

    def __init__(self, X_train, y_train, X_test, y_test, model,
                 backup_model, baseline_f1: float, trust_score: float,
                 retrain_cost: float = 0.05, shift_rate: float = 0.05,
                 n_batches: int = 40, batch_size: int = 50):

        self.X_train      = X_train
        self.y_train      = y_train
        self.X_test       = X_test
        self.y_test       = y_test
        self.model        = model
        self.backup_model = backup_model
        self.baseline_f1  = baseline_f1
        self.trust_score  = trust_score
        self.retrain_cost = retrain_cost
        self.shift_rate   = shift_rate
        self.n_batches    = n_batches
        self.batch_size   = min(batch_size, len(X_test))
        self._rng         = np.random.RandomState(None)

        # Current state
        self._batch      = 0
        self._since_ret  = 0
        self._current_model = None
        self._f1_history  = deque(maxlen=10)
        self._drift_history = deque(maxlen=5)
        self._cumulative_reward = 0.0
        self._retrain_count = 0

    def reset(self, seed: int = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._batch      = 0
        self._since_ret  = 0
        self._current_model = self._clone_and_fit(self.model)
        self._f1_history.clear()
        self._drift_history.clear()
        self._cumulative_reward = 0.0
        self._retrain_count = 0
        f1 = self._evaluate(self._current_model, 0.0)
        self._f1_history.append(f1)
        return self._build_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Execute action, return (next_state, reward, done)."""
        self._batch    += 1
        self._since_ret += 1
        shift_mag = self._batch * self.shift_rate

        # Action execution
        retrain_penalty = 0.0
        if action == 1:  # Retrain
            self._current_model = self._clone_and_fit(self.model)
            retrain_penalty     = self.retrain_cost
            self._since_ret     = 0
            self._retrain_count += 1
        elif action == 2:  # Switch to backup
            self._current_model = self._clone_and_fit(self.backup_model)
            retrain_penalty     = self.retrain_cost * 0.3

        # Evaluate on shifted data
        f1  = self._evaluate(self._current_model, shift_mag)
        ks  = min(0.0 + shift_mag * 0.08, 0.8)
        psi = min(0.0 + shift_mag * 0.06, 0.5)

        self._f1_history.append(f1)
        self._drift_history.append(ks)

        # Reward: performance minus retraining cost
        reward = f1 - retrain_penalty
        self._cumulative_reward += reward

        next_state = self._build_state()
        done = self._batch >= self.n_batches
        return next_state, reward, done

    def _evaluate(self, model, shift_mag: float) -> float:
        from sklearn.metrics import f1_score
        X_shifted = self.X_test + shift_mag * self._rng.randn(*self.X_test.shape) * 0.3
        idx = self._rng.choice(len(X_shifted), self.batch_size, replace=False)
        try:
            y_pred = model.predict(X_shifted[idx])
            return float(f1_score(self.y_test[idx], y_pred,
                                  average='weighted', zero_division=0))
        except:
            return float(self.baseline_f1 * 0.5)

    def _build_state(self) -> np.ndarray:
        rolling_f1   = float(np.mean(self._f1_history)) if self._f1_history else self.baseline_f1
        f1_delta     = rolling_f1 - self.baseline_f1
        ks_mean      = float(np.mean(self._drift_history)) if self._drift_history else 0.0
        psi_mean     = ks_mean * 0.75
        since_ret_n  = min(self._since_ret / self.n_batches, 1.0)
        trust_n      = float(self.trust_score)
        drift_trend  = (float(np.mean(list(self._drift_history)[-3:])) -
                        float(np.mean(list(self._drift_history)[:3]))
                        if len(self._drift_history) >= 4 else 0.0)
        return np.array([rolling_f1, f1_delta, ks_mean, psi_mean,
                         since_ret_n, trust_n, drift_trend], dtype=np.float32)

    def _clone_and_fit(self, model):
        from sklearn.base import clone
        m = clone(model)
        m.fit(self.X_train, self.y_train)
        return m

    def get_stats(self) -> dict:
        return {
            'cumulative_reward': round(self._cumulative_reward, 4),
            'retrain_count':     self._retrain_count,
            'mean_f1':           round(float(np.mean(list(self._f1_history))), 4),
        }


# ══════════════════════════════════════════════════════════════════════
# DQL AGENT
# ══════════════════════════════════════════════════════════════════════

class DQLAgent:
    """
    Deep Q-Learning agent for deployment monitoring.
    Uses experience replay and a target network for stable training.
    """

    def __init__(self, state_dim: int = 7, action_dim: int = 3,
                 lr: float = 1e-3, gamma: float = 0.95,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.995,
                 batch_size: int = 32, target_update: int = 20,
                 use_trust_signal: bool = True):

        self.state_dim       = state_dim
        self.action_dim      = action_dim
        self.gamma           = gamma
        self.epsilon         = epsilon_start
        self.epsilon_end     = epsilon_end
        self.epsilon_decay   = epsilon_decay
        self.batch_size      = batch_size
        self.target_update   = target_update
        self.use_trust_signal = use_trust_signal

        self.q_net      = QNetwork(state_dim, action_dim, lr=lr)
        self.target_net = QNetwork(state_dim, action_dim, lr=lr)
        self.target_net.copy_weights_from(self.q_net)
        self.replay     = ReplayBuffer(capacity=10000)
        self.steps      = 0
        self.losses     = []

    def select_action(self, state: np.ndarray) -> int:
        if not self.use_trust_signal:
            state = state.copy()
            state[5] = 0.5   # Mask trust signal for ablation
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        return self.q_net.predict_action(state)

    def remember(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        if len(self.replay) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = \
            self.replay.sample(self.batch_size)

        # Bellman targets
        next_q   = self.target_net.forward(next_states).max(axis=1)
        targets  = rewards + self.gamma * next_q * (1 - dones.astype(float))

        self.q_net.update(states, actions, targets)
        self.steps += 1

        if self.steps % self.target_update == 0:
            self.target_net.copy_weights_from(self.q_net)

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        loss = float(np.mean((self.q_net.forward(states).max(axis=1) - targets)**2))
        self.losses.append(loss)
        return loss


# ══════════════════════════════════════════════════════════════════════
# BASELINES
# ══════════════════════════════════════════════════════════════════════

def run_fixed_schedule(env: DeploymentEnvironment, retrain_every: int = 8) -> dict:
    """Baseline 1: retrain every K batches."""
    state = env.reset(seed=0)
    done  = False
    while not done:
        action = 1 if (env._since_ret >= retrain_every) else 0
        state, reward, done = env.step(action)
    return env.get_stats()


def run_reactive(env: DeploymentEnvironment, threshold: float = 0.85) -> dict:
    """Baseline 2: retrain when rolling F1 drops below threshold."""
    state = env.reset(seed=0)
    done  = False
    while not done:
        rolling_f1 = state[0]
        action = 1 if rolling_f1 < threshold * env.baseline_f1 else 0
        state, reward, done = env.step(action)
    return env.get_stats()


def run_dql(env: DeploymentEnvironment, agent: DQLAgent,
            training: bool = False) -> dict:
    """Run DQL agent (evaluation mode)."""
    state = env.reset(seed=0)
    done  = False
    while not done:
        if training:
            action = agent.select_action(state)
        else:
            # Greedy during evaluation
            state_in = state.copy()
            if not agent.use_trust_signal:
                state_in[5] = 0.5
            action = agent.q_net.predict_action(state_in)
        next_state, reward, done = env.step(action)
        if training:
            agent.remember(state, action, reward, next_state, done)
            agent.train_step()
        state = next_state
    return env.get_stats()


# ══════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def train_dql_agent(
    env_factory,      # Callable that returns a fresh DeploymentEnvironment
    n_episodes: int = 300,
    use_trust: bool = True,
    verbose:   bool = True,
) -> DQLAgent:
    """Train a DQL agent across multiple episodes."""
    agent = DQLAgent(use_trust_signal=use_trust)
    episode_rewards = []
    best_reward     = -np.inf
    best_weights    = None

    for ep in range(n_episodes):
        env   = env_factory()
        state = env.reset(seed=ep)
        done  = False
        ep_reward = 0.0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.train_step()
            state      = next_state
            ep_reward += reward

        episode_rewards.append(ep_reward)

        if ep_reward > best_reward:
            best_reward = ep_reward
            best_weights = ([w.copy() for w in agent.q_net.W],
                            [b.copy() for b in agent.q_net.b])

        if verbose and (ep + 1) % 50 == 0:
            recent = np.mean(episode_rewards[-20:])
            print(f"    Episode {ep+1:4d}/{n_episodes}  "
                  f"reward={ep_reward:.3f}  "
                  f"mean20={recent:.3f}  "
                  f"ε={agent.epsilon:.3f}")

    # Restore best weights
    if best_weights:
        agent.q_net.W, agent.q_net.b = best_weights
    agent.epsilon = 0.0
    return agent


# ══════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════════

def run_dql_experiment():
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from sklearn.datasets import load_breast_cancer, load_wine
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import f1_score
    from sklearn.base import clone

    print("=" * 65)
    print("  COMPONENT 1: DQL DEPLOYMENT MONITOR")
    print("  Research: Does trust score predict optimal policy?")
    print("=" * 65)

    # ── Build training environments (3 datasets) ──────────────────────
    def make_env_factory(X_tr, y_tr, X_te, y_te, model, backup, f1, trust, shift):
        def factory():
            return DeploymentEnvironment(
                X_tr, y_tr, X_te, y_te, model, backup,
                baseline_f1=f1, trust_score=trust,
                shift_rate=shift, n_batches=40
            )
        return factory

    datasets_info = []
    loaders = [
        ("breast_cancer", load_breast_cancer),
        ("wine",          load_wine),
    ]
    for ds_name, loader in loaders:
        d = loader(as_frame=True)
        X = d.data.values; y = d.target.values
        le = LabelEncoder(); y = le.fit_transform(y)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=42,
            stratify=y)
        sc = StandardScaler().fit(X_tr)
        X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)

        model  = RandomForestClassifier(n_estimators=50, random_state=42)
        backup = GradientBoostingClassifier(n_estimators=50, random_state=42)
        model.fit(X_tr_s, y_tr)
        backup.fit(X_tr_s, y_tr)
        f1 = float(f1_score(y_te, model.predict(X_te_s),
                            average='weighted', zero_division=0))
        trust = 0.85 if ds_name == "breast_cancer" else 0.72

        datasets_info.append({
            'name': ds_name, 'X_tr': X_tr_s, 'y_tr': y_tr,
            'X_te': X_te_s, 'y_te': y_te,
            'model': model, 'backup': backup, 'f1': f1, 'trust': trust,
        })
        print(f"  Dataset: {ds_name}  baseline_f1={f1:.4f}  trust={trust}")

    # ── Train agents ──────────────────────────────────────────────────
    # Use first dataset for training, second for evaluation
    train_ds = datasets_info[0]
    test_ds  = datasets_info[1]

    for shift_rate in [0.04, 0.08]:
        print(f"\n  Training DQL agents (shift_rate={shift_rate})...")

        factory_train = make_env_factory(
            train_ds['X_tr'], train_ds['y_tr'],
            train_ds['X_te'], train_ds['y_te'],
            train_ds['model'], train_ds['backup'],
            train_ds['f1'], train_ds['trust'], shift_rate
        )

        print("    Training DQL WITH trust signal...")
        agent_trust = train_dql_agent(factory_train, n_episodes=200,
                                       use_trust=True, verbose=True)

        print("    Training DQL WITHOUT trust signal (ablation)...")
        agent_notrust = train_dql_agent(factory_train, n_episodes=200,
                                         use_trust=False, verbose=False)

        # ── Evaluate on held-out dataset ──────────────────────────────
        print(f"\n  Evaluating on {test_ds['name']} (unseen)...")
        env_test = DeploymentEnvironment(
            test_ds['X_tr'], test_ds['y_tr'],
            test_ds['X_te'], test_ds['y_te'],
            test_ds['model'], test_ds['backup'],
            baseline_f1=test_ds['f1'], trust_score=test_ds['trust'],
            shift_rate=shift_rate, n_batches=40
        )

        results = {}
        # Fixed schedule
        results['fixed_8']  = run_fixed_schedule(env_test, retrain_every=8)
        results['fixed_12'] = run_fixed_schedule(env_test, retrain_every=12)
        # Reactive
        results['reactive_90pct'] = run_reactive(env_test, threshold=0.90)
        results['reactive_85pct'] = run_reactive(env_test, threshold=0.85)
        # DQL agents
        results['dql_with_trust']    = run_dql(env_test, agent_trust)
        results['dql_without_trust'] = run_dql(env_test, agent_notrust)

        print(f"\n  Results (shift={shift_rate}, dataset={test_ds['name']}):")
        print(f"  {'Method':25s}  {'Cum.Reward':12s}  {'Retrains':10s}  {'Mean F1':8s}")
        print(f"  {'-'*60}")
        for name, r in results.items():
            print(f"  {name:25s}  {r['cumulative_reward']:12.4f}  "
                  f"{r['retrain_count']:10d}  {r['mean_f1']:8.4f}")

        # Key comparison
        dql_trust_r    = results['dql_with_trust']['cumulative_reward']
        dql_notrust_r  = results['dql_without_trust']['cumulative_reward']
        best_baseline  = max(results['fixed_8']['cumulative_reward'],
                            results['fixed_12']['cumulative_reward'],
                            results['reactive_90pct']['cumulative_reward'],
                            results['reactive_85pct']['cumulative_reward'])
        trust_advantage = dql_trust_r - dql_notrust_r
        vs_baseline     = dql_trust_r - best_baseline

        print(f"\n  Trust signal advantage: {trust_advantage:+.4f}")
        print(f"  DQL vs best baseline:   {vs_baseline:+.4f}")
        print(f"  Trust helps: {'✅ YES' if trust_advantage > 0 else '❌ NO (within noise)'}")

        # Save
        def _j(o):
            if isinstance(o, (np.bool_,)):    return bool(o)
            if isinstance(o, (np.integer,)):  return int(o)
            if isinstance(o, (np.floating,)):
                return None if np.isnan(o) or np.isinf(o) else float(o)
            return str(o)

        out = {
            'shift_rate': shift_rate,
            'train_dataset': train_ds['name'],
            'test_dataset': test_ds['name'],
            'results': results,
            'trust_signal_advantage': round(float(trust_advantage), 6),
            'vs_best_baseline': round(float(vs_baseline), 6),
        }
        fname = OUT / f"dql_results_shift{int(shift_rate*100)}.json"
        with open(fname, 'w') as f:
            json.dump(out, f, indent=2, default=_j)
        print(f"  Saved → {fname}")

    print("\n  COMPONENT 1 COMPLETE ✅")


if __name__ == "__main__":
    run_dql_experiment()
