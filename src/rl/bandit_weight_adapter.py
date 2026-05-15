"""
EMMDS System 1: Contextual Bandit Trust Weight Adapter
=======================================================
Implements a Neural Epsilon-Greedy Contextual Bandit that learns
to select optimal trust component weights from dataset meta-features.

WHY CONTEXTUAL BANDIT AND NOT PPO/SAC:
  The weight selection problem has no temporal structure ACROSS datasets.
  Dataset A's result does not affect Dataset B. This makes it a
  contextual bandit problem (episode length = 1), not a full MDP.
  Contextual bandits are:
    - Theoretically correct for this problem structure
    - Interpretable (we can extract the learned policy)
    - Faster to train (no temporal credit assignment)
    - More stable (no variance from multi-step returns)

FORMULATION:
  Context (state):   15 dataset meta-features
  Arms (actions):    Discretised weight configurations (100 arms)
                     Each arm = one weight vector [w1..w5] summing to 1
  Reward:            -deployment_risk of selected model under those weights
  Policy:            Neural network: context → Q-values for each arm
  Exploration:       Epsilon-greedy with linear decay

RESEARCH QUESTION:
  Does a bandit policy for weight selection generalise better to
  out-of-distribution datasets than a linear meta-learner?
  Evaluation: LOO on dataset collection.
  Baselines: fixed weights, equal weights, linear meta-learner.
"""

import numpy as np
import json
from pathlib import Path
from typing import Optional

OUT = Path("outputs/research/rl")
OUT.mkdir(parents=True, exist_ok=True)

# Trust components
COMPONENTS = ['w_acc', 'w_cal', 'w_agr', 'w_dq', 'w_stab']
COL_MAP    = {
    'w_acc':  'test_f1',
    'w_cal':  'cal_score',
    'w_agr':  'agreement_score',
    'w_dq':   'dq_score',
    'w_stab': 'stability',
}


# ══════════════════════════════════════════════════════════════════════
# ARM SPACE: Discretised weight configurations
# ══════════════════════════════════════════════════════════════════════

def build_arm_space(n_arms: int = 120) -> np.ndarray:
    """
    Build a set of N diverse weight configurations as arms.
    Each row is a valid weight vector [w1..w5] summing to 1.0.

    Covers:
      - Extreme arms: one component = 1.0 (5 arms)
      - Equal weight: all 0.2 (1 arm)
      - Empirically derived: stability-dominant (1 arm)
      - Grid: systematic sampling of weight space
    """
    arms = []

    # Extreme arms (one component dominates)
    for i in range(5):
        w = np.zeros(5)
        w[i] = 1.0
        arms.append(w)

    # Equal weights
    arms.append(np.ones(5) / 5)

    # Empirically derived (from Direction 1)
    arms.append(np.array([0.05, 0.10, 0.10, 0.35, 0.40]))

    # Stability-dominant variants
    for s in [0.4, 0.5, 0.6]:
        remaining = 1.0 - s
        w = np.array([remaining/4]*4 + [s])
        arms.append(w)

    # Data quality dominant
    for d in [0.4, 0.5, 0.6]:
        remaining = 1.0 - d
        w = np.array([remaining/4]*3 + [d] + [remaining/4])
        arms.append(w)

    # Grid sample: generate diverse weight vectors
    rng = np.random.RandomState(42)
    while len(arms) < n_arms:
        raw   = rng.dirichlet(np.ones(5))     # Uniform over simplex
        if raw.max() > 0.7:                   # Skip too-extreme configs
            continue
        arms.append(raw)

    arms = np.array(arms[:n_arms])
    return arms


# ══════════════════════════════════════════════════════════════════════
# NEURAL Q-NETWORK (2-layer MLP from scratch)
# ══════════════════════════════════════════════════════════════════════

class NeuralQNetwork:
    """
    2-layer MLP that maps context → Q-values for each arm.
    Implemented from scratch using numpy for full interpretability
    and zero external dependencies.

    Architecture: input(15) → hidden(64) → hidden(32) → output(n_arms)
    Activation: ReLU
    Loss: MSE on target Q-values
    Optimiser: Adam (implemented manually)
    """

    def __init__(self, n_context: int, n_arms: int,
                 hidden1: int = 64, hidden2: int = 32,
                 lr: float = 0.001):
        self.n_context = n_context
        self.n_arms    = n_arms
        self.lr        = lr

        # Xavier initialisation
        rng = np.random.RandomState(42)
        self.W1 = rng.randn(n_context, hidden1) * np.sqrt(2/n_context)
        self.b1 = np.zeros(hidden1)
        self.W2 = rng.randn(hidden1, hidden2)   * np.sqrt(2/hidden1)
        self.b2 = np.zeros(hidden2)
        self.W3 = rng.randn(hidden2, n_arms)    * np.sqrt(2/hidden2)
        self.b3 = np.zeros(n_arms)

        # Adam optimiser state
        self._t = 0
        self._ms = {k: np.zeros_like(v)
                    for k, v in self._params().items()}
        self._vs = {k: np.zeros_like(v)
                    for k, v in self._params().items()}

    def _params(self) -> dict:
        return {'W1':self.W1,'b1':self.b1,'W2':self.W2,
                'b2':self.b2,'W3':self.W3,'b3':self.b3}

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _relu_grad(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    def forward(self, x: np.ndarray) -> tuple:
        """Forward pass. Returns (output, cache) for backprop."""
        z1 = x @ self.W1 + self.b1;   a1 = self._relu(z1)
        z2 = a1 @ self.W2 + self.b2;  a2 = self._relu(z2)
        z3 = a2 @ self.W3 + self.b3
        return z3, (x, z1, a1, z2, a2, z3)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict Q-values for context x."""
        q, _ = self.forward(x)
        return q

    def train_step(self, x: np.ndarray, arm_idx: int,
                   target: float) -> float:
        """
        Single gradient update for one (context, arm, target) triple.
        Updates only the Q-value for the selected arm (others unchanged).
        """
        q, (inp, z1, a1, z2, a2, z3) = self.forward(x)

        # Compute loss only on selected arm
        td_error = q[arm_idx] - target
        loss     = 0.5 * td_error ** 2

        # Backprop
        dL_dq     = np.zeros(self.n_arms)
        dL_dq[arm_idx] = td_error

        dL_dW3 = np.outer(a2, dL_dq)
        dL_db3 = dL_dq

        dL_da2 = dL_dq @ self.W3.T
        dL_dz2 = dL_da2 * self._relu_grad(z2)
        dL_dW2 = np.outer(a1, dL_dz2)
        dL_db2 = dL_dz2

        dL_da1 = dL_dz2 @ self.W2.T
        dL_dz1 = dL_da1 * self._relu_grad(z1)
        dL_dW1 = np.outer(inp, dL_dz1)
        dL_db1 = dL_dz1

        grads = {'W1':dL_dW1,'b1':dL_db1,'W2':dL_dW2,
                 'b2':dL_db2,'W3':dL_dW3,'b3':dL_db3}

        # Adam update
        self._t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for k, g in grads.items():
            self._ms[k] = beta1*self._ms[k] + (1-beta1)*g
            self._vs[k] = beta2*self._vs[k] + (1-beta2)*g**2
            m_hat = self._ms[k] / (1 - beta1**self._t)
            v_hat = self._vs[k] / (1 - beta2**self._t)
            p = getattr(self, k)
            setattr(self, k, p - self.lr * m_hat / (np.sqrt(v_hat) + eps))

        return float(loss)


# ══════════════════════════════════════════════════════════════════════
# CONTEXTUAL BANDIT AGENT
# ══════════════════════════════════════════════════════════════════════

class ContextualBanditAgent:
    """
    Neural epsilon-greedy contextual bandit for trust weight selection.

    Policy: with probability ε, choose random arm (explore)
            with probability 1-ε, choose arm with highest Q-value (exploit)
    Epsilon decays linearly from eps_start to eps_end over training.
    """

    def __init__(
        self,
        n_context:   int   = 15,
        n_arms:      int   = 120,
        eps_start:   float = 1.0,
        eps_end:     float = 0.05,
        eps_decay:   int   = 500,
        lr:          float = 0.001,
        replay_size: int   = 1000,
    ):
        self.arms       = build_arm_space(n_arms)
        self.n_arms     = len(self.arms)
        self.q_net      = NeuralQNetwork(n_context, self.n_arms, lr=lr)
        self.eps_start  = eps_start
        self.eps_end    = eps_end
        self.eps_decay  = eps_decay
        self.step_count = 0
        self.rng        = np.random.RandomState(42)

        # Experience replay buffer: (context, arm_idx, reward)
        self.replay_X   = []
        self.replay_arm = []
        self.replay_r   = []
        self.replay_size = replay_size

        # Training history
        self.reward_history = []
        self.loss_history   = []
        self.eps_history    = []

    @property
    def epsilon(self) -> float:
        """Current exploration rate."""
        progress = min(1.0, self.step_count / self.eps_decay)
        return self.eps_end + (self.eps_start - self.eps_end) * (1 - progress)

    def select_arm(self, context: np.ndarray,
                   greedy: bool = False) -> tuple:
        """
        Select an arm (weight configuration) for the given context.

        Args:
            context: Normalised meta-feature vector
            greedy:  If True, always exploit (for evaluation)

        Returns:
            (arm_index, weight_vector)
        """
        if not greedy and self.rng.random() < self.epsilon:
            # Explore: random arm
            idx = self.rng.randint(0, self.n_arms)
        else:
            # Exploit: best Q-value
            q_values = self.q_net.predict(context)
            idx      = int(np.argmax(q_values))

        return idx, self.arms[idx].copy()

    def update(self, context: np.ndarray, arm_idx: int,
               reward: float, batch_size: int = 32) -> float:
        """
        Store experience and perform one gradient update.

        Args:
            context:  Meta-feature context
            arm_idx:  Arm that was selected
            reward:   Observed reward (negative deployment risk)
            batch_size: Mini-batch size for replay update

        Returns:
            Training loss
        """
        # Store in replay buffer
        self.replay_X.append(context.copy())
        self.replay_arm.append(arm_idx)
        self.replay_r.append(reward)

        # Trim buffer
        if len(self.replay_X) > self.replay_size:
            self.replay_X   = self.replay_X[-self.replay_size:]
            self.replay_arm = self.replay_arm[-self.replay_size:]
            self.replay_r   = self.replay_r[-self.replay_size:]

        self.step_count += 1
        self.reward_history.append(reward)
        self.eps_history.append(self.epsilon)

        # Mini-batch update from replay
        n = len(self.replay_X)
        if n < 4:
            return 0.0

        batch_idx = self.rng.choice(n, min(batch_size, n), replace=False)
        total_loss = 0.0
        for i in batch_idx:
            loss = self.q_net.train_step(
                self.replay_X[i],
                self.replay_arm[i],
                self.replay_r[i],
            )
            total_loss += loss

        avg_loss = total_loss / len(batch_idx)
        self.loss_history.append(avg_loss)
        return avg_loss

    def get_policy_weights(self, context: np.ndarray) -> np.ndarray:
        """Return the greedy weight selection for a given context."""
        _, weights = self.select_arm(context, greedy=True)
        return weights

    def save(self, path: str) -> None:
        """Save agent state."""
        state = {
            'W1': self.q_net.W1.tolist(), 'b1': self.q_net.b1.tolist(),
            'W2': self.q_net.W2.tolist(), 'b2': self.q_net.b2.tolist(),
            'W3': self.q_net.W3.tolist(), 'b3': self.q_net.b3.tolist(),
            'arms':        self.arms.tolist(),
            'step_count':  self.step_count,
            'reward_history': self.reward_history[-100:],
        }
        Path(path).write_text(json.dumps(state))

    def load(self, path: str) -> None:
        """Load agent state."""
        state = json.loads(Path(path).read_text())
        self.q_net.W1 = np.array(state['W1'])
        self.q_net.b1 = np.array(state['b1'])
        self.q_net.W2 = np.array(state['W2'])
        self.q_net.b2 = np.array(state['b2'])
        self.q_net.W3 = np.array(state['W3'])
        self.q_net.b3 = np.array(state['b3'])
        self.arms      = np.array(state['arms'])
        self.step_count = state['step_count']


# ══════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

class BanditTrainer:
    """
    Trains the contextual bandit across a dataset collection.
    Each dataset is one episode: observe context → select weights →
    measure deployment risk → update policy.
    """

    def __init__(self, n_arms: int = 120, n_episodes_per_dataset: int = 3):
        self.agent   = ContextualBanditAgent(n_context=15, n_arms=n_arms)
        self.n_eps   = n_episodes_per_dataset
        self.context_scaler_mean = None
        self.context_scaler_std  = None
        self.train_log = []

    def _compute_reward(
        self,
        weights:   np.ndarray,
        df_models: 'pd.DataFrame',
    ) -> tuple:
        """
        Apply weight vector to model measurements, select best model,
        return negative deployment risk as reward.
        """
        import pandas as pd, numpy as np
        trust = sum(
            weights[i] * np.clip(df_models[COL_MAP[c]], 0, 1)
            for i, c in enumerate(COMPONENTS)
        )
        selected     = df_models.loc[trust.idxmax()]
        reward       = -float(selected['deployment_risk'])
        return reward, selected['model']

    def _normalise_context(self, ctx: np.ndarray) -> np.ndarray:
        if self.context_scaler_mean is None:
            return ctx
        return (ctx - self.context_scaler_mean) / (self.context_scaler_std + 1e-8)

    def fit_scaler(self, all_contexts: np.ndarray) -> None:
        self.context_scaler_mean = all_contexts.mean(axis=0)
        self.context_scaler_std  = all_contexts.std(axis=0)

    def train(self, dataset_records: list, verbose: bool = True) -> dict:
        """
        Train the bandit on a list of dataset records.

        Each record: {'context': np.array(15), 'df_models': DataFrame}

        Returns training summary.
        """
        import numpy as np

        # Fit context normaliser
        contexts = np.stack([r['context'] for r in dataset_records])
        self.fit_scaler(contexts)

        print(f"  Training bandit: {len(dataset_records)} datasets "
              f"× {self.n_eps} episodes = "
              f"{len(dataset_records)*self.n_eps} total steps")

        total_rewards = []
        for episode in range(len(dataset_records) * self.n_eps):
            # Sample a random dataset each episode
            rec = dataset_records[episode % len(dataset_records)]
            ctx = self._normalise_context(rec['context'])

            # Select arm
            arm_idx, weights = self.agent.select_arm(ctx)

            # Observe reward
            reward, chosen_model = self._compute_reward(weights, rec['df_models'])

            # Update
            loss = self.agent.update(ctx, arm_idx, reward)

            total_rewards.append(reward)
            self.train_log.append({
                'episode':     episode,
                'reward':      round(reward, 6),
                'loss':        round(float(loss), 6),
                'epsilon':     round(self.agent.epsilon, 4),
                'dataset':     rec['name'],
                'chosen_model': chosen_model,
            })

            if verbose and (episode + 1) % 50 == 0:
                recent = np.mean(total_rewards[-50:])
                print(f"    Episode {episode+1:4d}  "
                      f"reward(50)={recent:.4f}  "
                      f"ε={self.agent.epsilon:.3f}  "
                      f"loss={loss:.4f}")

        self.agent.save(str(OUT / 'bandit_agent.json'))

        return {
            'total_episodes':  len(dataset_records) * self.n_eps,
            'final_epsilon':   round(self.agent.epsilon, 4),
            'mean_reward':     round(float(np.mean(total_rewards)), 6),
            'final_reward_50': round(float(np.mean(total_rewards[-50:])), 6),
        }

    def evaluate_loo(self, dataset_records: list) -> dict:
        """
        Leave-one-out evaluation: for each dataset, train on all others,
        evaluate on the held-out dataset.
        Compare bandit vs fixed vs equal vs linear meta-learner.
        """
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.multioutput import MultiOutputRegressor

        FIXED   = np.array([0.05, 0.10, 0.10, 0.35, 0.40])  # empirical v3
        EQUAL   = np.ones(5) / 5
        ACC_ONLY = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

        results = []
        n = len(dataset_records)
        print(f"\n  LOO evaluation: {n} datasets")

        for held_idx in range(n):
            held_rec  = dataset_records[held_idx]
            train_rec = [r for i, r in enumerate(dataset_records) if i != held_idx]

            # Train fresh bandit on training split
            fresh_agent = ContextualBanditAgent(n_context=15, n_arms=len(self.agent.arms))
            fresh_agent.arms = self.agent.arms.copy()

            train_ctxs = np.stack([r['context'] for r in train_rec])
            scaler_mean = train_ctxs.mean(axis=0)
            scaler_std  = train_ctxs.std(axis=0)

            for ep in range(len(train_rec) * 3):
                rec = train_rec[ep % len(train_rec)]
                ctx = (rec['context'] - scaler_mean) / (scaler_std + 1e-8)
                arm_idx, weights = fresh_agent.select_arm(ctx)
                reward, _ = self._compute_reward(weights, rec['df_models'])
                fresh_agent.update(ctx, arm_idx, reward, batch_size=16)

            # Evaluate on held-out
            held_ctx   = (held_rec['context'] - scaler_mean) / (scaler_std + 1e-8)
            _, bandit_w = fresh_agent.select_arm(held_ctx, greedy=True)

            # Linear meta-learner on same training split
            X_meta = np.stack([r['context'] for r in train_rec])
            Y_meta = np.stack([r['optimal_weights'] for r in train_rec
                               if 'optimal_weights' in r])
            meta_w = FIXED.copy()
            if len(Y_meta) >= 3:
                try:
                    mlr = MultiOutputRegressor(RandomForestRegressor(n_estimators=50, random_state=42))
                    mlr.fit(X_meta[:len(Y_meta)], Y_meta)
                    raw = mlr.predict(held_rec['context'].reshape(1,-1))[0]
                    raw = np.clip(raw, 0, 1)
                    meta_w = raw / (raw.sum() + 1e-8)
                except Exception:
                    pass

            # Compute risk for each strategy
            row = {'dataset': held_rec['name']}
            for sname, w in [('bandit', bandit_w), ('fixed', FIXED),
                              ('equal', EQUAL), ('acc_only', ACC_ONLY),
                              ('linear_meta', meta_w)]:
                reward, _ = self._compute_reward(w, held_rec['df_models'])
                row[f'{sname}_risk'] = round(-reward, 6)
                row[f'{sname}_wins_vs_fixed'] = bool(-reward <= -self._compute_reward(FIXED, held_rec['df_models'])[0])
            results.append(row)
            print(f"    [{held_idx+1:2d}/{n}] {held_rec['name']:30s}  "
                  f"bandit={row['bandit_risk']:.4f}  "
                  f"fixed={row['fixed_risk']:.4f}  "
                  f"meta={row['linear_meta_risk']:.4f}  "
                  f"{'✅' if row['bandit_risk']<=row['fixed_risk'] else '  '}")

        import pandas as pd
        loo_df = pd.DataFrame(results)
        loo_df.to_csv(OUT / 'bandit_loo_results.csv', index=False)

        # Summary
        bandit_wins = int(loo_df['bandit_wins_vs_fixed'].sum())
        summary = {
            'n_datasets':         n,
            'bandit_wins_vs_fixed': bandit_wins,
            'bandit_win_rate':    round(bandit_wins/n, 4),
            'mean_risk': {
                sname: round(float(loo_df[f'{sname}_risk'].mean()), 6)
                for sname in ['bandit','fixed','equal','acc_only','linear_meta']
            },
        }
        print(f"\n  Summary: bandit wins {bandit_wins}/{n} vs fixed weights")
        for sname, risk in summary['mean_risk'].items():
            print(f"    {sname:15s}  mean_risk={risk:.6f}")

        return summary, loo_df
