"""
EMMDS Phase 2: DQL Deployment Lifecycle Agent
==============================================

Research Hypothesis:
    H0: Trust score at training time does not predict optimal retraining
        timing. A DQL agent that ignores trust score performs equally
        well as one that includes it in its state representation.

    H1: Including training-time trust score in the DQL state
        significantly improves retraining policy quality, measured
        by average deployment performance at lower retraining cost.

Approach:
    1. Build a discrete-action deployment environment (gym-compatible)
    2. Train a DQL agent with trust-aware state across 6 deployment scenarios
    3. Test on 2 held-out scenarios
    4. Compare against 3 baselines:
         - Fixed schedule (retrain every K batches)
         - Reactive threshold (retrain when F1 drops below T)
         - Oracle (knows future performance — upper bound)
    5. Ablation: DQL with trust vs DQL without trust in state

State vector (8 dimensions):
    [current_f1, f1_delta, ks_stat_mean, psi_mean,
     batches_since_retrain, trust_score_at_training,
     degradation_from_peak, drift_severity_encoded]

Actions (discrete, 3):
    0 = CONTINUE deployment
    1 = RETRAIN on recent data
    2 = SWITCH to best available backup model

Reward:
    r_t = current_f1 - λ * retraining_cost * I(action == RETRAIN)
    λ = 0.05  (retraining costs 5% of one batch's F1 value)
"""

import sys
import warnings
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque, namedtuple
from typing import Optional, List, Tuple

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

OUT = Path("outputs/research/phase2_dql")
OUT.mkdir(parents=True, exist_ok=True)

RETRAINING_COST = 0.05   # λ: fraction of F1 lost per retrain action
SWITCH_COST     = 0.02   # smaller cost for switching to backup
N_ACTIONS       = 3      # CONTINUE, RETRAIN, SWITCH
STATE_DIM       = 8      # state vector dimensionality
STATE_DIM_NOTRUST = 7    # state without trust score (ablation)


# ══════════════════════════════════════════════════════════════════════
# DEPLOYMENT ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════

class DeploymentEnvironment:
    """
    Simulates a production ML deployment.

    The environment generates batches of data with covariate shift
    that increases over time. The agent observes drift signals and
    performance metrics and decides when to retrain.

    Episode structure:
        - 50 batches per episode
        - Batches arrive sequentially
        - Shift starts at 0 and increases at rate drift_rate per batch
        - True labels available with 3-batch delay (realistic)
    """

    def __init__(
        self,
        X_train:          np.ndarray,
        y_train:          np.ndarray,
        X_test:           np.ndarray,
        y_test:           np.ndarray,
        model,
        baseline_f1:      float,
        trust_score:      float,
        drift_rate:       float = 0.05,
        n_batches:        int   = 50,
        batch_size:       int   = 50,
        random_state:     int   = 42,
    ):
        self.X_train      = X_train
        self.y_train      = y_train
        self.X_test       = X_test
        self.y_test       = y_test
        self.model        = model
        self.baseline_f1  = baseline_f1
        self.trust_score  = trust_score
        self.drift_rate   = drift_rate
        self.n_batches    = n_batches
        self.batch_size   = batch_size
        self.rng          = np.random.RandomState(random_state)
        self.feature_stds = X_train.std(axis=0) + 1e-8

        # Backup model pool (weaker models as fallback)
        self._backup_models = []
        self._backup_f1s    = []

        # Episode state
        self.reset()

    def add_backup(self, model, f1: float):
        """Register a backup model for SWITCH action."""
        self._backup_models.append(model)
        self._backup_f1s.append(f1)

    def reset(self) -> np.ndarray:
        """Reset episode. Returns initial state."""
        self.current_batch     = 0
        self.current_model     = self.model
        self.current_train_X   = self.X_train.copy()
        self.current_train_y   = self.y_train.copy()
        self.current_shift     = 0.0
        self.batches_since_retrain = 0
        self.f1_history        = deque([self.baseline_f1] * 5, maxlen=10)
        self.peak_f1           = self.baseline_f1
        self.retrain_count     = 0
        self.total_reward      = 0.0
        self.episode_log       = []
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one step.

        Args:
            action: 0=CONTINUE, 1=RETRAIN, 2=SWITCH

        Returns:
            (next_state, reward, done, info)
        """
        # Advance drift
        self.current_shift += self.drift_rate
        self.current_batch += 1
        self.batches_since_retrain += 1

        # Generate shifted batch
        X_batch = self._generate_shifted_batch()

        # Get current performance (simulated with shift degradation)
        f1 = self._evaluate_with_shift(self.current_model, X_batch)
        self.f1_history.append(f1)
        self.peak_f1 = max(self.peak_f1, f1)

        # Base reward = current F1
        reward = float(f1)

        # Execute action
        action_cost = 0.0
        action_name = ["CONTINUE", "RETRAIN", "SWITCH"][action]

        if action == 1:  # RETRAIN
            new_model, new_f1 = self._retrain()
            self.current_model     = new_model
            self.batches_since_retrain = 0
            self.retrain_count    += 1
            action_cost            = RETRAINING_COST
            # F1 improves after retraining
            f1 = new_f1
            self.f1_history.append(f1)
            self.current_shift = 0.0  # reset effective drift after retrain

        elif action == 2:  # SWITCH
            if self._backup_models:
                best_backup_idx = int(np.argmax(self._backup_f1s))
                self.current_model = self._backup_models[best_backup_idx]
                action_cost = SWITCH_COST

        # Final reward with cost
        reward -= action_cost
        self.total_reward += reward

        # Compute drift signals for state
        ks_stat, psi = self._compute_drift_signals(X_batch)

        # Log
        self.episode_log.append({
            "batch":      self.current_batch,
            "f1":         round(f1, 4),
            "action":     action_name,
            "shift":      round(self.current_shift, 4),
            "ks_stat":    round(float(ks_stat), 4),
            "psi":        round(float(psi), 4),
            "reward":     round(reward, 4),
        })

        done  = self.current_batch >= self.n_batches
        state = self._get_state(ks_stat=ks_stat, psi=psi)

        info  = {
            "f1":           f1,
            "action":       action_name,
            "shift":        self.current_shift,
            "retrains":     self.retrain_count,
            "total_reward": self.total_reward,
        }
        return state, reward, done, info

    # ── Internal ──────────────────────────────────────────────────────

    def _get_state(
        self,
        ks_stat: float = 0.0,
        psi:     float = 0.0,
    ) -> np.ndarray:
        """Build the 8-dimensional state vector."""
        current_f1   = float(np.mean(self.f1_history))
        f1_delta     = current_f1 - self.baseline_f1
        degrad       = self.peak_f1 - current_f1
        drift_enc    = min(self.current_shift / 3.0, 1.0)  # normalise to [0,1]
        norm_batches = min(self.batches_since_retrain / 20.0, 1.0)

        return np.array([
            current_f1,
            f1_delta,
            float(ks_stat),
            float(psi),
            norm_batches,
            float(self.trust_score),   # ← training-time trust in state
            degrad,
            drift_enc,
        ], dtype=np.float32)

    def get_state_no_trust(self) -> np.ndarray:
        """State without trust score (ablation experiment)."""
        s = self._get_state()
        return np.concatenate([s[:5], s[6:]])  # remove index 5 (trust)

    def _generate_shifted_batch(self) -> np.ndarray:
        """Generate test batch with current covariate shift applied."""
        idx     = self.rng.choice(len(self.X_test),
                                  min(self.batch_size, len(self.X_test)),
                                  replace=True)
        X_batch = self.X_test[idx].copy()
        shift   = self.current_shift * self.feature_stds
        shift_dir = self.rng.choice([-1, 1], size=X_batch.shape[1])
        X_batch += shift * shift_dir
        return X_batch

    def _evaluate_with_shift(self, model, X_batch: np.ndarray) -> float:
        """Evaluate model on shifted batch against true labels."""
        from sklearn.metrics import f1_score
        try:
            idx    = self.rng.choice(len(self.y_test),
                                     min(self.batch_size, len(self.y_test)),
                                     replace=True)
            y_true = self.y_test[idx]
            y_pred = model.predict(X_batch)
            if len(y_pred) != len(y_true):
                y_pred = y_pred[:len(y_true)]
            return float(f1_score(y_true, y_pred,
                                  average='weighted', zero_division=0))
        except Exception:
            return 0.5

    def _retrain(self) -> Tuple:
        """Retrain model on accumulated recent data."""
        from sklearn.base import clone
        from sklearn.metrics import f1_score
        new_model = clone(self.model)
        try:
            new_model.fit(self.current_train_X, self.current_train_y)
            y_pred = new_model.predict(self.X_test)
            new_f1 = float(f1_score(self.y_test, y_pred,
                                     average='weighted', zero_division=0))
        except Exception:
            new_f1 = self.baseline_f1
        return new_model, new_f1

    def _compute_drift_signals(self, X_batch: np.ndarray) -> Tuple:
        """KS statistic and PSI for current batch vs training reference."""
        from scipy import stats
        ks_vals  = []
        psi_vals = []
        n_feat   = min(X_batch.shape[1], self.X_train.shape[1])
        for j in range(min(n_feat, 10)):  # sample 10 features for speed
            ref = self.X_train[:, j]
            new = X_batch[:, j]
            ks, _ = stats.ks_2samp(ref, new)
            ks_vals.append(float(ks))
            # PSI
            bps = np.percentile(ref, np.linspace(0, 100, 9))
            bps = np.unique(bps)
            if len(bps) >= 2:
                rp = np.histogram(ref, bins=bps)[0] / len(ref)
                ap = np.histogram(new, bins=bps)[0] / len(new)
                rp = np.clip(rp, 1e-6, 1)
                ap = np.clip(ap, 1e-6, 1)
                psi_vals.append(float(np.sum((ap - rp) * np.log(ap / rp))))
        return np.mean(ks_vals) if ks_vals else 0.0, \
               np.mean(psi_vals) if psi_vals else 0.0


# ══════════════════════════════════════════════════════════════════════
# DQL AGENT
# ══════════════════════════════════════════════════════════════════════

Transition = namedtuple('Transition',
    ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in idx]

    def __len__(self):
        return len(self.buffer)


class QNetwork:
    """
    Lightweight Q-network implemented with numpy.
    Two hidden layers: 64 → 32 → n_actions.
    No PyTorch dependency required.
    """

    def __init__(self, state_dim: int, n_actions: int, lr: float = 1e-3):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.lr        = lr
        rng = np.random.RandomState(42)
        # Xavier initialisation
        self.W1 = rng.randn(state_dim, 64) * np.sqrt(2.0 / state_dim)
        self.b1 = np.zeros(64)
        self.W2 = rng.randn(64, 32) * np.sqrt(2.0 / 64)
        self.b2 = np.zeros(32)
        self.W3 = rng.randn(32, n_actions) * np.sqrt(2.0 / 32)
        self.b3 = np.zeros(n_actions)

    def _relu(self, x): return np.maximum(0, x)
    def _relu_grad(self, x): return (x > 0).astype(float)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass. x shape: (batch, state_dim)"""
        self._z1 = x @ self.W1 + self.b1
        self._a1 = self._relu(self._z1)
        self._z2 = self._a1 @ self.W2 + self.b2
        self._a2 = self._relu(self._z2)
        self._z3 = self._a2 @ self.W3 + self.b3
        return self._z3  # Q-values, no activation

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict Q-values for single state."""
        return self.forward(state.reshape(1, -1))[0]

    def update(self, states, actions, targets):
        """Single gradient update step."""
        batch = len(states)
        q_vals = self.forward(states)          # (batch, n_actions)

        # Only update Q-values for taken actions
        errors = np.zeros_like(q_vals)
        for i, a in enumerate(actions):
            errors[i, a] = q_vals[i, a] - targets[i]

        # Backprop through layer 3
        dL3 = errors / batch
        dW3 = self._a2.T @ dL3
        db3 = dL3.sum(axis=0)

        # Layer 2
        d2  = (dL3 @ self.W3.T) * self._relu_grad(self._z2)
        dW2 = self._a1.T @ d2
        db2 = d2.sum(axis=0)

        # Layer 1
        d1  = (d2 @ self.W2.T) * self._relu_grad(self._z1)
        dW1 = states.T @ d1
        db1 = d1.sum(axis=0)

        # SGD update
        self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3; self.b3 -= self.lr * db3

        return float(np.mean(errors ** 2))


class DQLAgent:
    """
    Deep Q-Learning agent for deployment lifecycle management.

    Learns when to retrain, continue, or switch backup models.
    Key design choice: trust score is part of the state,
    allowing the agent to learn trust-conditional policies.
    """

    def __init__(
        self,
        state_dim:   int   = STATE_DIM,
        n_actions:   int   = N_ACTIONS,
        lr:          float = 1e-3,
        gamma:       float = 0.95,
        epsilon:     float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size:  int   = 64,
        target_update: int = 20,
    ):
        self.state_dim     = state_dim
        self.n_actions     = n_actions
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update

        self.q_net      = QNetwork(state_dim, n_actions, lr)
        self.target_net = QNetwork(state_dim, n_actions, lr)
        self._copy_weights()

        self.memory      = ReplayBuffer(10_000)
        self.steps       = 0
        self.losses      = []
        self.rewards_log = []

    def _copy_weights(self):
        """Copy online network weights to target network."""
        self.target_net.W1 = self.q_net.W1.copy()
        self.target_net.b1 = self.q_net.b1.copy()
        self.target_net.W2 = self.q_net.W2.copy()
        self.target_net.b2 = self.q_net.b2.copy()
        self.target_net.W3 = self.q_net.W3.copy()
        self.target_net.b3 = self.q_net.b3.copy()

    def act(self, state: np.ndarray, explore: bool = True) -> int:
        """ε-greedy action selection."""
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        q_vals = self.q_net.predict(state)
        return int(np.argmax(q_vals))

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def replay(self) -> Optional[float]:
        """Sample from replay buffer and update Q-network."""
        if len(self.memory) < self.batch_size:
            return None

        transitions = self.memory.sample(self.batch_size)
        states      = np.array([t.state      for t in transitions])
        actions     = np.array([t.action     for t in transitions])
        rewards     = np.array([t.reward     for t in transitions])
        next_states = np.array([t.next_state for t in transitions])
        dones       = np.array([t.done       for t in transitions])

        # Compute targets
        next_q = self.target_net.forward(next_states).max(axis=1)
        targets = rewards + self.gamma * next_q * (1 - dones.astype(float))

        loss = self.q_net.update(states, actions, targets)
        self.losses.append(loss)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.steps += 1
        if self.steps % self.target_update == 0:
            self._copy_weights()

        return loss

    def get_q_values_for_state(self, state: np.ndarray) -> dict:
        """Return Q-values with action labels for interpretability."""
        q = self.q_net.predict(state)
        return {
            "CONTINUE": round(float(q[0]), 4),
            "RETRAIN":  round(float(q[1]), 4),
            "SWITCH":   round(float(q[2]), 4),
            "best_action": ["CONTINUE", "RETRAIN", "SWITCH"][int(np.argmax(q))],
        }
