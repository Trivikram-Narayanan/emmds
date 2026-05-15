"""
EMMDS Phase 2: DQL Deployment Environment
==========================================
MDP formulation for the retraining timing problem.

This is a Partially Observable Markov Decision Process (POMDP):
  - We observe drift signals and performance metrics
  - We do NOT observe the true future data distribution
  - We must decide when to retrain under this uncertainty

State space (7-dimensional, continuous):
  s = [current_f1, f1_delta_from_peak, ks_stat_mean,
       psi_score, batches_since_retrain,
       trust_score_at_training, cumulative_cost]

Action space (discrete, 3 actions):
  0: CONTINUE  — keep deploying, do nothing
  1: RETRAIN   — retrain on most recent data window
  2: FALLBACK  — switch to backup (simpler) model

Reward:
  r_t = α × current_f1 - β × retraining_cost × I(retrained)
  where α=1.0, β=λ (tunable cost parameter)
  This creates the fundamental tension: performance vs retraining cost

Episode:
  One deployment scenario = one dataset, one base model,
  N batches of incoming data with controlled drift injection.
  Episode ends after max_batches or when model is retrained twice.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, brier_score_loss
from sklearn.base import clone
from scipy import stats


class DeploymentEnvironment:
    """
    Simulates a production deployment environment for RL training.

    The environment:
    1. Trains a base model on clean data
    2. Generates a sequence of batches with injected covariate shift
    3. At each step the agent observes the state and chooses an action
    4. Reward reflects the performance/cost tradeoff

    Design principle: the trust score at training time is part of the
    state. This allows the agent to learn that low-trust models need
    different policies than high-trust models — which is our hypothesis.
    """

    # State dimensions
    STATE_DIM = 7
    # Action space
    N_ACTIONS = 3  # 0=continue, 1=retrain, 2=fallback

    def __init__(
        self,
        retraining_cost: float = 0.15,
        max_batches: int = 20,
        batch_size: int = 100,
        drift_schedule: str = "gradual",  # gradual | sudden | cyclic | none
        random_state: int = 42,
    ):
        self.retraining_cost = retraining_cost
        self.max_batches     = max_batches
        self.batch_size      = batch_size
        self.drift_schedule  = drift_schedule
        self.rng             = np.random.RandomState(random_state)

        # Will be set by reset()
        self._X_train:  Optional[np.ndarray] = None
        self._X_pool:   Optional[np.ndarray] = None  # Held-out for batches
        self._y_pool:   Optional[np.ndarray] = None
        self._model     = None
        self._backup    = None
        self._scaler    = None
        self._baseline_f1   = 0.0
        self._trust_score   = 0.5
        self._current_model = None

        # Episode state
        self._batch_num:     int   = 0
        self._peak_f1:       float = 0.0
        self._last_retrain:  int   = 0
        self._retrain_count: int   = 0
        self._cumulative_reward: float = 0.0

        # Drift state
        self._shift_magnitude: float = 0.0

    def setup(
        self,
        X_train: np.ndarray,
        X_test:  np.ndarray,
        y_train: np.ndarray,
        y_test:  np.ndarray,
        model,
        backup_model,
        trust_score: float,
        baseline_f1: float,
        scaler: StandardScaler,
    ) -> None:
        """Load a prepared dataset and models into the environment."""
        self._X_train  = X_train
        self._X_pool   = X_test
        self._y_pool   = y_test
        self._model    = clone(model)
        self._model.fit(X_train, y_train)
        self._backup   = clone(backup_model)
        self._backup.fit(X_train, y_train)
        self._scaler       = scaler
        self._trust_score  = float(trust_score)
        self._baseline_f1  = float(baseline_f1)
        self._current_model = self._model

    def reset(self) -> np.ndarray:
        """Reset episode state. Returns initial observation."""
        self._batch_num         = 0
        self._peak_f1           = self._baseline_f1
        self._last_retrain      = 0
        self._retrain_count     = 0
        self._cumulative_reward = 0.0
        self._shift_magnitude   = 0.0
        self._current_model     = self._model
        return self._get_state(self._baseline_f1)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one environment step.

        Args:
            action: 0=continue, 1=retrain, 2=fallback

        Returns:
            (next_state, reward, done, info)
        """
        self._batch_num += 1

        # Update drift
        self._update_drift()

        # Get current batch
        X_batch, y_batch = self._get_batch()

        # Apply covariate shift
        X_shifted = self._apply_shift(X_batch)

        # Compute current performance
        try:
            y_pred   = self._current_model.predict(X_shifted)
            curr_f1  = float(f1_score(y_batch, y_pred,
                                       average='weighted', zero_division=0))
        except Exception:
            curr_f1 = 0.0

        # Update peak
        if curr_f1 > self._peak_f1:
            self._peak_f1 = curr_f1

        # Execute action
        cost = 0.0
        if action == 1:  # RETRAIN
            self._retrain_count += 1
            self._last_retrain   = self._batch_num
            cost = self.retraining_cost
            # Retrain on recent window (simulate getting new labelled data)
            window_X, window_y = self._get_recent_window()
            try:
                new_model = clone(self._model)
                new_model.fit(window_X, window_y)
                self._current_model = new_model
                # Re-evaluate after retrain
                y_pred2 = self._current_model.predict(X_shifted)
                curr_f1 = float(f1_score(y_batch, y_pred2,
                                          average='weighted', zero_division=0))
            except Exception:
                pass

        elif action == 2:  # FALLBACK
            self._current_model = self._backup
            cost = 0.05  # Small switching cost

        # Reward: performance minus retraining cost
        reward = float(curr_f1 - cost)
        self._cumulative_reward += reward

        # Next state
        f1_delta = curr_f1 - self._peak_f1
        ks_stat, psi = self._compute_drift_signals(X_shifted, X_batch)
        next_state = self._get_state_from_values(
            curr_f1, f1_delta, ks_stat, psi)

        # Done conditions
        done = (self._batch_num >= self.max_batches or
                self._retrain_count >= 3)

        info = {
            "batch":         self._batch_num,
            "f1":            curr_f1,
            "shift":         self._shift_magnitude,
            "action":        ["CONTINUE","RETRAIN","FALLBACK"][action],
            "cost":          cost,
            "cumulative_r":  self._cumulative_reward,
        }
        return next_state, reward, done, info

    # ── State construction ────────────────────────────────────────────

    def _get_state(self, current_f1: float) -> np.ndarray:
        """Initial state at start of episode."""
        return np.array([
            current_f1,           # current F1
            0.0,                  # delta from peak
            0.0,                  # KS statistic
            0.0,                  # PSI score
            0.0,                  # batches since retrain (normalised)
            self._trust_score,    # training-time trust (KEY: this is the novel state feature)
            0.0,                  # cumulative cost
        ], dtype=np.float32)

    def _get_state_from_values(
        self, curr_f1: float, f1_delta: float,
        ks_stat: float, psi: float
    ) -> np.ndarray:
        batches_since = (self._batch_num - self._last_retrain) / self.max_batches
        cum_cost = self._retrain_count * self.retraining_cost / 3.0  # normalise
        return np.array([
            np.clip(curr_f1,      0, 1),
            np.clip(f1_delta,    -1, 0),
            np.clip(ks_stat,      0, 1),
            np.clip(psi,          0, 1),
            np.clip(batches_since,0, 1),
            np.clip(self._trust_score, 0, 1),
            np.clip(cum_cost,     0, 1),
        ], dtype=np.float32)

    # ── Drift mechanics ───────────────────────────────────────────────

    def _update_drift(self):
        """Update shift magnitude based on drift schedule."""
        t = self._batch_num / self.max_batches
        if self.drift_schedule == "gradual":
            self._shift_magnitude = t * 2.5
        elif self.drift_schedule == "sudden":
            self._shift_magnitude = 2.5 if t > 0.4 else 0.0
        elif self.drift_schedule == "cyclic":
            self._shift_magnitude = 1.5 * abs(np.sin(t * np.pi * 2))
        elif self.drift_schedule == "none":
            self._shift_magnitude = 0.0
        else:
            self._shift_magnitude = t * 2.0

    def _apply_shift(self, X: np.ndarray) -> np.ndarray:
        """Apply covariate shift to a batch."""
        if self._shift_magnitude < 0.01:
            return X
        feature_stds = self._X_train.std(axis=0) + 1e-8
        direction    = self.rng.choice([-1, 1], size=X.shape[1])
        shift        = self._shift_magnitude * feature_stds * direction
        return X + shift

    def _get_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a batch from the pool."""
        n = len(self._X_pool)
        idx = self.rng.choice(n, size=min(self.batch_size, n), replace=False)
        return self._X_pool[idx], self._y_pool[idx]

    def _get_recent_window(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get recent data window for retraining."""
        n   = len(self._X_pool)
        idx = self.rng.choice(n, size=min(200, n), replace=False)
        return self._X_pool[idx], self._y_pool[idx]

    def _compute_drift_signals(
        self, X_shifted: np.ndarray, X_clean: np.ndarray
    ) -> Tuple[float, float]:
        """Compute KS statistic and PSI between clean and shifted."""
        ks_vals  = []
        psi_vals = []
        for j in range(min(X_shifted.shape[1], 10)):  # Sample features
            ref = self._X_train[:, j]
            new = X_shifted[:, j]
            ks_stat, _ = stats.ks_2samp(ref, new)
            ks_vals.append(float(ks_stat))
            # PSI
            bp = np.percentile(ref, np.linspace(0,100,9))
            bp = np.unique(bp)
            if len(bp) > 1:
                r = np.histogram(ref, bins=bp)[0]/len(ref)
                a = np.histogram(new, bins=bp)[0]/len(new)
                r,a = np.clip(r,1e-6,1), np.clip(a,1e-6,1)
                psi_vals.append(float(np.sum((a-r)*np.log(a/r))))
        ks  = float(np.mean(ks_vals))  if ks_vals  else 0.0
        psi = float(np.mean(psi_vals)) if psi_vals else 0.0
        return np.clip(ks,0,1), np.clip(abs(psi),0,1)

    def get_state_dim(self) -> int:
        return self.STATE_DIM

    def get_n_actions(self) -> int:
        return self.N_ACTIONS
