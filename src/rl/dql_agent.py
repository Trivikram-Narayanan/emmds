"""
EMMDS Phase 2: DQL Agent
========================
Deep Q-Learning agent for retraining timing.

Architecture:
  Q-network: 3-layer MLP
    Input:  state (7-dim)
    Hidden: [64, 64] with ReLU
    Output: Q-values for each action (3-dim)

Training:
  - Experience replay buffer (capacity 10,000)
  - Mini-batch gradient descent (batch=64)
  - Target network (updated every 100 steps)
  - Epsilon-greedy exploration (decaying from 1.0 to 0.05)
  - Discount factor γ = 0.95

Key design decision:
  The trust score at training time is part of the state vector.
  The hypothesis is that the agent will learn different Q-value
  patterns for low-trust vs high-trust starting conditions.
  This is what we measure in the analysis phase.
"""

import numpy as np
import json
from pathlib import Path
from collections import deque
from typing import List, Tuple, Optional


class QNetwork:
    """
    3-layer MLP Q-network implemented in pure numpy.
    No deep learning frameworks needed — this keeps
    the dependency footprint minimal and the code auditable.

    For a production implementation, use PyTorch.
    For research demonstration with sklearn models, numpy is sufficient.
    """

    def __init__(
        self,
        input_dim:   int,
        hidden_dim:  int,
        output_dim:  int,
        lr:          float = 0.001,
        random_state: int  = 42,
    ):
        rng = np.random.RandomState(random_state)

        # Xavier initialisation
        def xavier(fan_in, fan_out):
            bound = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-bound, bound, (fan_in, fan_out))

        # Weights and biases
        self.W1 = xavier(input_dim,  hidden_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = xavier(hidden_dim, hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        self.W3 = xavier(hidden_dim, output_dim)
        self.b3 = np.zeros(output_dim)

        self.lr = lr

        # Adam optimiser state
        self._t    = 0
        self._beta1 = 0.9
        self._beta2 = 0.999
        self._eps   = 1e-8
        params = [self.W1,self.b1,self.W2,self.b2,self.W3,self.b3]
        self._m = [np.zeros_like(p) for p in params]
        self._v = [np.zeros_like(p) for p in params]

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _relu_grad(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass. x: (batch, input_dim) → (batch, output_dim)."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        self._cache_x  = x
        self._z1 = x @ self.W1 + self.b1
        self._a1 = self._relu(self._z1)
        self._z2 = self._a1 @ self.W2 + self.b2
        self._a2 = self._relu(self._z2)
        self._z3 = self._a2 @ self.W3 + self.b3
        return self._z3  # Q-values (no activation on output)

    def backward(self, targets: np.ndarray) -> float:
        """
        Compute MSE loss and update weights via Adam.
        targets: (batch, output_dim) — target Q-values.
        """
        batch_size = targets.shape[0]
        preds   = self._z3
        loss    = float(np.mean((preds - targets) ** 2))

        # Backprop
        dL_dz3  = 2.0 * (preds - targets) / batch_size
        dL_dW3  = self._a2.T @ dL_dz3
        dL_db3  = dL_dz3.sum(axis=0)

        dL_da2  = dL_dz3 @ self.W3.T
        dL_dz2  = dL_da2 * self._relu_grad(self._z2)
        dL_dW2  = self._a1.T @ dL_dz2
        dL_db2  = dL_dz2.sum(axis=0)

        dL_da1  = dL_dz2 @ self.W2.T
        dL_dz1  = dL_da1 * self._relu_grad(self._z1)
        dL_dW1  = self._cache_x.T @ dL_dz1
        dL_db1  = dL_dz1.sum(axis=0)

        grads = [dL_dW1,dL_db1,dL_dW2,dL_db2,dL_dW3,dL_db3]
        params = [self.W1,self.b1,self.W2,self.b2,self.W3,self.b3]

        # Adam update
        self._t += 1
        for i,(p,g) in enumerate(zip(params, grads)):
            self._m[i] = self._beta1 * self._m[i] + (1-self._beta1) * g
            self._v[i] = self._beta2 * self._v[i] + (1-self._beta2) * g**2
            m_hat = self._m[i] / (1 - self._beta1**self._t)
            v_hat = self._v[i] / (1 - self._beta2**self._t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self._eps)

        return loss

    def copy_weights_from(self, other: "QNetwork") -> None:
        """Copy weights from another QNetwork (for target network update)."""
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()
        self.W3 = other.W3.copy()
        self.b3 = other.b3.copy()

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Return Q-values for a state (inference only)."""
        return self.forward(state)

    def get_weights(self) -> dict:
        return {
            'W1': self.W1.tolist(), 'b1': self.b1.tolist(),
            'W2': self.W2.tolist(), 'b2': self.b2.tolist(),
            'W3': self.W3.tolist(), 'b3': self.b3.tolist(),
        }


class ReplayBuffer:
    """Experience replay buffer for DQL."""

    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        states      = np.stack([b[0] for b in batch])
        actions     = np.array([b[1] for b in batch])
        rewards     = np.array([b[2] for b in batch])
        next_states = np.stack([b[3] for b in batch])
        dones       = np.array([b[4] for b in batch], dtype=float)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


class DQLAgent:
    """
    Deep Q-Learning agent for deployment lifecycle management.

    The agent learns WHEN to retrain a deployed model by balancing:
      - The cost of degraded predictions (from not retraining)
      - The cost of retraining (compute, downtime)

    Research contribution:
      Trust score at training time is part of the state.
      We analyse whether the learned policy is trust-dependent.
    """

    def __init__(
        self,
        state_dim:      int   = 7,
        n_actions:      int   = 3,
        hidden_dim:     int   = 64,
        lr:             float = 0.001,
        gamma:          float = 0.95,
        epsilon_start:  float = 1.0,
        epsilon_min:    float = 0.05,
        epsilon_decay:  float = 0.995,
        batch_size:     int   = 64,
        buffer_capacity:int   = 10_000,
        target_update:  int   = 100,
        random_state:   int   = 42,
    ):
        np.random.seed(random_state)

        self.state_dim    = state_dim
        self.n_actions    = n_actions
        self.gamma        = gamma
        self.epsilon      = epsilon_start
        self.epsilon_min  = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size   = batch_size
        self.target_update = target_update

        # Q-network and target network
        self.q_net      = QNetwork(state_dim, hidden_dim, n_actions, lr, random_state)
        self.target_net = QNetwork(state_dim, hidden_dim, n_actions, lr, random_state)
        self.target_net.copy_weights_from(self.q_net)

        self.buffer = ReplayBuffer(buffer_capacity)

        self._step_count  = 0
        self.loss_history: List[float] = []
        self.reward_history: List[float] = []

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        q_vals = self.q_net.predict(state)
        return int(np.argmax(q_vals))

    def push_experience(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    def update(self) -> Optional[float]:
        """
        Sample a mini-batch and perform one gradient update.
        Returns loss or None if buffer not ready.
        """
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = \
            self.buffer.sample(self.batch_size)

        # Current Q-values
        self.q_net.forward(states)
        current_q = self.q_net._z3.copy()  # (batch, n_actions)

        # Target Q-values using Bellman equation
        next_q_target = self.target_net.forward(next_states)
        max_next_q    = next_q_target.max(axis=1)
        targets       = current_q.copy()

        for i in range(self.batch_size):
            td_target = rewards[i]
            if not dones[i]:
                td_target += self.gamma * max_next_q[i]
            targets[i, actions[i]] = td_target

        # Gradient update
        self.q_net.forward(states)   # Re-forward to set cache
        loss = self.q_net.backward(targets)
        self.loss_history.append(loss)

        # Decay epsilon
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)

        # Update target network
        self._step_count += 1
        if self._step_count % self.target_update == 0:
            self.target_net.copy_weights_from(self.q_net)

        return loss

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for analysis (greedy, no exploration)."""
        return self.q_net.predict(state).flatten()

    def get_policy_action(self, state: np.ndarray) -> int:
        """Greedy action (for evaluation, no exploration)."""
        return int(np.argmax(self.q_net.predict(state)))

    def save(self, path: str) -> None:
        """Persist agent weights."""
        data = {
            "q_net":         self.q_net.get_weights(),
            "epsilon":       self.epsilon,
            "step_count":    self._step_count,
            "loss_history":  self.loss_history[-100:],
            "reward_history": self.reward_history[-100:],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def training_summary(self) -> dict:
        """Return training statistics."""
        if not self.loss_history:
            return {}
        return {
            "steps":         self._step_count,
            "epsilon":       round(self.epsilon, 4),
            "mean_loss_last50": round(float(np.mean(self.loss_history[-50:])), 6),
            "mean_reward_last20": round(float(np.mean(self.reward_history[-20:])), 4)
                                    if self.reward_history else 0.0,
        }
