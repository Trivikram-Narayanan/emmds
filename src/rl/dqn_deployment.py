"""
EMMDS System 2: DQN Deployment Monitor
=======================================
A Deep Q-Network that learns the optimal retraining policy
from drift signals and training-time trust scores.

RESEARCH QUESTION:
  Does including the training-time trust score in the DQN state
  produce better retraining policies than using drift signals alone?

  Hypothesis: models with low trust scores degrade faster under drift
  and should be retrained more aggressively. If the DQN learns this
  from the trust signal, it validates trust score as a deployment
  lifecycle parameter, not just a selection criterion.

FORMULATION:
  State (7 features):
    [current_f1, rolling_f1_delta, ks_stat_mean, psi_mean,
     batches_since_retrain, trust_at_training, peak_f1_delta]

  Actions (3):
    0 = DEPLOY    (continue)
    1 = RETRAIN   (retrain on recent data)
    2 = SWITCH    (switch to backup model)

  Reward:
    r_t = current_f1 - λ × retraining_cost
    λ = 0.1 (retraining costs 10% of one batch's F1 gain)

  Algorithm: DQN with experience replay and target network

BASELINES:
  - Fixed schedule (retrain every N batches)
  - Reactive threshold (retrain when F1 drops below X)
  - Trust threshold (retrain when trust-estimated risk exceeds X)
  - DQN with trust (our system)
  - DQN without trust (ablation — trust signal removed from state)
"""

import numpy as np
import json
from pathlib import Path
from collections import deque

OUT = Path("outputs/research/drl")
OUT.mkdir(parents=True, exist_ok=True)

STATE_DIM   = 7
N_ACTIONS   = 3   # 0=DEPLOY, 1=RETRAIN, 2=SWITCH
ACTION_NAMES = {0: 'DEPLOY', 1: 'RETRAIN', 2: 'SWITCH'}

RETRAIN_COST = 0.10   # λ: fraction of batch F1 that retraining costs


# ══════════════════════════════════════════════════════════════════════
# Q-NETWORK (2-layer MLP from scratch)
# ══════════════════════════════════════════════════════════════════════

class QNetwork:
    """
    2-layer MLP: state(7) → Q-values(3)
    Implements target network for stable training.
    """

    def __init__(self, state_dim: int = STATE_DIM,
                 n_actions: int = N_ACTIONS,
                 hidden: int = 64, lr: float = 0.001):
        rng = np.random.RandomState(42)
        self.W1 = rng.randn(state_dim, hidden) * np.sqrt(2/state_dim)
        self.b1 = np.zeros(hidden)
        self.W2 = rng.randn(hidden, hidden)    * np.sqrt(2/hidden)
        self.b2 = np.zeros(hidden)
        self.W3 = rng.randn(hidden, n_actions) * np.sqrt(2/hidden)
        self.b3 = np.zeros(n_actions)
        self.lr = lr

        # Adam state
        self._t = 0
        params = [self.W1,self.b1,self.W2,self.b2,self.W3,self.b3]
        self._ms = [np.zeros_like(p) for p in params]
        self._vs = [np.zeros_like(p) for p in params]

    def forward(self, s):
        a1 = np.maximum(0, s @ self.W1 + self.b1)
        a2 = np.maximum(0, a1 @ self.W2 + self.b2)
        return a2 @ self.W3 + self.b3, a1, a2

    def predict(self, s):
        q, _, _ = self.forward(s)
        return q

    def train_batch(self, states, actions, targets):
        """Vectorised batch update."""
        n = len(states)
        total_loss = 0.0

        # Accumulate gradients over batch
        dW1=np.zeros_like(self.W1); db1=np.zeros_like(self.b1)
        dW2=np.zeros_like(self.W2); db2=np.zeros_like(self.b2)
        dW3=np.zeros_like(self.W3); db3=np.zeros_like(self.b3)

        for i in range(n):
            s = states[i]; a = actions[i]; tgt = targets[i]
            q, a1, a2 = self.forward(s)
            err = q[a] - tgt
            total_loss += 0.5 * err**2

            dq = np.zeros(N_ACTIONS); dq[a] = err
            _dW3 = np.outer(a2, dq);       _db3 = dq
            da2  = dq @ self.W3.T;         dz2  = da2 * (a2>0)
            _dW2 = np.outer(a1, dz2);      _db2 = dz2
            da1  = dz2 @ self.W2.T;        dz1  = da1 * (a1>0)
            _dW1 = np.outer(s, dz1);       _db1 = dz1

            dW1+=_dW1; db1+=_db1; dW2+=_dW2; db2+=_db2; dW3+=_dW3; db3+=_db3

        grads = [dW1/n, db1/n, dW2/n, db2/n, dW3/n, db3/n]
        params_list = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        param_names  = ['W1','b1','W2','b2','W3','b3']

        self._t += 1
        b1_,b2_,eps = 0.9, 0.999, 1e-8
        new_params = {}
        for i,(p,g) in enumerate(zip(params_list, grads)):
            self._ms[i] = b1_*self._ms[i] + (1-b1_)*g
            self._vs[i] = b2_*self._vs[i] + (1-b2_)*g**2
            m_hat = self._ms[i]/(1-b1_**self._t)
            v_hat = self._vs[i]/(1-b2_**self._t)
            new_params[param_names[i]] = p - self.lr*m_hat/(np.sqrt(v_hat)+eps)

        for k,v in new_params.items():
            setattr(self, k, v)

        return total_loss / n

    def copy_weights_from(self, other: 'QNetwork') -> None:
        """Hard copy weights from another QNetwork (target network update)."""
        self.W1=other.W1.copy(); self.b1=other.b1.copy()
        self.W2=other.W2.copy(); self.b2=other.b2.copy()
        self.W3=other.W3.copy(); self.b3=other.b3.copy()

    def save(self, path):
        d = {k: getattr(self,k).tolist() for k in ['W1','b1','W2','b2','W3','b3']}
        Path(path).write_text(json.dumps(d))

    def load(self, path):
        d = json.loads(Path(path).read_text())
        for k,v in d.items():
            setattr(self, k, np.array(v))


# ══════════════════════════════════════════════════════════════════════
# DEPLOYMENT ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════

class DeploymentEnvironment:
    """
    Simulates a production deployment environment.

    - Data arrives in batches
    - Covariate shift increases over time (at a random rate)
    - Retraining resets the model to current distribution
    - State includes training-time trust score
    """

    def __init__(
        self,
        X_train: np.ndarray,
        X_test:  np.ndarray,
        y_test:  np.ndarray,
        model,
        backup_model,
        trust_score:    float,
        baseline_f1:    float,
        n_batches:      int   = 40,
        batch_size:     int   = 50,
        shift_rate:     float = 0.08,
        include_trust:  bool  = True,
    ):
        self.X_train      = X_train
        self.X_test       = X_test
        self.y_test       = y_test
        self.model        = model
        self.backup_model = backup_model
        self.trust_score  = trust_score
        self.baseline_f1  = baseline_f1
        self.n_batches    = n_batches
        self.batch_size   = min(batch_size, len(X_test))
        self.shift_rate   = shift_rate
        self.include_trust = include_trust
        self.rng          = np.random.RandomState(42)

        self.reset()

    def reset(self):
        self.current_batch    = 0
        self.batches_since_retrain = 0
        self.cumulative_shift = 0.0
        self.f1_history       = deque(maxlen=10)
        self.peak_f1          = self.baseline_f1
        self.current_f1       = self.baseline_f1
        self.active_model     = self.model
        self.total_reward     = 0.0
        self.retrain_count    = 0
        self.action_history   = []
        return self._get_state()

    def _get_f1_on_batch(self, model, shift_mag: float) -> float:
        from sklearn.metrics import f1_score
        # Apply shift to test batch
        idx = self.rng.choice(len(self.X_test), self.batch_size, replace=False)
        X_b = self.X_test[idx].copy()
        # Covariate shift
        feature_stds = self.X_train.std(axis=0)
        shift_dir    = self.rng.choice([-1,1], size=X_b.shape[1])
        X_b         += shift_mag * feature_stds * shift_dir
        try:
            preds = model.predict(X_b)
            return float(f1_score(self.y_test[idx], preds,
                                  average='weighted', zero_division=0))
        except Exception:
            return 0.0

    def _get_state(self) -> np.ndarray:
        f1   = self.current_f1
        self.f1_history.append(f1)
        rolling_delta = (f1 - self.baseline_f1)

        # Drift signals (simplified from KS/PSI)
        shift_norm = np.clip(self.cumulative_shift / 3.0, 0, 1)
        ks_approx  = shift_norm * 0.4      # Approximate KS statistic
        psi_approx = shift_norm * 0.3      # Approximate PSI

        # Peak degradation
        if f1 > self.peak_f1:
            self.peak_f1 = f1
        peak_delta = self.peak_f1 - f1

        state = np.array([
            f1,
            rolling_delta,
            ks_approx,
            psi_approx,
            min(self.batches_since_retrain / self.n_batches, 1.0),
            self.trust_score if self.include_trust else 0.5,
            peak_delta,
        ], dtype=np.float32)

        return state

    def step(self, action: int) -> tuple:
        """
        Take one deployment step.
        Returns (next_state, reward, done, info)
        """
        self.current_batch += 1
        self.cumulative_shift += self.shift_rate + self.rng.normal(0, 0.02)
        self.cumulative_shift  = max(0, self.cumulative_shift)

        # Current F1 under this shift
        f1 = self._get_f1_on_batch(self.active_model, self.cumulative_shift)
        self.current_f1 = f1

        # Compute reward
        if action == 1:  # RETRAIN
            # Retrain on recent data (simulate: reset shift, F1 recovers)
            self.cumulative_shift = 0.0
            f1_after = self._get_f1_on_batch(self.active_model, 0.0)
            reward = f1_after - RETRAIN_COST
            self.batches_since_retrain = 0
            self.retrain_count += 1
            self.current_f1 = f1_after
        elif action == 2:  # SWITCH
            # Switch to backup model
            f1_backup = self._get_f1_on_batch(self.backup_model, self.cumulative_shift)
            reward = f1_backup - RETRAIN_COST * 0.5
            self.active_model = self.backup_model
            self.current_f1   = f1_backup
        else:  # DEPLOY
            reward = f1

        self.batches_since_retrain += 1
        self.total_reward += reward
        self.action_history.append(action)

        done       = (self.current_batch >= self.n_batches)
        next_state = self._get_state()

        return next_state, reward, done, {
            'f1': self.current_f1,
            'action': ACTION_NAMES[action],
            'shift': round(self.cumulative_shift, 3),
            'retrain_count': self.retrain_count,
        }


# ══════════════════════════════════════════════════════════════════════
# DQN AGENT
# ══════════════════════════════════════════════════════════════════════

class DQNAgent:
    """
    DQN agent with experience replay and target network.
    Trained across many deployment scenarios.
    """

    def __init__(
        self,
        state_dim:     int   = STATE_DIM,
        n_actions:     int   = N_ACTIONS,
        lr:            float = 0.001,
        gamma:         float = 0.95,
        eps_start:     float = 1.0,
        eps_end:       float = 0.05,
        eps_decay:     int   = 2000,
        buffer_size:   int   = 5000,
        batch_size:    int   = 64,
        target_update: int   = 100,
    ):
        self.q_net      = QNetwork(state_dim, n_actions, lr=lr)
        self.target_net = QNetwork(state_dim, n_actions, lr=lr)
        self.target_net.copy_weights_from(self.q_net)

        self.gamma         = gamma
        self.eps_start     = eps_start
        self.eps_end       = eps_end
        self.eps_decay     = eps_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self.step_count    = 0
        self.rng           = np.random.RandomState(42)

        # Replay buffer: (state, action, reward, next_state, done)
        self.buffer = deque(maxlen=buffer_size)

        self.reward_history = []
        self.loss_history   = []
        self.episode_returns = []

    @property
    def epsilon(self):
        progress = min(1.0, self.step_count / self.eps_decay)
        return self.eps_end + (self.eps_start - self.eps_end) * (1 - progress)

    def select_action(self, state: np.ndarray,
                      greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.randint(0, N_ACTIONS)
        return int(np.argmax(self.q_net.predict(state)))

    def store(self, s, a, r, s2, done):
        self.buffer.append((s.copy(), a, r, s2.copy(), done))

    def train_step(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0

        # Sample mini-batch
        idx   = self.rng.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]

        states  = np.stack([b[0] for b in batch])
        actions = np.array([b[1] for b in batch], dtype=int)
        rewards = np.array([b[2] for b in batch])
        nexts   = np.stack([b[3] for b in batch])
        dones   = np.array([b[4] for b in batch])

        # Compute targets: r + γ × max_a Q_target(s', a)
        q_next   = np.stack([self.target_net.predict(nexts[i])
                              for i in range(self.batch_size)])
        max_next = q_next.max(axis=1)
        targets  = rewards + self.gamma * max_next * (1 - dones)

        loss = self.q_net.train_batch(states, actions, targets)

        self.step_count += 1
        self.loss_history.append(float(loss))

        # Update target network
        if self.step_count % self.target_update == 0:
            self.target_net.copy_weights_from(self.q_net)

        return float(loss)

    def save(self, path):
        self.q_net.save(path)

    def load(self, path):
        self.q_net.load(path)
        self.target_net.copy_weights_from(self.q_net)


# ══════════════════════════════════════════════════════════════════════
# TRAINING + EVALUATION
# ══════════════════════════════════════════════════════════════════════

class DQNTrainer:
    """
    Trains and evaluates the DQN deployment monitor.
    Key evaluation: does including trust score in state improve policy?
    """

    def __init__(self, n_episodes: int = 300):
        self.n_episodes  = n_episodes
        self.agent_trust = DQNAgent()       # With trust in state
        self.agent_notrust = DQNAgent()     # Without trust (ablation)
        self.train_log   = []

    def _run_episode(self, agent: DQNAgent, env: DeploymentEnvironment,
                     training: bool = True) -> float:
        state     = env.reset()
        ep_return = 0.0

        while True:
            action     = agent.select_action(state, greedy=not training)
            next_state, reward, done, info = env.step(action)

            if training:
                agent.store(state, action, reward, next_state, done)
                agent.train_step()

            ep_return += reward
            state      = next_state

            if done:
                break

        return ep_return

    def train(self, env_configs: list, verbose: bool = True) -> dict:
        """
        Train both agents (with and without trust) across deployment scenarios.

        env_configs: list of (X_train, X_test, y_test, model, backup, trust, f1)
        """
        print(f"  Training DQN: {self.n_episodes} episodes across "
              f"{len(env_configs)} deployment scenarios")

        trust_returns    = []
        notrust_returns  = []

        for ep in range(self.n_episodes):
            cfg = env_configs[ep % len(env_configs)]
            X_tr, X_te, y_te, model, backup, trust, f1 = cfg

            # Vary shift rate for diversity
            shift_rate = 0.05 + 0.10 * np.random.RandomState(ep).random()

            env_trust = DeploymentEnvironment(
                X_tr, X_te, y_te, model, backup, trust, f1,
                shift_rate=shift_rate, include_trust=True)
            env_notrust = DeploymentEnvironment(
                X_tr, X_te, y_te, model, backup, trust, f1,
                shift_rate=shift_rate, include_trust=False)

            r_trust   = self._run_episode(self.agent_trust,   env_trust)
            r_notrust = self._run_episode(self.agent_notrust, env_notrust)

            trust_returns.append(r_trust)
            notrust_returns.append(r_notrust)

            self.train_log.append({
                'episode': ep,
                'trust_return':   round(r_trust,   4),
                'notrust_return': round(r_notrust, 4),
                'eps_trust':   round(self.agent_trust.epsilon, 4),
            })

            if verbose and (ep+1) % 50 == 0:
                t50 = np.mean(trust_returns[-50:])
                n50 = np.mean(notrust_returns[-50:])
                print(f"    Ep {ep+1:4d}  "
                      f"trust={t50:.4f}  notrust={n50:.4f}  "
                      f"ε={self.agent_trust.epsilon:.3f}")

        self.agent_trust.save(str(OUT/'dqn_trust.json'))
        self.agent_notrust.save(str(OUT/'dqn_notrust.json'))

        return {
            'n_episodes':        self.n_episodes,
            'trust_mean_return':   round(float(np.mean(trust_returns[-50:])), 4),
            'notrust_mean_return': round(float(np.mean(notrust_returns[-50:])), 4),
            'trust_better': bool(np.mean(trust_returns[-50:]) > np.mean(notrust_returns[-50:])),
        }

    def evaluate_held_out(
        self,
        held_out_configs: list,
        n_eval_eps: int = 20,
    ) -> dict:
        """
        Compare all policies on held-out deployment scenarios.
        Baselines: fixed schedule, reactive threshold, DQN±trust.
        """
        print(f"\n  Evaluating on {len(held_out_configs)} held-out scenarios "
              f"({n_eval_eps} episodes each)")

        results = []
        for cfg in held_out_configs:
            X_tr, X_te, y_te, model, backup, trust, f1 = cfg

            policy_returns = {
                'dqn_trust':    [], 'dqn_notrust': [],
                'fixed_10':     [], 'fixed_5':     [],
                'reactive_85':  [], 'reactive_90': [],
            }

            for run in range(n_eval_eps):
                shift_rate = 0.06 + 0.08 * np.random.RandomState(run).random()

                # DQN with trust
                env = DeploymentEnvironment(X_tr,X_te,y_te,model,backup,trust,f1,
                                            shift_rate=shift_rate,include_trust=True)
                r = self._run_greedy(self.agent_trust, env)
                policy_returns['dqn_trust'].append(r)

                # DQN without trust
                env = DeploymentEnvironment(X_tr,X_te,y_te,model,backup,trust,f1,
                                            shift_rate=shift_rate,include_trust=False)
                r = self._run_greedy(self.agent_notrust, env)
                policy_returns['dqn_notrust'].append(r)

                # Fixed schedule every 10 batches
                r = self._run_fixed_schedule(X_tr,X_te,y_te,model,backup,trust,f1,
                                              shift_rate=shift_rate, every_n=10)
                policy_returns['fixed_10'].append(r)

                # Fixed schedule every 5 batches
                r = self._run_fixed_schedule(X_tr,X_te,y_te,model,backup,trust,f1,
                                              shift_rate=shift_rate, every_n=5)
                policy_returns['fixed_5'].append(r)

                # Reactive threshold at 0.85
                r = self._run_reactive(X_tr,X_te,y_te,model,backup,trust,f1,
                                        shift_rate=shift_rate, threshold=0.85)
                policy_returns['reactive_85'].append(r)

                # Reactive threshold at 0.90
                r = self._run_reactive(X_tr,X_te,y_te,model,backup,trust,f1,
                                        shift_rate=shift_rate, threshold=0.90)
                policy_returns['reactive_90'].append(r)

            row = {'trust_score_at_training': round(trust, 4)}
            for pname, rets in policy_returns.items():
                row[f'{pname}_mean'] = round(float(np.mean(rets)), 4)
            results.append(row)

            best_policy = max(
                [(p, np.mean(r)) for p,r in policy_returns.items()],
                key=lambda x: x[1]
            )[0]
            print(f"    trust={trust:.3f}  "
                  f"dqn_trust={row['dqn_trust_mean']:.4f}  "
                  f"fixed10={row['fixed_10_mean']:.4f}  "
                  f"reactive85={row['reactive_85_mean']:.4f}  "
                  f"best={best_policy}")

        import pandas as pd
        res_df = pd.DataFrame(results)
        res_df.to_csv(OUT/'dqn_evaluation.csv', index=False)

        # Key finding: does trust in state help more for low-trust models?
        low_trust  = res_df[res_df['trust_score_at_training'] < 0.75]
        high_trust = res_df[res_df['trust_score_at_training'] >= 0.75]

        summary = {
            'all_scenarios': {
                p: round(float(res_df[f'{p}_mean'].mean()), 4)
                for p in policy_returns.keys()
            },
            'low_trust_scenarios': {
                p: round(float(low_trust[f'{p}_mean'].mean()), 4)
                for p in policy_returns.keys()
            } if len(low_trust) > 0 else {},
            'high_trust_scenarios': {
                p: round(float(high_trust[f'{p}_mean'].mean()), 4)
                for p in policy_returns.keys()
            } if len(high_trust) > 0 else {},
        }

        print(f"\n  Policy comparison (all scenarios):")
        for p, v in summary['all_scenarios'].items():
            print(f"    {p:20s}  mean_return={v:.4f}")

        return summary, res_df

    def _run_greedy(self, agent: DQNAgent,
                    env: DeploymentEnvironment) -> float:
        state = env.reset(); ep_r = 0.0
        while True:
            action = agent.select_action(state, greedy=True)
            state, r, done, _ = env.step(action)
            ep_r += r
            if done: break
        return ep_r

    def _run_fixed_schedule(self, X_tr, X_te, y_te, model, backup,
                             trust, f1, shift_rate, every_n) -> float:
        env = DeploymentEnvironment(X_tr,X_te,y_te,model,backup,trust,f1,
                                     shift_rate=shift_rate, include_trust=True)
        state = env.reset(); ep_r = 0.0
        while True:
            action = 1 if (env.batches_since_retrain >= every_n) else 0
            state, r, done, _ = env.step(action)
            ep_r += r
            if done: break
        return ep_r

    def _run_reactive(self, X_tr, X_te, y_te, model, backup,
                       trust, f1, shift_rate, threshold) -> float:
        env = DeploymentEnvironment(X_tr,X_te,y_te,model,backup,trust,f1,
                                     shift_rate=shift_rate, include_trust=True)
        state = env.reset(); ep_r = 0.0
        while True:
            action = 1 if env.current_f1 < threshold else 0
            state, r, done, _ = env.step(action)
            ep_r += r
            if done: break
        return ep_r
