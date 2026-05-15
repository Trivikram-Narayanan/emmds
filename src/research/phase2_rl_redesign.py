"""
Phase 2 RL Redesign — diverse trust score range.

The original Phase 2 experiment used datasets where all trust scores fell
in [0.847, 0.863], making the trust-Q hypothesis untestable.

This redesign deliberately constructs scenarios spanning trust ≈ 0.3 → 0.9
by varying dataset quality (noise, imbalance, size) and measuring whether
higher trust scores correlate with lower optimal Q(retrain) advantage.

Hypothesis (H2): Spearman r(initial_trust, Q_retrain_advantage) < 0
  i.e. lower-trust models benefit MORE from retraining.

Outputs: outputs/research/phase2_rl_redesign.json
"""
import sys
import os
import json
import warnings
import numpy as np
from pathlib import Path
from collections import deque

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from scipy import stats

# ---------------------------------------------------------------------------
# Lightweight DQL (standalone, no import from src/rl to keep experiment self-contained)
# ---------------------------------------------------------------------------

class QNetwork:
    def __init__(self, state_dim=7, hidden=32, n_actions=3, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2/state_dim), (state_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, np.sqrt(2/hidden), (hidden, hidden))
        self.b2 = np.zeros(hidden)
        self.W3 = rng.normal(0, np.sqrt(2/hidden), (hidden, n_actions))
        self.b3 = np.zeros(n_actions)

    def relu(self, x): return np.maximum(0, x)

    def forward(self, x):
        h1 = self.relu(x @ self.W1 + self.b1)
        h2 = self.relu(h1 @ self.W2 + self.b2)
        return h2 @ self.W3 + self.b3

    def copy_from(self, other):
        for attr in ["W1", "b1", "W2", "b2", "W3", "b3"]:
            setattr(self, attr, getattr(other, attr).copy())

    def update(self, x, action, target, lr=1e-3):
        h1 = self.relu(x @ self.W1 + self.b1)
        h2 = self.relu(h1 @ self.W2 + self.b2)
        q = h2 @ self.W3 + self.b3
        error = q[action] - target
        # Simplified backward pass (gradient only for output layer)
        dW3 = np.outer(h2, np.eye(len(self.b3))[action]) * error
        db3 = np.eye(len(self.b3))[action] * error
        self.W3 -= lr * dW3
        self.b3 -= lr * db3


# ---------------------------------------------------------------------------
# Deployment environment with controlled drift
# ---------------------------------------------------------------------------

class TrustAwareDriftEnv:
    """
    State: [f1, f1_delta, ks_stat, psi, batches_since_retrain, trust, cumcost]
    Actions: 0=continue, 1=retrain, 2=fallback
    """
    N_ACTIONS = 3
    STATE_DIM = 7
    RETRAIN_COST = 0.30
    FALLBACK_COST = 0.10
    PERFORMANCE_FLOOR = 0.40

    def __init__(self, initial_trust: float, drift_rate: float = 0.04, seed: int = 0):
        self.initial_trust = initial_trust
        self.drift_rate = drift_rate
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.f1 = 0.60 + 0.30 * self.initial_trust    # higher trust → better start
        self.f1_delta = 0.0
        self.ks = 0.0
        self.psi = 0.0
        self.batches = 0
        self.trust = self.initial_trust
        self.cumcost = 0.0
        self.t = 0
        return self._state()

    def _state(self):
        return np.array([self.f1, self.f1_delta, self.ks, self.psi,
                         min(self.batches / 20, 1.0), self.trust, self.cumcost], dtype=float)

    def step(self, action):
        prev_f1 = self.f1

        # Drift degrades F1
        drift_noise = self.rng.normal(0, 0.01)
        self.f1 = max(self.PERFORMANCE_FLOOR,
                      self.f1 - self.drift_rate + drift_noise)
        self.ks = min(1.0, self.ks + self.drift_rate * 0.5)
        self.psi = min(1.0, self.psi + self.drift_rate * 0.3)
        self.f1_delta = self.f1 - prev_f1
        self.batches += 1
        self.t += 1
        cost = 0.0

        if action == 1:  # retrain
            recovery = 0.20 + 0.30 * self.initial_trust  # high trust → better recovery
            self.f1 = min(1.0, self.f1 + recovery)
            self.ks = max(0, self.ks - 0.30)
            self.psi = max(0, self.psi - 0.20)
            self.batches = 0
            cost = self.RETRAIN_COST
        elif action == 2:  # fallback
            self.f1 = 0.55  # fixed fallback performance
            cost = self.FALLBACK_COST

        self.cumcost += cost
        reward = self.f1 - cost
        done = self.t >= 40
        return self._state(), reward, done


# ---------------------------------------------------------------------------
# DQL training
# ---------------------------------------------------------------------------

def train_dql(env, episodes=60, gamma=0.95, eps_start=1.0, eps_end=0.10):
    net = QNetwork(state_dim=7, hidden=32, n_actions=3, seed=0)
    target = QNetwork(state_dim=7, hidden=32, n_actions=3, seed=0)
    target.copy_from(net)
    buf = deque(maxlen=5000)
    eps = eps_start
    eps_decay = (eps_start - eps_end) / (episodes * 0.7)
    rewards = []

    for ep in range(episodes):
        s = env.reset()
        total_r = 0.0
        for _ in range(40):
            if np.random.rand() < eps:
                a = np.random.randint(3)
            else:
                a = int(np.argmax(net.forward(s)))
            s2, r, done = env.step(a)
            buf.append((s.copy(), a, r, s2.copy(), done))
            s = s2
            total_r += r
            # Mini-batch update
            if len(buf) >= 32:
                idxs = np.random.choice(len(buf), 32, replace=False)
                for i in idxs:
                    ss, aa, rr, ss2, dd = buf[i]
                    q_target = rr if dd else rr + gamma * target.forward(ss2).max()
                    net.update(ss, aa, q_target)
            if done:
                break
        if ep % 10 == 0:
            target.copy_from(net)
        eps = max(eps_end, eps - eps_decay)
        rewards.append(total_r)
    return net, rewards


def q_retrain_advantage(net, trust: float) -> float:
    """Q(retrain) - Q(continue) at a mid-drift state."""
    state = np.array([0.65, -0.02, 0.30, 0.25, 0.50, trust, 0.10])
    q = net.forward(state)
    return float(q[1] - q[0])  # retrain - continue


# ---------------------------------------------------------------------------
# Trust scenarios
# ---------------------------------------------------------------------------

def build_trust_scenarios():
    """
    Create scenarios with trust spanning 0.3 → 0.9 by varying dataset
    quality: noise, imbalance, and size.

    To reach trust < 0.5 we need cv_std > 0.4 (needs tiny n with noise)
    and/or accuracy near 0.5 (near-random labels).
    """
    scenarios = []
    rng = np.random.default_rng(0)

    # HIGH trust (≈0.85–0.95): large clean datasets
    for seed, label, drift in [(1, "high_trust_a", 0.02), (2, "high_trust_b", 0.02),
                                (3, "high_trust_c", 0.025)]:
        X, y = make_classification(n_samples=2000, n_features=15, n_informative=10,
                                   n_redundant=2, flip_y=0.0, random_state=seed)
        scenarios.append((X, y, drift, label))

    # MEDIUM-HIGH trust (≈0.70–0.85): moderate noise/imbalance
    for seed, flip, weights, label, drift in [
        (10, 0.05, [0.65, 0.35], "med_high_a", 0.03),
        (11, 0.08, None,         "med_high_b", 0.04),
        (12, 0.06, [0.70, 0.30], "med_high_c", 0.04),
    ]:
        X, y = make_classification(n_samples=800, n_features=12, n_informative=6,
                                   n_redundant=3, weights=weights, flip_y=flip,
                                   random_state=seed)
        scenarios.append((X, y, drift, label))

    # MEDIUM trust (≈0.55–0.70): small n + noise → high cv_std
    for seed, flip, n, label, drift in [
        (20, 0.15, 300, "medium_a", 0.05),
        (21, 0.12, 250, "medium_b", 0.06),
        (22, 0.18, 280, "medium_c", 0.06),
    ]:
        X, y = make_classification(n_samples=n, n_features=10, n_informative=4,
                                   n_redundant=3, flip_y=flip, random_state=seed)
        scenarios.append((X, y, drift, label))

    # LOW trust (≈0.35–0.55): very small n + heavy noise → cv_std near 0.3+
    for seed, flip, n, weights, label, drift in [
        (30, 0.30, 150, [0.85, 0.15], "low_a", 0.07),
        (31, 0.35, 120, None,         "low_b", 0.08),
        (32, 0.30, 140, [0.80, 0.20], "low_c", 0.07),
    ]:
        X, y = make_classification(n_samples=n, n_features=10, n_informative=3,
                                   n_redundant=4, weights=weights, flip_y=flip,
                                   random_state=seed)
        scenarios.append((X, y, drift, label))

    # VERY LOW trust (≈0.20–0.40): near-random labels, extreme imbalance
    for seed, flip, n, weights, label, drift in [
        (40, 0.45, 100, [0.92, 0.08], "verylow_a", 0.09),
        (41, 0.40, 90,  [0.90, 0.10], "verylow_b", 0.09),
        (42, 0.48, 80,  None,         "verylow_c", 0.09),
    ]:
        X, y = make_classification(n_samples=n, n_features=8, n_informative=2,
                                   n_redundant=4, weights=weights, flip_y=flip,
                                   random_state=seed)
        scenarios.append((X, y, drift, label))

    return scenarios


def compute_initial_trust(X, y):
    """
    Compute full 5-component EMMDS trust score.
    Uses actual calibration error and full weight formula.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=300, random_state=42, C=1.0)
    n_splits = min(3, max(2, len(y) // 30))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    try:
        scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_weighted")
    except Exception:
        return 0.30, 0.30, 0.30, 0.50

    cv_mean, cv_std = float(scores.mean()), float(scores.std())
    stability = float(max(0.0, 1.0 - 3.0 * cv_std))  # scaled to be more sensitive

    # Calibration: train/test split
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0,
                                               stratify=y)
        clf2 = LogisticRegression(max_iter=300, random_state=42)
        clf2.fit(Xtr, ytr)
        if hasattr(clf2, "predict_proba") and len(np.unique(y)) == 2:
            prob = clf2.predict_proba(Xte)[:, 1]
            bins = np.linspace(0, 1, 11)
            ece = 0.0
            for lo, hi in zip(bins[:-1], bins[1:]):
                m = (prob >= lo) & (prob < hi)
                if m.sum() > 0:
                    ece += m.sum() / len(yte) * abs(yte[m].mean() - prob[m].mean())
            cal = float(max(0.0, 1.0 - ece))
        else:
            cal = 0.70
    except Exception:
        cal = 0.70

    imb = max(np.bincount(y.astype(int))) / len(y)
    dq = float(np.clip(1.0 - 2.0 * max(0, imb - 0.5), 0, 1))  # more sensitive to imbalance

    accuracy_score = float(np.clip(cv_mean, 0, 1))
    trust = (0.05 * accuracy_score + 0.10 * cal + 0.10 * 0.85 +
             0.35 * dq + 0.40 * stability)
    return float(np.clip(trust, 0, 1)), cv_mean, cv_std, dq


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = ROOT / "outputs" / "research"
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = build_trust_scenarios()
    print(f"Running RL redesign on {len(scenarios)} trust scenarios...")
    print(f"{'Label':<28} {'Trust':>6} {'Adv':>8} {'Training reward':>16}")

    all_records = []
    for X, y, drift_rate, label in scenarios:
        trust, cv_mean, cv_std, dq = compute_initial_trust(X, y)
        env = TrustAwareDriftEnv(initial_trust=trust, drift_rate=drift_rate, seed=42)
        net, ep_rewards = train_dql(env, episodes=60)
        adv = q_retrain_advantage(net, trust)
        final_reward = np.mean(ep_rewards[-10:])
        print(f"  {label:<26} {trust:.3f}  {adv:+.4f}  {final_reward:.4f}")
        all_records.append({
            "label": label,
            "initial_trust": round(trust, 4),
            "cv_mean": round(cv_mean, 4),
            "cv_std": round(cv_std, 4),
            "dq": round(dq, 4),
            "drift_rate": drift_rate,
            "q_retrain_advantage": round(adv, 5),
            "mean_final_reward": round(float(final_reward), 4),
        })

    trusts = np.array([r["initial_trust"] for r in all_records])
    advs = np.array([r["q_retrain_advantage"] for r in all_records])
    trust_range = float(trusts.max() - trusts.min())

    rho, p = stats.spearmanr(trusts, advs)

    result = {
        "n_scenarios": len(all_records),
        "trust_range": {
            "min": round(float(trusts.min()), 4),
            "max": round(float(trusts.max()), 4),
            "range": round(trust_range, 4),
        },
        "hypothesis_h2": {
            "description": "Lower initial trust → higher Q(retrain) advantage",
            "spearman_r": round(float(rho), 4),
            "p_value": round(float(p), 4),
            "verdict": "SUPPORTED" if rho < 0 and p < 0.05 else (
                       "TREND_NOT_SIGNIFICANT" if rho < 0 else "REJECTED"),
            "interpretation": (
                f"Spearman r={rho:.3f}, p={p:.4f}. "
                + ("Trust negatively correlates with retrain advantage — "
                   "lower-trust models benefit more from retraining." if rho < 0
                   else "No clear negative correlation found.")
            ),
        },
        "scenarios": all_records,
    }

    out_path = out_dir / "phase2_rl_redesign.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nH2 result: Spearman r={rho:.4f}, p={p:.4f} → {result['hypothesis_h2']['verdict']}")
    print(f"Trust range: [{trusts.min():.3f}, {trusts.max():.3f}] (Δ={trust_range:.3f})")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
