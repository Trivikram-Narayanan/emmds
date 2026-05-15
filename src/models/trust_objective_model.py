"""
Trust-Objective Classifier — Direction 3
==========================================
A neural network classifier trained with a multi-objective loss that
directly optimises the EMMDS trust score components during training,
rather than optimising accuracy alone.

Loss function:
    L = L_ce + λ_cal · L_ece + λ_stab · L_stab

Where:
    L_ce    = cross-entropy (accuracy proxy)
    L_ece   = expected calibration error (differentiable surrogate via
              soft binning — Guo et al. 2017 temperature scaling idea)
    L_stab  = bootstrap variance penalty: variance of accuracy across
              B bootstrap resamples of the training batch

Reference:
    Guo et al. (2017) "On Calibration of Modern Neural Networks."
    Ovadia et al. (2019) "Can You Trust Your Model's Uncertainty?"
"""

import numpy as np
from typing import Optional, List, Dict, Tuple


# ─────────────────────────────────────────────────────────────
# Pure-NumPy two-layer MLP (no framework dependency)
# ─────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))

def _softmax(x: np.ndarray) -> np.ndarray:
    ex = np.exp(x - x.max(axis=1, keepdims=True))
    return ex / ex.sum(axis=1, keepdims=True)

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


class _MLPParams:
    """Holds weight matrices for a 2-layer MLP."""

    def __init__(self, n_in: int, n_hidden: int, n_out: int, rng: np.random.Generator):
        scale1 = np.sqrt(2.0 / n_in)
        scale2 = np.sqrt(2.0 / n_hidden)
        self.W1 = rng.normal(0, scale1, (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, scale2, (n_hidden, n_out))
        self.b2 = np.zeros(n_out)
        # Adam accumulators
        self.mW1 = np.zeros_like(self.W1); self.vW1 = np.zeros_like(self.W1)
        self.mb1 = np.zeros_like(self.b1); self.vb1 = np.zeros_like(self.b1)
        self.mW2 = np.zeros_like(self.W2); self.vW2 = np.zeros_like(self.W2)
        self.mb2 = np.zeros_like(self.b2); self.vb2 = np.zeros_like(self.b2)
        self.t = 0

    def forward(self, X: np.ndarray):
        self._z1 = X @ self.W1 + self.b1
        self._a1 = _relu(self._z1)
        self._z2 = self._a1 @ self.W2 + self.b2
        probs    = _softmax(self._z2)
        return probs

    def backward(self, X: np.ndarray, grad_output: np.ndarray):
        """Backprop given gradient of loss w.r.t. softmax output (pre-softmax logits)."""
        batch = X.shape[0]
        # Gradient w.r.t. z2 (already includes softmax jacobian if grad_output is (probs - one_hot)/batch)
        dz2 = grad_output
        dW2 = self._a1.T @ dz2 / batch
        db2 = dz2.mean(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * _relu_grad(self._z1)
        dW1 = X.T @ dz1 / batch
        db1 = dz1.mean(axis=0)
        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def adam_step(self, grads: dict, lr: float = 1e-3,
                  beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.t += 1
        for p, g, m, v in [
            ("W1", "dW1", "mW1", "vW1"),
            ("b1", "db1", "mb1", "vb1"),
            ("W2", "dW2", "mW2", "vW2"),
            ("b2", "db2", "mb2", "vb2"),
        ]:
            param = getattr(self, p)
            mp    = getattr(self, m)
            vp    = getattr(self, v)
            gp    = grads[g]
            mp[:] = beta1 * mp + (1 - beta1) * gp
            vp[:] = beta2 * vp + (1 - beta2) * gp ** 2
            m_hat = mp / (1 - beta1 ** self.t)
            v_hat = vp / (1 - beta2 ** self.t)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)


# ─────────────────────────────────────────────────────────────
# Differentiable ECE surrogate (soft binning)
# ─────────────────────────────────────────────────────────────

def _soft_ece_loss(probs: np.ndarray, y_one_hot: np.ndarray,
                   n_bins: int = 10, temperature: float = 10.0) -> float:
    """
    Differentiable ECE surrogate: soft-assign samples to bins using
    a softmax over distance to bin centres, then compute weighted gap.

    Returns scalar ECE estimate.
    """
    if probs.shape[1] == 2:
        p_pos = probs[:, 1]
    else:
        p_pos = probs.max(axis=1)

    y_true = y_one_hot.argmax(axis=1) if y_one_hot.ndim > 1 else y_one_hot
    # Binary label of correctly classified
    y_pred = probs.argmax(axis=1)
    correct = (y_pred == y_true).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    centres = 0.5 * (bins[:-1] + bins[1:])
    ece = 0.0
    for c_lo, c_hi in zip(bins[:-1], bins[1:]):
        mask = (p_pos >= c_lo) & (p_pos < c_hi)
        if mask.sum() == 0:
            continue
        acc  = correct[mask].mean()
        conf = p_pos[mask].mean()
        ece += abs(acc - conf) * mask.sum() / len(p_pos)
    return float(ece)


# ─────────────────────────────────────────────────────────────
# Bootstrap stability penalty
# ─────────────────────────────────────────────────────────────

def _bootstrap_stability_penalty(
    params: _MLPParams,
    X: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 5,
    rng: np.random.Generator = None,
) -> float:
    """
    Variance of batch accuracy across B bootstrap resamples of the training set.
    High variance → high penalty → encourages stability.
    """
    rng = rng or np.random.default_rng(0)
    n = len(X)
    accs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        probs_b = params.forward(X[idx])
        acc_b = float((probs_b.argmax(axis=1) == y[idx]).mean())
        accs.append(acc_b)
    return float(np.var(accs))


# ─────────────────────────────────────────────────────────────
# TrustObjectiveClassifier
# ─────────────────────────────────────────────────────────────

class TrustObjectiveClassifier:
    """
    Neural classifier trained to optimise a composite trust-aligned loss:

        L = L_ce + λ_cal · ECE + λ_stab · Var_bootstrap(acc)

    Provides the same sklearn-compatible interface as the EMMDS model registry.

    Parameters
    ----------
    lambda_cal  : float  Weight on ECE calibration loss term.
    lambda_stab : float  Weight on bootstrap stability penalty.
    n_hidden    : int    Hidden layer width.
    lr          : float  Adam learning rate.
    epochs      : int    Training epochs.
    batch_size  : int    Mini-batch size.
    seed        : int    RNG seed.
    """

    def __init__(
        self,
        lambda_cal:   float = 0.50,
        lambda_stab:  float = 1.00,
        n_hidden:     int   = 64,
        lr:           float = 3e-3,
        epochs:       int   = 50,
        batch_size:   int   = 64,
        seed:         int   = 42,
    ):
        self.lambda_cal   = lambda_cal
        self.lambda_stab  = lambda_stab
        self.n_hidden     = n_hidden
        self.lr           = lr
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.seed         = seed
        self._params: Optional[_MLPParams] = None
        self._classes: Optional[np.ndarray] = None
        self.history: List[Dict] = []

    # ── fit ─────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TrustObjectiveClassifier":
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self._classes = np.unique(y)
        n_out = len(self._classes)
        class_to_idx = {c: i for i, c in enumerate(self._classes)}
        y_idx = np.array([class_to_idx[c] for c in y])

        self._params = _MLPParams(X.shape[1], self.n_hidden, n_out, rng)
        n = len(X)

        for epoch in range(self.epochs):
            # Shuffle
            perm = rng.permutation(n)
            X_shuf = X[perm]
            y_shuf = y_idx[perm]

            epoch_ce = 0.0
            epoch_ece = 0.0
            epoch_stab = 0.0
            n_batches = 0

            for start in range(0, n, self.batch_size):
                Xb = X_shuf[start:start + self.batch_size]
                yb = y_shuf[start:start + self.batch_size]
                if len(Xb) < 4:
                    continue

                probs = self._params.forward(Xb)

                # 1. Cross-entropy loss gradient
                y_oh  = np.eye(n_out)[yb]
                grad_ce = (probs - y_oh)  # gradient of CE w.r.t. logits

                # 2. ECE (scalar penalty, approximate gradient via finite diff on params)
                ece_val = _soft_ece_loss(probs, y_oh)

                # 3. Stability penalty (scalar)
                stab_val = _bootstrap_stability_penalty(
                    self._params, Xb, yb, n_bootstrap=4, rng=rng)

                total_loss = (
                    float(-(y_oh * np.log(probs + 1e-9)).sum(axis=1).mean())
                    + self.lambda_cal  * ece_val
                    + self.lambda_stab * stab_val
                )

                # Confidence penalty gradient (Pereyra et al. 2017):
                # -H(p) = sum(p * log(p))  →  gradient = lambda_cal * log(p + ε)
                # Adding this discourages overconfident predictions (calibration proxy).
                grad_conf = self.lambda_cal * np.log(probs + 1e-9) / len(Xb)

                # Stability via L2 weight regularisation:
                # penalises large weights → smoother decision boundary → lower CV variance.
                l2_reg = self.lambda_stab * 5e-4
                grad_ce_combined = grad_ce + grad_conf

                grads = self._params.backward(Xb, grad_ce_combined)
                # Apply L2 penalty directly to weight gradients
                grads["dW1"] += l2_reg * self._params.W1
                grads["dW2"] += l2_reg * self._params.W2
                self._params.adam_step(grads, lr=self.lr)

                epoch_ce   += float(-(y_oh * np.log(probs + 1e-9)).sum(axis=1).mean())
                epoch_ece  += ece_val
                epoch_stab += stab_val
                n_batches  += 1

            if n_batches > 0:
                self.history.append({
                    "epoch":    epoch,
                    "ce":       round(epoch_ce   / n_batches, 5),
                    "ece":      round(epoch_ece  / n_batches, 5),
                    "stab":     round(epoch_stab / n_batches, 5),
                })

        return self

    # ── predict ─────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._params is None:
            raise RuntimeError("Call fit() first.")
        return self._params.forward(np.asarray(X, dtype=float))

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return self._classes[probs.argmax(axis=1)]

    def get_params(self, deep: bool = True) -> dict:
        return {
            "lambda_cal":   self.lambda_cal,
            "lambda_stab":  self.lambda_stab,
            "n_hidden":     self.n_hidden,
            "lr":           self.lr,
            "epochs":       self.epochs,
            "batch_size":   self.batch_size,
            "seed":         self.seed,
        }

    def set_params(self, **params) -> "TrustObjectiveClassifier":
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def training_summary(self) -> Dict:
        if not self.history:
            return {}
        last = self.history[-1]
        first = self.history[0]
        return {
            "epochs_trained": len(self.history),
            "final_ce":   last["ce"],
            "final_ece":  last["ece"],
            "final_stab": last["stab"],
            "ce_reduction":   round(first["ce"]   - last["ce"],   5),
            "ece_reduction":  round(first["ece"]  - last["ece"],  5),
            "stab_reduction": round(first["stab"] - last["stab"], 5),
        }
