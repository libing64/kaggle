"""Naive 2-hidden-layer MLP (NumPy + mini-batch SGD).

Intentionally plain: ReLU hidden units, sigmoid output, BCE loss,
no dropout / batch-norm / Adam. sklearn-compatible so it can sit
in the same Pipeline + StratifiedKFold loop as LR / RF.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


class NaiveMLP(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (32, 16),
        learning_rate: float = 0.05,
        epochs: int = 200,
        batch_size: int = 32,
        l2: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.random_state = random_state

    def _init_params(self, n_in: int) -> None:
        rng = np.random.default_rng(self.random_state)
        dims = [n_in, *self.hidden_layer_sizes, 1]
        self.weights_ = []
        self.biases_ = []
        for i in range(len(dims) - 1):
            # He for ReLU hidden layers; Xavier-ish for the sigmoid head.
            scale = np.sqrt(2.0 / dims[i]) if i < len(dims) - 2 else np.sqrt(1.0 / dims[i])
            self.weights_.append(rng.normal(0.0, scale, size=(dims[i], dims[i + 1])))
            self.biases_.append(np.zeros((1, dims[i + 1])))

    def _forward(self, x: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activations = [x]
        preacts = []
        a = x
        last = len(self.weights_) - 1
        for i, (w, b) in enumerate(zip(self.weights_, self.biases_)):
            z = a @ w + b
            preacts.append(z)
            a = _sigmoid(z) if i == last else np.maximum(0.0, z)
            activations.append(a)
        return activations, preacts

    def _backward(self, y: np.ndarray, activations: list[np.ndarray], preacts: list[np.ndarray]) -> None:
        # dL/dz for BCE + sigmoid is (p - y).
        delta = activations[-1] - y
        for i in range(len(self.weights_) - 1, -1, -1):
            a_prev = activations[i]
            grad_w = a_prev.T @ delta / len(y) + self.l2 * self.weights_[i]
            grad_b = delta.mean(axis=0, keepdims=True)
            self.weights_[i] -= self.learning_rate * grad_w
            self.biases_[i] -= self.learning_rate * grad_b
            if i == 0:
                break
            delta = (delta @ self.weights_[i].T) * (preacts[i - 1] > 0.0)

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=np.float64)
        self.classes_ = np.array([0, 1])
        y = y.reshape(-1, 1).astype(np.float64)
        self._init_params(X.shape[1])

        rng = np.random.default_rng(self.random_state)
        n = len(X)
        for _ in range(self.epochs):
            order = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                activations, preacts = self._forward(X[idx])
                self._backward(y[idx], activations, preacts)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "weights_")
        X = check_array(X, dtype=np.float64)
        p1 = self._forward(X)[0][-1].ravel()
        p1 = np.clip(p1, 1e-7, 1.0 - 1e-7)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
