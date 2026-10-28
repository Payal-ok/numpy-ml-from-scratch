"""Linear Support Vector Machine trained with sub-gradient descent on the hinge loss."""

from __future__ import annotations

import numpy as np


class LinearSVM:
    """Soft-margin linear SVM.

    Minimises   lambda * ||w||^2  +  mean(max(0, 1 - y * (w.x + b)))

    Parameters
    ----------
    lr : float
        Initial learning rate (decays as 1 / (1 + decay * epoch)).
    lambda_ : float
        Regularisation strength. Larger values give a wider margin.
    n_epochs : int
    batch_size : int
    seed : int
    """

    def __init__(self, lr=0.1, lambda_=0.01, n_epochs=200, batch_size=32, decay=0.01, seed=0):
        self.lr = lr
        self.lambda_ = lambda_
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.decay = decay
        self.seed = seed
        self.w_ = None
        self.b_ = 0.0
        self.loss_history_ = []

    @staticmethod
    def _check_labels(y):
        y = np.asarray(y, dtype=float)
        if not set(np.unique(y)).issubset({-1.0, 1.0}):
            raise ValueError("Labels must be -1 or +1")
        return y

    def _loss(self, X, y):
        margins = y * (X @ self.w_ + self.b_)
        return float(self.lambda_ * (self.w_ @ self.w_) + np.mean(np.maximum(0, 1 - margins)))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = self._check_labels(y)
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        self.w_ = np.zeros(d)
        self.b_ = 0.0
        self.loss_history_ = []

        for epoch in range(self.n_epochs):
            lr = self.lr / (1 + self.decay * epoch)
            order = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                Xb, yb = X[idx], y[idx]
                margins = yb * (Xb @ self.w_ + self.b_)
                violating = margins < 1
                grad_w = 2 * self.lambda_ * self.w_
                grad_b = 0.0
                if violating.any():
                    grad_w = grad_w - (yb[violating, None] * Xb[violating]).sum(axis=0) / len(idx)
                    grad_b = -yb[violating].sum() / len(idx)
                self.w_ -= lr * grad_w
                self.b_ -= lr * grad_b
            self.loss_history_.append(self._loss(X, y))
        return self

    def decision_function(self, X):
        if self.w_ is None:
            raise RuntimeError("Call fit() before predict()")
        return np.asarray(X, dtype=float) @ self.w_ + self.b_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1.0, -1.0)
