"""Linear regression with two solvers: the normal equation and gradient descent."""

from __future__ import annotations

import numpy as np


class LinearRegression:
    """Ordinary least squares with optional L2 (ridge) regularisation.

    Parameters
    ----------
    method : "normal" or "gd"
        "normal" solves the closed-form normal equation.
        "gd" uses batch gradient descent.
    lr : float
        Learning rate for gradient descent.
    n_iters : int
        Number of gradient descent steps.
    alpha : float
        Ridge penalty strength (0 means plain least squares).
    """

    def __init__(self, method="normal", lr=0.05, n_iters=2000, alpha=0.0):
        if method not in ("normal", "gd"):
            raise ValueError("method must be 'normal' or 'gd'")
        self.method = method
        self.lr = lr
        self.n_iters = n_iters
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = 0.0
        self.loss_history_ = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D (n_samples, n_features)")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")

        n, d = X.shape
        Xb = np.hstack([np.ones((n, 1)), X])  # add bias column

        if self.method == "normal":
            penalty = self.alpha * np.eye(d + 1)
            penalty[0, 0] = 0.0  # do not penalise the intercept
            w = np.linalg.solve(Xb.T @ Xb + penalty, Xb.T @ y)
            self.loss_history_ = [float(np.mean((Xb @ w - y) ** 2))]
        else:
            w = np.zeros(d + 1)
            self.loss_history_ = []
            for _ in range(self.n_iters):
                error = Xb @ w - y
                grad = (2 / n) * (Xb.T @ error)
                reg = 2 * self.alpha * w / n
                reg[0] = 0.0
                w -= self.lr * (grad + reg)
                self.loss_history_.append(float(np.mean(error**2)))

        self.intercept_ = float(w[0])
        self.coef_ = w[1:]
        return self

    def predict(self, X):
        if self.coef_ is None:
            raise RuntimeError("Call fit() before predict()")
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_
