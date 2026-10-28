"""Principal Component Analysis using the singular value decomposition."""

from __future__ import annotations

import numpy as np


class PCA:
    """Project data onto the directions of greatest variance.

    Parameters
    ----------
    n_components : int
        Number of principal components to keep.
    """

    def __init__(self, n_components=2):
        if n_components < 1:
            raise ValueError("n_components must be at least 1")
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        if self.n_components > min(n, d):
            raise ValueError("n_components cannot exceed min(n_samples, n_features)")

        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)

        
        signs = np.sign(Vt[np.arange(len(Vt)), np.abs(Vt).argmax(axis=1)])
        signs[signs == 0] = 1
        Vt = Vt * signs[:, None]

        variance = (S**2) / (n - 1)
        self.components_ = Vt[: self.n_components]
        self.explained_variance_ = variance[: self.n_components]
        self.explained_variance_ratio_ = variance[: self.n_components] / variance.sum()
        return self

    def transform(self, X):
        if self.components_ is None:
            raise RuntimeError("Call fit() before transform()")
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        if self.components_ is None:
            raise RuntimeError("Call fit() before inverse_transform()")
        return np.asarray(Z, dtype=float) @ self.components_ + self.mean_
