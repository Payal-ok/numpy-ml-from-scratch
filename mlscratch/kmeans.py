"""k-Means clustering with k-means++ initialisation."""

from __future__ import annotations

import numpy as np


class KMeans:
    """Lloyd's algorithm with k-means++ seeding.

    Parameters
    ----------
    n_clusters : int
    max_iter : int
    tol : float
        Stop when centroids move less than this amount.
    seed : int
    """

    def __init__(self, n_clusters=3, max_iter=300, tol=1e-6, seed=0):
        if n_clusters < 1:
            raise ValueError("n_clusters must be at least 1")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

    def _init_centroids(self, X, rng):
        n = len(X)
        centroids = [X[rng.integers(n)]]
        for _ in range(1, self.n_clusters):
            d2 = np.min(
                ((X[:, None, :] - np.array(centroids)[None, :, :]) ** 2).sum(axis=2),
                axis=1,
            )
            total = d2.sum()
            if total == 0:  
                centroids.append(X[rng.integers(n)])
                continue
            centroids.append(X[rng.choice(n, p=d2 / total)])
        return np.array(centroids)

    @staticmethod
    def _assign(X, centroids):
        d2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        return d2.argmin(axis=1), d2

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if len(X) < self.n_clusters:
            raise ValueError("Need at least n_clusters samples")
        rng = np.random.default_rng(self.seed)
        centroids = self._init_centroids(X, rng)

        for i in range(1, self.max_iter + 1):
            labels, _ = self._assign(X, centroids)
            new_centroids = centroids.copy()
            for k in range(self.n_clusters):
                members = X[labels == k]
                if len(members):
                    new_centroids[k] = members.mean(axis=0)
                else:  
                    new_centroids[k] = X[rng.integers(len(X))]
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            self.n_iter_ = i
            if shift < self.tol:
                break

        self.centroids_ = centroids
        self.labels_, d2 = self._assign(X, centroids)
        self.inertia_ = float(d2[np.arange(len(X)), self.labels_].sum())
        return self

    def predict(self, X):
        if self.centroids_ is None:
            raise RuntimeError("Call fit() before predict()")
        labels, _ = self._assign(np.asarray(X, dtype=float), self.centroids_)
        return labels
