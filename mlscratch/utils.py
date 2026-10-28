"""Small helpers shared by the models: data generation, scaling and metrics."""

from __future__ import annotations

import numpy as np


def make_regression_data(n_samples=200, n_features=3, noise=0.5, seed=0):
    """Create a synthetic regression problem with a known true weight vector."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    true_w = rng.uniform(-3, 3, size=n_features)
    true_b = rng.uniform(-2, 2)
    y = X @ true_w + true_b + rng.normal(scale=noise, size=n_samples)
    return X, y, true_w, true_b


def make_blobs(n_samples=300, centers=3, spread=0.8, seed=0):
    """Create 2-D Gaussian clusters. Returns points and the true cluster labels."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, centers, endpoint=False)
    means = 5 * np.column_stack([np.cos(angles), np.sin(angles)])
    labels = rng.integers(0, centers, size=n_samples)
    X = means[labels] + rng.normal(scale=spread, size=(n_samples, 2))
    return X, labels


def make_two_class(n_samples=300, separation=2.5, seed=0):
    """Two roughly separable 2-D classes with labels -1 and +1."""
    rng = np.random.default_rng(seed)
    half = n_samples // 2
    pos = rng.normal(loc=[separation, separation], scale=1.0, size=(half, 2))
    neg = rng.normal(loc=[-separation, -separation], scale=1.0, size=(n_samples - half, 2))
    X = np.vstack([pos, neg])
    y = np.concatenate([np.ones(half), -np.ones(n_samples - half)])
    order = rng.permutation(n_samples)
    return X[order], y[order]


def train_test_split(X, y, test_size=0.25, seed=0):
    """Shuffle and split arrays into train and test parts."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(len(X) * (1 - test_size))
    train, test = idx[:cut], idx[cut:]
    return X[train], X[test], y[train], y[test]


def standardize(X, mean=None, std=None):
    """Scale features to zero mean and unit variance.

    Pass the mean/std from the training set when scaling the test set.
    """
    X = np.asarray(X, dtype=float)
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std, mean, std


def mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    ss_res = np.sum((y_true - np.asarray(y_pred)) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
