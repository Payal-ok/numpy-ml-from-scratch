import numpy as np
import pytest

from mlscratch import PCA, KMeans, LinearRegression, LinearSVM
from mlscratch.utils import (
    accuracy,
    make_blobs,
    make_regression_data,
    make_two_class,
    r2_score,
    standardize,
    train_test_split,
)


# ---------- Linear regression ----------
def test_normal_equation_recovers_true_weights():
    X, y, true_w, true_b = make_regression_data(n_samples=500, noise=0.01, seed=1)
    model = LinearRegression(method="normal").fit(X, y)
    assert np.allclose(model.coef_, true_w, atol=0.05)
    assert abs(model.intercept_ - true_b) < 0.05


def test_gradient_descent_matches_normal_equation():
    X, y, _, _ = make_regression_data(n_samples=300, noise=0.3, seed=2)
    exact = LinearRegression(method="normal").fit(X, y)
    gd = LinearRegression(method="gd", lr=0.1, n_iters=3000).fit(X, y)
    assert np.allclose(exact.coef_, gd.coef_, atol=1e-3)
    assert abs(exact.intercept_ - gd.intercept_) < 1e-3


def test_gd_loss_decreases():
    X, y, _, _ = make_regression_data(seed=3)
    model = LinearRegression(method="gd", lr=0.05, n_iters=200).fit(X, y)
    assert model.loss_history_[-1] < model.loss_history_[0]


def test_ridge_shrinks_weights():
    X, y, _, _ = make_regression_data(n_samples=100, seed=4)
    plain = LinearRegression(alpha=0.0).fit(X, y)
    ridge = LinearRegression(alpha=50.0).fit(X, y)
    assert np.linalg.norm(ridge.coef_) < np.linalg.norm(plain.coef_)


def test_regression_generalises():
    X, y, _, _ = make_regression_data(n_samples=400, noise=0.5, seed=5)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y)
    model = LinearRegression().fit(X_tr, y_tr)
    assert r2_score(y_te, model.predict(X_te)) > 0.95


def test_regression_input_validation():
    with pytest.raises(ValueError):
        LinearRegression(method="bad")
    with pytest.raises(ValueError):
        LinearRegression().fit(np.zeros(5), np.zeros(5))
    with pytest.raises(RuntimeError):
        LinearRegression().predict(np.zeros((2, 2)))


# ---------- k-Means ----------
def test_kmeans_finds_well_separated_clusters():
    X, true_labels = make_blobs(n_samples=300, centers=3, spread=0.5, seed=0)
    model = KMeans(n_clusters=3, seed=0).fit(X)
    # Each true cluster should map onto exactly one predicted cluster.
    for k in range(3):
        assert len(set(model.labels_[true_labels == k])) == 1
    assert len(set(model.labels_)) == 3


def test_kmeans_predict_matches_labels():
    X, _ = make_blobs(seed=1)
    model = KMeans(n_clusters=3, seed=1).fit(X)
    assert np.array_equal(model.predict(X), model.labels_)


def test_kmeans_inertia_drops_with_more_clusters():
    X, _ = make_blobs(seed=2)
    inertias = [KMeans(n_clusters=k, seed=0).fit(X).inertia_ for k in (1, 2, 3, 4)]
    assert all(a > b for a, b in zip(inertias, inertias[1:]))


def test_kmeans_validation():
    with pytest.raises(ValueError):
        KMeans(n_clusters=0)
    with pytest.raises(ValueError):
        KMeans(n_clusters=5).fit(np.zeros((2, 2)))


# ---------- PCA ----------
def test_pca_variance_ratios_sum_to_one_with_all_components():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4)) @ rng.normal(size=(4, 4))
    pca = PCA(n_components=4).fit(X)
    assert np.isclose(pca.explained_variance_ratio_.sum(), 1.0)


def test_pca_components_are_orthonormal():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 5))
    pca = PCA(n_components=3).fit(X)
    gram = pca.components_ @ pca.components_.T
    assert np.allclose(gram, np.eye(3), atol=1e-8)


def test_pca_perfect_reconstruction_with_all_components():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 3))
    pca = PCA(n_components=3).fit(X)
    assert np.allclose(pca.inverse_transform(pca.transform(X)), X)


def test_pca_finds_dominant_direction():
    rng = np.random.default_rng(3)
    t = rng.normal(size=500)
    X = np.column_stack([t, 2 * t]) + rng.normal(scale=0.01, size=(500, 2))
    pca = PCA(n_components=1).fit(X)
    direction = pca.components_[0]
    expected = np.array([1, 2]) / np.sqrt(5)
    assert abs(abs(direction @ expected) - 1) < 1e-3
    assert pca.explained_variance_ratio_[0] > 0.99


def test_pca_validation():
    with pytest.raises(ValueError):
        PCA(n_components=10).fit(np.zeros((5, 3)))


# ---------- SVM ----------
def test_svm_separates_two_classes():
    X, y = make_two_class(n_samples=400, separation=2.5, seed=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y)
    X_tr, m, s = standardize(X_tr)
    X_te, _, _ = standardize(X_te, m, s)
    model = LinearSVM(seed=0).fit(X_tr, y_tr)
    assert accuracy(y_te, model.predict(X_te)) > 0.95


def test_svm_loss_decreases():
    X, y = make_two_class(seed=1)
    model = LinearSVM(n_epochs=100, seed=1).fit(X, y)
    assert model.loss_history_[-1] < model.loss_history_[0]


def test_svm_rejects_bad_labels():
    with pytest.raises(ValueError):
        LinearSVM().fit(np.zeros((4, 2)), np.array([0, 1, 0, 1]))


def test_svm_predict_only_returns_valid_labels():
    X, y = make_two_class(seed=2)
    preds = LinearSVM(seed=2).fit(X, y).predict(X)
    assert set(np.unique(preds)).issubset({-1.0, 1.0})
