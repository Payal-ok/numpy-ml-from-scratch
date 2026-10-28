"""Run every model on synthetic data, print metrics and save plots to images/."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # works without a display
import matplotlib.pyplot as plt
import numpy as np

from mlscratch import PCA, KMeans, LinearRegression, LinearSVM
from mlscratch.utils import (
    accuracy,
    make_blobs,
    make_regression_data,
    make_two_class,
    mse,
    r2_score,
    standardize,
    train_test_split,
)

OUT = Path(__file__).parent / "images"
OUT.mkdir(exist_ok=True)


def demo_linear_regression():
    print("\n=== Linear Regression ===")
    X, y, true_w, true_b = make_regression_data(n_samples=300, n_features=3, noise=0.5, seed=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y)

    exact = LinearRegression(method="normal").fit(X_tr, y_tr)
    gd = LinearRegression(method="gd", lr=0.1, n_iters=500).fit(X_tr, y_tr)

    print("true weights      :", np.round(true_w, 3), "bias", round(true_b, 3))
    print("normal equation   :", np.round(exact.coef_, 3), "bias", round(exact.intercept_, 3))
    print("gradient descent  :", np.round(gd.coef_, 3), "bias", round(gd.intercept_, 3))
    print(f"test MSE = {mse(y_te, exact.predict(X_te)):.4f}   test R^2 = {r2_score(y_te, exact.predict(X_te)):.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(gd.loss_history_)
    ax[0].set(title="Gradient descent loss", xlabel="iteration", ylabel="MSE", yscale="log")
    ax[1].scatter(y_te, exact.predict(X_te), s=12)
    lims = [y_te.min(), y_te.max()]
    ax[1].plot(lims, lims, "r--", label="perfect")
    ax[1].set(title="Predicted vs actual (test)", xlabel="actual", ylabel="predicted")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "linear_regression.png", dpi=120)
    plt.close(fig)


def demo_kmeans():
    print("\n=== k-Means ===")
    X, true_labels = make_blobs(n_samples=400, centers=4, spread=0.9, seed=1)
    ks = range(1, 8)
    inertias = [KMeans(n_clusters=k, seed=0).fit(X).inertia_ for k in ks]
    model = KMeans(n_clusters=4, seed=0).fit(X)
    print(f"converged in {model.n_iter_} iterations, inertia = {model.inertia_:.2f}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(list(ks), inertias, "o-")
    ax[0].set(title="Elbow method", xlabel="k", ylabel="inertia")
    ax[1].scatter(X[:, 0], X[:, 1], c=model.labels_, s=12, cmap="viridis")
    ax[1].scatter(*model.centroids_.T, c="red", marker="X", s=150, label="centroids")
    ax[1].set(title="k-Means clusters (k=4)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "kmeans.png", dpi=120)
    plt.close(fig)


def demo_pca():
    print("\n=== PCA ===")
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(300, 2)) * [5.0, 1.5]
    mixing = rng.normal(size=(2, 6))
    X = latent @ mixing + rng.normal(scale=0.3, size=(300, 6))
    pca = PCA(n_components=6).fit(X)
    ratios = pca.explained_variance_ratio_
    print("explained variance ratio:", np.round(ratios, 4))
    print(f"first 2 components keep {ratios[:2].sum():.1%} of the variance")

    Z = PCA(n_components=2).fit_transform(X)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(range(1, 7), ratios)
    ax[0].set(title="Explained variance ratio", xlabel="component", ylabel="ratio")
    ax[1].scatter(Z[:, 0], Z[:, 1], s=12)
    ax[1].set(title="Data projected to 2-D", xlabel="PC1", ylabel="PC2")
    fig.tight_layout()
    fig.savefig(OUT / "pca.png", dpi=120)
    plt.close(fig)


def demo_svm():
    print("\n=== Linear SVM ===")
    X, y = make_two_class(n_samples=400, separation=1.6, seed=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y)
    X_tr, mean, std = standardize(X_tr)
    X_te, _, _ = standardize(X_te, mean, std)
    model = LinearSVM(lr=0.1, lambda_=0.01, n_epochs=200, seed=0).fit(X_tr, y_tr)
    print(f"train accuracy = {accuracy(y_tr, model.predict(X_tr)):.3f}")
    print(f"test  accuracy = {accuracy(y_te, model.predict(X_te)):.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(model.loss_history_)
    ax[0].set(title="SVM training loss", xlabel="epoch", ylabel="loss")
    xs = np.linspace(X_te[:, 0].min() - 0.5, X_te[:, 0].max() + 0.5, 200)
    w, b = model.w_, model.b_
    ax[1].scatter(X_te[:, 0], X_te[:, 1], c=y_te, cmap="coolwarm", s=14)
    ax[1].plot(xs, -(w[0] * xs + b) / w[1], "k-", label="boundary")
    ax[1].plot(xs, -(w[0] * xs + b - 1) / w[1], "k--", linewidth=0.8)
    ax[1].plot(xs, -(w[0] * xs + b + 1) / w[1], "k--", linewidth=0.8)
    ax[1].set(title="Decision boundary (test set)", ylim=(X_te[:, 1].min() - 0.5, X_te[:, 1].max() + 0.5))
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "svm.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    demo_linear_regression()
    demo_kmeans()
    demo_pca()
    demo_svm()
    print(f"\nPlots saved to {OUT}/")
