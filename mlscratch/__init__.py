"""mlscratch: classic machine learning algorithms implemented with NumPy only."""

from .kmeans import KMeans
from .linear_regression import LinearRegression
from .pca import PCA
from .svm import LinearSVM

__all__ = ["LinearRegression", "KMeans", "PCA", "LinearSVM"]
__version__ = "1.0.0"
