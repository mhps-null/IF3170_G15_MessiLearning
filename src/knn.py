"""
src/knn.py
Modul K-Nearest Neighbors (KNN) from Scratch (Anggota 1).
Menginduk pada BaseModel dan mendukung metrik jarak Euclidean dan Manhattan.
"""

import numpy as np
from .base import BaseModel


class KNN(BaseModel):
    """
    Implementasi K-Nearest Neighbors (KNN) Classifier from Scratch.

    Parameter:
        k (int): Jumlah tetangga terdekat (default=5).
        metric (str): Metrik jarak ('euclidean' atau 'manhattan').
    """

    def __init__(self, k: int = 5, metric: str = "euclidean", **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.metric = metric.lower()
        self.X_train_ = None
        self.y_train_ = None

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Menghitung matriks jarak antara setiap sampel di X dengan seluruh sampel di X_train_.
        - Euclidean: d(x, x') = sqrt(sum((x_i - x'_i)^2))
        - Manhattan: d(x, x') = sum(|x_i - x'_i|)
        """
        # TODO (Anggota 1): Implementasikan perhitungan jarak vektorisasi NumPy
        raise NotImplementedError("KNN._compute_distances belum diimplementasikan oleh Anggota 1.")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Menyimpan data latih X dan label y.
        """
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y, dtype=int)
        self.classes_ = np.unique(self.y_train_)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Menghasilkan probabilitas posterior P(y=0|X) dan P(y=1|X) dari rasio K tetangga terdekat:
        P(y=1|X) = (1 / K) * sum_{k in N_K(X)} I(y_k == 1)
        Mengembalikan np.ndarray berukuran (n_samples, 2).
        """
        # TODO (Anggota 1): Implementasikan predict_proba KNN
        raise NotImplementedError("KNN.predict_proba belum diimplementasikan oleh Anggota 1.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Melakukan prediksi label kelas diskrit (0 atau 1).
        """
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

