"""
src/knn.py
Modul K-Nearest Neighbors (KNN) from Scratch (Anggota 1).
Menginduk pada BaseModel dan mendukung metrik jarak Euclidean dan Manhattan.
"""

import numpy as np
from .base import BaseModel
from collections import Counter


class KNN(BaseModel):
    """
    Implementasi K-Nearest Neighbors (KNN) Classifier from Scratch.

    Parameter:
        k (int): Jumlah tetangga terdekat (default=5).
        metric (str): Metrik jarak ('euclidean' atau 'manhattan').
    """

    def __init__(self, k: int = 5, metric: str = "euclidean", **kwargs):
        super().__init__(**kwargs)
        if k <= 0:
            raise ValueError("k harus lebih besar dari 0.")
        if metric.lower() not in ("euclidean", "manhattan"):
            raise ValueError("Metric harus 'euclidean' atau 'manhattan'.")
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
   
        distances = np.zeros((len(X), len(self.X_train_)))    #inisiasi matriks hasil 

        for i in range(len(X)):
            for j in range(len(self.X_train_)):

                if self.metric == "euclidean":
                    distance = np.sqrt(np.sum((X[i]-self.X_train_[j])**2))
                elif self.metric == "manhattan":
                    distance = np.sum(np.abs(X[i]-self.X_train_[j]))
                else:
                    raise ValueError("Metric harus 'euclidean' atau 'manhattan'.")
                distances[i, j] = distance

        return distances

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Menyimpan data latih X dan label y.
        """
        if len(X) == 0:
            raise ValueError("Data training tidak boleh kosong.")
        if len(X) != len(y):
            raise ValueError("Jumlah sampel X dan y harus sama.")
        if self.k > len(X):
            raise ValueError("k tidak boleh melebihi jumlah data training.")
        
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

        if not self.is_fitted:
            raise ValueError("Model belum dilatih.")
        X = np.asarray(X, dtype=float)

        distances = self._compute_distances(X)
        probabilities = []

        # Tentukan label untuk setiap data test
        for i in range(len(X)):
            sorted_distance_indices = np.argsort(distances[i])

            top_k_indices = sorted_distance_indices[:self.k]

            top_k_labels = self.y_train_[top_k_indices]

            # Hitung probabilitas
            prob_class_1 = np.sum(top_k_labels == 1) / self.k
            prob_class_0 = 1 - prob_class_1

            probabilities.append([prob_class_0, prob_class_1])

        return np.array(probabilities)


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Melakukan prediksi label kelas diskrit (0 atau 1).
        """
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

