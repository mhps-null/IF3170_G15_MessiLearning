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
        Menghitung matriks jarak antara setiap sampel di X dengan seluruh sampel di X_train_ menggunakan NumPy broadcasting.
        - Euclidean: d(x, x') = sqrt(sum((x_i - x'_i)^2))
        - Manhattan: d(x, x') = sum(|x_i - x'_i|)
        Mengembalikan np.ndarray: Matriks jarak berukuran (n_samples, n_train).
        """
        
        differences = X[:, np.newaxis, :] - self.X_train_[np.newaxis, :, :]   

        if self.metric == "euclidean":
            distances = np.sqrt(np.sum(differences**2, axis=2))

        else:  #self.metric == "manhattan"
            distances = np.sum(np.abs(differences), axis=2)

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

        # Untuk setiap data uji: urut indeks data training, ambil indeks dan label k tetangga terdekat, hitung probabilitas
        sorted_distance_indices = np.argsort(distances, axis=1)

        top_k_indices = sorted_distance_indices[:, :self.k]

        top_k_labels = self.y_train_[top_k_indices]

        prob_class_1 = np.mean(top_k_labels == 1, axis=1) 
        prob_class_0 = 1 - prob_class_1

        return np.column_stack((prob_class_0, prob_class_1))


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Melakukan prediksi label kelas diskrit (0 atau 1).
        """
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

