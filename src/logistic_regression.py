"""
src/logistic_regression.py
Modul Logistic Regression from Scratch & Loss Trajectory Video Generator (Anggota 3).
Mendukung 3 jenis regularisasi: L1 (Lasso), L2 (Ridge), dan Elastic Net,
serta Bonus Optimizer Adam (+2 poin) dan Bonus Video Lintasan Loss (+2 poin).

Referensi Ilmiah:
- Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. arXiv:1412.6980.
"""

import numpy as np
from .base import BaseModel


class LogisticRegression(BaseModel):
    """
    Implementasi Logistic Regression from Scratch.

    Parameter:
        lr (float): Learning rate (eta).
        n_iters (int): Jumlah iterasi / epoch pelatihan.
        penalty (str): Jenis regularisasi ('none', 'l1', 'l2', 'elasticnet').
        reg_lambda (float): Kekuatan regularisasi (lambda). Bias (theta_0) tidak dikenai penalti.
        l1_ratio (float): Rasio alpha in [0, 1] khusus untuk Elastic Net.
        optimizer (str): 'gd' (Gradient Descent standar) atau 'adam' (Bonus Optimasi +2 poin).
        beta1 (float): Parameter peluruhan momen pertama Adam (default=0.9).
        beta2 (float): Parameter peluruhan momen kedua Adam (default=0.999).
        eps (float): Konstanta stabilitas numerik Adam (default=1e-8).
        record_trajectory (bool): Simpan riwayat (theta_0, theta_1, loss) tiap epoch untuk Bonus Video.
    """

    def __init__(
        self,
        lr: float = 0.01,
        n_iters: int = 1000,
        penalty: str = "l2",
        reg_lambda: float = 0.01,
        l1_ratio: float = 0.5,
        optimizer: str = "adam",
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        record_trajectory: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lr = lr
        self.n_iters = n_iters
        self.penalty = penalty.lower()
        self.reg_lambda = reg_lambda
        self.l1_ratio = l1_ratio
        self.optimizer = optimizer.lower()
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.record_trajectory = record_trajectory

        self.bias_ = 0.0  # theta_0 (tidak diregularisasi)
        self.weights_ = None  # theta_1 ... theta_d
        self.loss_history_ = []
        self.trajectory_ = []  # List of tuples (theta_0, theta_1, loss)

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Fungsi aktivasi Sigmoid dengan clipping untuk mencegah overflow."""
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Menghitung Binary Cross-Entropy (Log-Loss) beserta penalti regularisasi (L1 / L2 / Elastic Net).
        Catatan: Bias (self.bias_) tidak dikenai penalti regularisasi.
        """
        # TODO (Anggota 3): Implementasikan perhitungan fungsi objektif J(theta)
        raise NotImplementedError("LogisticRegression.compute_loss belum diimplementasikan oleh Anggota 3.")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Melatih parameter (bias_ dan weights_) menggunakan Gradient Descent atau Adam Optimizer.
        """
        # TODO (Anggota 3): Implementasikan pelatihan dengan regularisasi L1/L2/ElasticNet & Adam
        raise NotImplementedError("LogisticRegression.fit belum diimplementasikan oleh Anggota 3.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Menghasilkan probabilitas posterior [P(y=0|X), P(y=1|X)] berukuran (n_samples, 2).
        """
        # TODO (Anggota 3): Implementasikan predict_proba menggunakan _sigmoid
        raise NotImplementedError("LogisticRegression.predict_proba belum diimplementasikan oleh Anggota 3.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Melakukan prediksi label kelas diskrit (0 atau 1).
        """
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)


def generate_loss_contour_video(
    model: LogisticRegression,
    X_2d: np.ndarray,
    y: np.ndarray,
    output_path: str = "output/videos/logreg_loss_contour.mp4",
    fps: int = 15,
):
    """
    Bonus Video Lintasan Loss & Kontur (theta_0, theta_1) (+2 Poin):
    Merender animasi lintasan konvergensi parameter menuju minimum global di atas grid kontur Log-Loss.
    """
    # TODO (Anggota 3): Implementasikan generator video menggunakan matplotlib.animation / imageio
    raise NotImplementedError("generate_loss_contour_video belum diimplementasikan oleh Anggota 3.")

