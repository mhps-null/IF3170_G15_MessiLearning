"""
src/svm.py
Modul Support Vector Machine (SVM) with Kernels & 2D Margin Plotter (Anggota 4).
Mendukung 3 kernel wajib: Linear, Polynomial, dan Gaussian (RBF),
serta Bonus Visualisasi 2D Bidang Pemisah (+1 poin) dan Bonus Optimasi SMO/Pegasos (+2 poin).

Referensi Ilmiah:
- Platt, J. (1998). Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines.
  Microsoft Research Technical Report MSR-TR-98-14.
- Shalev-Shwartz, S., Singer, Y., Srebro, N., & Cotter, A. (2011). Pegasos: Primal Estimated sub-GrAdient
  SOlver for SVM. Mathematical Programming, 127(1), 3-30.
"""

import numpy as np
import math
from .base import BaseModel


class SVM(BaseModel):
    """
    Implementasi Support Vector Machine (SVM) Classifier from Scratch.

    Parameter:
        C (float): Parameter regularisasi soft-margin.
        kernel (str): Jenis kernel ('linear', 'poly', 'rbf').
        degree (int): Derajat polinomial untuk kernel 'poly' (default=3).
        gamma (float): Koefisien kernel untuk 'poly' dan 'rbf' (jika None, gunakan 1 / n_features).
        coef0 (float): Konstanta independen r pada kernel polinomial (default=1.0).
        solver (str): 'smo' (Sequential Minimal Optimization) atau 'pegasos'.
        max_iter (int): Jumlah iterasi maksimum solver.
        tol (float): Toleransi konvergensi KKT.
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        degree: int = 3,
        gamma: float = None,
        coef0: float = 1.0,
        solver: str = "smo",
        max_iter: int = 1000,
        tol: float = 1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.C = C
        self.kernel = kernel.lower()
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.solver = solver.lower()
        self.max_iter = max_iter
        self.tol = tol

        self.x_ = None
        self.y_ = None

        # Parameter hasil pelatihan
        self.alphas_ = None
        self.b_ = 0.0
        self.support_vectors_ = None
        self.support_vector_labels_ = None
        self.support_vector_indices_ = None

        # Parameter Platt Scaling (A, B) untuk predict_proba
        self.platt_A_ = -1.0
        self.platt_B_ = 0.0

    def _compute_kernel(self, X1: np.ndarray, X2: np.ndarray) :
        if self.kernel == 'linear' :
            return np.dot(X1,X2)
        elif self.kernel == 'poly' :
            return pow(self.gamma*np.dot(X1,X2) + self.coef0, self.degree)
        elif self.kernel == 'rbf' :
            return math.exp(-self.gamma*pow(np.linalg.norm(X1 - X2), 2))
        else :
            raise ValueError(self.kernel + " is not a valid kernel option!")

    def _decision_function(self, X1: np.ndarray) :
        """
        Menghitung skor keputusan f(x) = sum_{i in SV} alpha_i * y_i * K(x_i, x) + b.
        """
        sum = 0
        for i in range(len(self.x_)) :
            sum += self.alphas_[i] * self.y_[i] * self._compute_kernel(self.x_[i], X1) + self.b

        return sum

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Melatih model SVM (mengonversi label {0, 1} ke {-1, +1} secara internal)
        menggunakan algoritma SMO (Platt, 1998) atau Pegasos, serta melatih parameter Platt Scaling.
        """
        # TODO (Anggota 4): Implementasikan solver SMO / Pegasos & Platt Scaling
        raise NotImplementedError("SVM.fit belum diimplementasikan oleh Anggota 4.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Menghasilkan probabilitas posterior [P(y=0|X), P(y=1|X)] menggunakan Platt Scaling:
        P(y=1|x) = 1 / (1 + exp(A * f(x) + B))
        """
        # TODO (Anggota 4): Implementasikan predict_proba berbasis decision_function
        raise NotImplementedError("SVM.predict_proba belum diimplementasikan oleh Anggota 4.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Melakukan prediksi label kelas diskrit (0 atau 1) dari tanda f(x).
        """
        scores = self.decision_function(X)
        return (scores >= 0.0).astype(int)


def plot_svm_margin_2d(
    model: SVM,
    X_2d: np.ndarray,
    y: np.ndarray,
    output_path: str = "output/plots/svm_margin_linear.png",
    title: str = "SVM Decision Boundary & Margin",
):
    """
    Bonus Visualisasi Bidang Pemisah 2D (+1 Poin):
    Memplot titik data 2D, kurva bidang pemisah f(x) = 0, batas margin f(x) = +1 dan f(x) = -1,
    serta menyorot titik-titik Support Vectors (alpha_i > 0).
    Wajib dieksekusi untuk ketiga kernel:
    - output/plots/svm_margin_linear.png
    - output/plots/svm_margin_poly.png
    - output/plots/svm_margin_rbf.png
    """
    # TODO (Anggota 4): Implementasikan visualisasi 2D bidang pemisah, margin, & support vectors
    raise NotImplementedError("plot_svm_margin_2d belum diimplementasikan oleh Anggota 4.")

