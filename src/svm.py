import os
from cvxopt import matrix, spmatrix, solvers
import warnings

import numpy as np

from .base import BaseModel
from .utils import RANDOM_SEED

# Di atas jumlah sampel ini, QP dense cvxopt mulai berat (memori & waktu).
_QP_WARN_SAMPLES = 3000
# Ukuran batch saat menghitung decision_function agar matriks kernel tidak membengkak.
_PREDICT_BATCH = 4096


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid 1 / (1 + exp(-z)) yang stabil secara numerik."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class SVM(BaseModel):
    """
    Implementasi Support Vector Machine (SVM) Classifier from Scratch.

    Parameter:
        C (float): Parameter regularisasi soft-margin.
        kernel (str): Jenis kernel ('linear', 'poly', 'rbf').
        degree (int): Derajat polinomial untuk kernel 'poly' (default=3).
        gamma (float): Koefisien kernel untuk 'poly' dan 'rbf' (jika None, gunakan 1 / n_features).
        coef0 (float): Konstanta independen r pada kernel polinomial (default=1.0).
        solver (str): Saat ini hanya 'cvxopt' (QP dual).
        max_iter (int): Jumlah iterasi maksimum interior-point cvxopt.
        tol (float): Toleransi numerik solver cvxopt (abstol/reltol/feastol).
        max_samples (int | None): Jika diisi dan n > max_samples, QP dilatih pada subsampel
            berstrata sebesar max_samples (kelas dipertahankan proporsional). Model akhir
            tetap diskalakan Platt pada SELURUH data latih.
        random_state (int): Seed untuk subsampling.
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        degree: int = 3,
        gamma: float = None,
        coef0: float = 1.0,
        solver: str = "cvxopt",
        max_iter: int = 1000,
        tol: float = 1e-3,
        max_samples: int = None,
        random_state: int = RANDOM_SEED,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if C <= 0:
            raise ValueError("C harus lebih besar dari 0.")
        if kernel.lower() not in ("linear", "poly", "rbf"):
            raise ValueError("Kernel harus 'linear', 'poly', atau 'rbf'.")
        if solver.lower() != "cvxopt":
            raise ValueError("Hanya solver 'cvxopt' yang diimplementasikan di berkas ini.")
        if max_samples is not None and max_samples < 2:
            raise ValueError("max_samples harus None atau >= 2.")

        self.C = C
        self.kernel = kernel.lower()
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.solver = solver.lower()
        self.max_iter = max_iter
        self.tol = tol
        self.max_samples = max_samples
        self.random_state = random_state

        # Parameter hasil pelatihan
        self.alphas_ = None                    # alpha untuk SELURUH data latih (0 jika tidak dipakai QP)
        self.b_ = 0.0
        self.support_vectors_ = None
        self.support_vector_labels_ = None     # label {-1, +1}
        self.support_vector_indices_ = None    # indeks pada data latih asli

        # Parameter Platt Scaling (A, B) untuk predict_proba
        self.platt_A_ = -1.0
        self.platt_B_ = 0.0

        self._gamma_ = None  # gamma efektif (setelah gamma=None diselesaikan di fit)

    # ------------------------------------------------------------------ kernel
    def _compute_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Menghitung matriks Gram K(X1, X2) sesuai pilihan kernel:
        1. Linear: K(x, x') = x^T x'
        2. Polynomial: K(x, x') = (gamma * x^T x' + coef0)^degree
        3. Gaussian (RBF): K(x, x') = exp(-gamma * ||x - x'||^2)
        """
        X1 = np.asarray(X1, dtype=float)
        X2 = np.asarray(X2, dtype=float)
        gamma = self._gamma_
        if gamma is None:
            gamma = self.gamma if self.gamma is not None else 1.0 / X1.shape[1]

        if self.kernel == "linear":
            return X1 @ X2.T
        if self.kernel == "poly":
            return (gamma * (X1 @ X2.T) + self.coef0) ** self.degree
        if self.kernel == "rbf":
            # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b  (dipotong >= 0 untuk galat pembulatan)
            sq = (X1 ** 2).sum(1)[:, None] + (X2 ** 2).sum(1)[None, :] - 2.0 * (X1 @ X2.T)
            return np.exp(-gamma * np.maximum(sq, 0.0))
        raise ValueError(f"Kernel tidak dikenal: '{self.kernel}'.")

    # --------------------------------------------------------- decision function
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Menghitung skor keputusan f(x) = sum_{i in SV} alpha_i * y_i * K(x_i, x) + b.
        """
        if not self.is_fitted:
            raise ValueError("Model belum dilatih.")
        X = np.asarray(X, dtype=float)
        coef = self.alphas_[self.support_vector_indices_] * self.support_vector_labels_
        out = np.empty(len(X))
        for s in range(0, len(X), _PREDICT_BATCH):  # batch: hindari matriks kernel raksasa
            K = self._compute_kernel(self.support_vectors_, X[s:s + _PREDICT_BATCH])
            out[s:s + _PREDICT_BATCH] = coef @ K
        return out + self.b_

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Melatih model SVM (mengonversi label {0, 1} ke {-1, +1} secara internal)
        dengan menyelesaikan masalah dual sebagai QP lewat cvxopt, lalu melatih
        parameter Platt Scaling.

        Dual:  min_a  1/2 a^T Q a - 1^T a
               s.t.   0 <= a_i <= C,   y^T a = 0,   Q_ij = y_i y_j K(x_i, x_j)
        """

        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        if len(X) == 0:
            raise ValueError("Data training tidak boleh kosong.")
        if len(X) != len(y):
            raise ValueError("Jumlah sampel X dan y harus sama.")
        if set(np.unique(y).tolist()) - {0, 1}:
            raise ValueError("Label y harus berupa {0, 1}.")
        if len(np.unique(y)) < 2:
            raise ValueError("Data latih harus memuat kedua kelas (0 dan 1).")
        y_pm = np.where(y == 1, 1.0, -1.0)
        n_total = len(y_pm)

        self._gamma_ = self.gamma if self.gamma is not None else 1.0 / X.shape[1]

        
        if self.max_samples is not None and n_total > self.max_samples:
            rng = np.random.default_rng(self.random_state)
            qp_idx = []
            for label in (-1.0, 1.0):
                cls_idx = np.where(y_pm == label)[0]
                k = max(1, int(round(self.max_samples * len(cls_idx) / n_total)))
                qp_idx.append(rng.choice(cls_idx, size=min(k, len(cls_idx)), replace=False))
            qp_idx = np.sort(np.concatenate(qp_idx))
        else:
            qp_idx = np.arange(n_total)
        Xq, yq = X[qp_idx], y_pm[qp_idx]
        n = len(qp_idx)



        K = self._compute_kernel(Xq, Xq)
        Q = np.outer(yq, yq) * K
        P = matrix(Q + 1e-10 * np.eye(n))
        q = matrix(-np.ones(n))
        rows = list(range(2 * n))
        cols = list(range(n)) * 2
        vals = [-1.0] * n + [1.0] * n
        G = spmatrix(vals, rows, cols, (2 * n, n))
        h = matrix(np.r_[np.zeros(n), np.full(n, float(self.C))])
        A = matrix(yq.reshape(1, -1))
        b = matrix(0.0)

        sol = solvers.qp(
            P, q, G, h, A, b,
            options={
                "show_progress": False,
                "maxiters": int(self.max_iter),
                "abstol": 1e-8,
                "reltol": 1e-8,
                "feastol": 1e-8,
            },
        )
        if sol["status"] not in ("optimal", "unknown"):
            raise RuntimeError(f"cvxopt gagal menyelesaikan QP dual (status: {sol['status']}).")
        alphas_q = np.array(sol["x"]).ravel()

        sv_tol = 1e-5 * self.C
        sv_mask = alphas_q > sv_tol
        free_mask = sv_mask & (alphas_q < self.C - sv_tol)

        f_no_bias = K @ (alphas_q * yq)
        if free_mask.any():
            b_val = np.mean(yq[free_mask] - f_no_bias[free_mask])
        elif sv_mask.any():
            b_val = np.mean(yq[sv_mask] - f_no_bias[sv_mask])
        else:
            b_val = 0.0


        alphas_full = np.zeros(n_total)
        alphas_full[qp_idx] = alphas_q
        sv_global = qp_idx[sv_mask]

        self.alphas_ = alphas_full
        self.b_ = float(b_val)
        self.support_vector_indices_ = sv_global
        self.support_vectors_ = X[sv_global]
        self.support_vector_labels_ = y_pm[sv_global]
        self.classes_ = np.array([0, 1])
        self.is_fitted = True

        scores = self.decision_function(X)
        self.platt_A_, self.platt_B_ = self._fit_platt(scores, y_pm)
        return self

    # ------------------------------------------------------------- Platt scaling
    @staticmethod
    def _fit_platt(f: np.ndarray, y_pm: np.ndarray, max_iter: int = 100):
        """
        Mencari (A, B) yang meminimalkan negative log-likelihood dari
        P(y=1|f) = 1 / (1 + exp(A f + B)) dengan metode Newton + backtracking
        (Lin, Lin & Weng, 2007), memakai target yang dihaluskan untuk mengurangi overfitting.
        """
        n_pos = float(np.sum(y_pm == 1))
        n_neg = float(len(y_pm) - n_pos)
        t = np.where(y_pm == 1, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))

        A, B = 0.0, np.log((n_neg + 1.0) / (n_pos + 1.0))
        sigma = 1e-12

        def nll(A_, B_):
            z = A_ * f + B_
            return float(np.sum((t - 1.0) * z + np.logaddexp(0.0, z)))

        fval = nll(A, B)
        for _ in range(max_iter):
            p = _sigmoid(-(A * f + B))
            d2 = p * (1.0 - p)
            d1 = t - p
            g1, g2 = np.sum(f * d1), np.sum(d1)
            if abs(g1) < 1e-5 and abs(g2) < 1e-5:
                break
            h11 = sigma + np.sum(f * f * d2)
            h22 = sigma + np.sum(d2)
            h21 = np.sum(f * d2)
            det = h11 * h22 - h21 * h21
            dA = -(h22 * g1 - h21 * g2) / det
            dB = -(-h21 * g1 + h11 * g2) / det
            gd = g1 * dA + g2 * dB

            step, improved = 1.0, False
            while step >= 1e-10:
                new_A, new_B = A + step * dA, B + step * dB
                new_f = nll(new_A, new_B)
                if new_f < fval + 1e-4 * step * gd:
                    A, B, fval, improved = new_A, new_B, new_f, True
                    break
                step *= 0.5
            if not improved:
                break
        return float(A), float(B)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Menghasilkan probabilitas posterior [P(y=0|X), P(y=1|X)] menggunakan Platt Scaling:
        P(y=1|x) = 1 / (1 + exp(A * f(x) + B))
        """
        f = self.decision_function(X)
        p1 = _sigmoid(-(self.platt_A_ * f + self.platt_B_))
        return np.column_stack([1.0 - p1, p1])

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

    `model` harus sudah dilatih pada data 2 fitur yang sama dengan `X_2d`
    (mis. dua komponen PCA pertama). Mengembalikan `output_path`.
    """
    import matplotlib.pyplot as plt

    if not model.is_fitted:
        raise ValueError("Model belum dilatih.")
    X_2d = np.asarray(X_2d, dtype=float)
    if X_2d.ndim != 2 or X_2d.shape[1] != 2 or model.support_vectors_.shape[1] != 2:
        raise ValueError("X_2d dan model harus berdimensi 2 fitur.")
    y01 = (np.asarray(y).ravel() == 1).astype(int)

    pad_x = 0.1 * np.ptp(X_2d[:, 0]) + 1e-9
    pad_y = 0.1 * np.ptp(X_2d[:, 1]) + 1e-9
    xs = np.linspace(X_2d[:, 0].min() - pad_x, X_2d[:, 0].max() + pad_x, 300)
    ys = np.linspace(X_2d[:, 1].min() - pad_y, X_2d[:, 1].max() + pad_y, 300)
    xx, yy = np.meshgrid(xs, ys)
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    lo, hi = min(Z.min(), -1.0) - 1.0, max(Z.max(), 1.0) + 1.0
    ax.contourf(xx, yy, Z, levels=[lo, 0.0, hi], colors=["#f4a261", "#8ecae6"], alpha=0.25)
    ax.contour(xx, yy, Z, levels=[-1.0, 0.0, 1.0], colors="k",
               linestyles=["--", "-", "--"], linewidths=[1.0, 2.0, 1.0])
    ax.scatter(X_2d[y01 == 0, 0], X_2d[y01 == 0, 1], c="#e76f51", s=18, edgecolors="none",
               label="Kelas 0")
    ax.scatter(X_2d[y01 == 1, 0], X_2d[y01 == 1, 1], c="#219ebc", s=18, edgecolors="none",
               label="Kelas 1")
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=120,
               facecolors="none", edgecolors="k", linewidths=1.2,
               label=f"Support Vectors ({len(model.support_vectors_)})")
    ax.plot([], [], "k-", linewidth=2.0, label="f(x) = 0")
    ax.plot([], [], "k--", linewidth=1.0, label="Margin f(x) = ±1")
    ax.set_xlabel("Fitur 1")
    ax.set_ylabel("Fitur 2")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path