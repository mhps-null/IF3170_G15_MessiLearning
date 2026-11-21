import os
import warnings
import numpy as np

from .base import BaseModel
from .utils import RANDOM_SEED

# Di atas jumlah sampel ini, matriks kernel penuh mulai berat (memori & waktu).
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
    Implementasi Support Vector Machine (SVM) Classifier from Scratch (NumPy).

    Parameter:
        C (float): Parameter regularisasi soft-margin.
        kernel (str): Jenis kernel ('linear', 'poly', 'rbf').
        degree (int): Derajat polinomial untuk kernel 'poly' (default=3).
        gamma (float | str | None): Koefisien kernel untuk 'poly' dan 'rbf'
            (jika None atau 'scale', gunakan 1 / (n_features * var); jika 'auto', 1 / n_features).
        coef0 (float): Konstanta independen r pada kernel polinomial (default=1.0).
        solver (str): Solver from-scratch ('smo' atau 'cvxopt' untuk kompatibilitas API).
        max_iter (int): Jumlah iterasi maksimum dual coordinate ascent / SMO.
        tol (float): Toleransi pelanggaran kondisi KKT.
        max_samples (int | None): Jika diisi dan n > max_samples, model dilatih pada subsampel
            berstrata sebesar max_samples (kelas dipertahankan proporsional). Model akhir
            tetap diskalakan Platt pada SELURUH data latih.
        random_state (int): Seed untuk subsampling dan urutan pembaruan SMO.
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        degree: int = 3,
        gamma: float | str = None,
        coef0: float = 1.0,
        solver: str = "smo",
        max_iter: int = 200,
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
        if solver.lower() not in ("smo", "cvxopt"):
            raise ValueError("Solver harus 'smo' atau 'cvxopt'.")
        if max_samples is not None and max_samples < 2:
            raise ValueError("max_samples harus None atau >= 2.")

        self.C = float(C)
        self.kernel = kernel.lower()
        self.degree = int(degree)
        self.gamma = gamma
        self.coef0 = float(coef0)
        self.solver = solver.lower()
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.max_samples = max_samples
        self.random_state = int(random_state)

        # Parameter hasil pelatihan
        self.alphas_ = None                    # alpha untuk SELURUH data latih
        self.b_ = 0.0
        self.support_vectors_ = None
        self.support_vector_labels_ = None     # label {-1, +1}
        self.support_vector_indices_ = None    # indeks pada data latih asli
        self.n_support_ = 0

        # Parameter Platt Scaling (A, B) untuk predict_proba
        self.platt_A_ = -1.0
        self.platt_B_ = 0.0

        self._gamma_ = None  # gamma efektif (diselesaikan saat fit)

    def _resolve_gamma(self, X: np.ndarray) -> float:
        if isinstance(self.gamma, (int, float)):
            return float(self.gamma)
        n_features = max(X.shape[1], 1)
        if self.gamma == "auto":
            return 1.0 / n_features
        # Default ('scale' atau None): 1 / (n_features * var) dengan fallback aman
        var = float(np.var(X))
        denom = n_features * var
        return 1.0 / denom if denom > 0.0 else 1.0 / n_features

    # ------------------------------------------------------------------ kernel
    def _compute_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Menghitung matriks Gram K(X1, X2) sesuai pilihan kernel:
        1. Linear: K(x, x') = x^T x'
        2. Polynomial: K(x, x') = (gamma * x^T x' + coef0)^degree
        3. Gaussian (RBF): K(x, x') = exp(-gamma * ||x - x'||^2)
        """
        X1 = np.asarray(X1, dtype=np.float64)
        X2 = np.asarray(X2, dtype=np.float64)
        gamma = self._gamma_
        if gamma is None:
            gamma = self._resolve_gamma(X1)

        if self.kernel == "linear":
            return X1 @ X2.T
        if self.kernel == "poly":
            return (gamma * (X1 @ X2.T) + self.coef0) ** self.degree
        if self.kernel == "rbf":
            # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b (dipotong >= 0 untuk galat pembulatan)
            x1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
            x2_sq = np.sum(X2 ** 2, axis=1, keepdims=True).T
            dist_sq = np.maximum(x1_sq + x2_sq - 2.0 * (X1 @ X2.T), 0.0)
            return np.exp(-gamma * dist_sq)
        raise ValueError(f"Kernel tidak dikenal: '{self.kernel}'.")

    # --------------------------------------------------------- decision function
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Menghitung skor keputusan f(x) = sum_{i in SV} alpha_i * y_i * K(x_i, x) + b.
        """
        if not self.is_fitted:
            raise ValueError("Model belum dilatih.")
        X = np.asarray(X, dtype=np.float64)
        if self.support_vectors_ is None or len(self.support_vectors_) == 0:
            return np.full(len(X), self.b_, dtype=np.float64)

        coef = self.alphas_[self.support_vector_indices_] * self.support_vector_labels_
        out = np.empty(len(X), dtype=np.float64)
        for s in range(0, len(X), _PREDICT_BATCH):  # batch: hindari matriks kernel raksasa
            K = self._compute_kernel(self.support_vectors_, X[s:s + _PREDICT_BATCH])
            out[s:s + _PREDICT_BATCH] = coef @ K
        return out + self.b_

    def _solve_smo(self, K: np.ndarray, y_sv: np.ndarray, rng: np.random.Generator):
        """
        Menyelesaikan masalah dual SVM dengan pembaruan koordinat ganda (SMO-style)
        pada variabel dual alpha_i in [0, C].
        """
        n_samples = len(y_sv)
        alphas = np.zeros(n_samples, dtype=np.float64)
        b = 0.0
        f = np.zeros(n_samples, dtype=np.float64)

        for _ in range(self.max_iter):
            num_changed = 0
            order = rng.permutation(n_samples)

            for i in order:
                E_i = f[i] - y_sv[i]
                violates_kkt = (y_sv[i] * E_i < -self.tol and alphas[i] < self.C) or (
                    y_sv[i] * E_i > self.tol and alphas[i] > 0.0
                )
                if not violates_kkt:
                    continue

                pool_size = min(64, n_samples)
                cand_pool = rng.choice(n_samples, size=pool_size, replace=False)
                cand_pool = cand_pool[cand_pool != i]
                if len(cand_pool) == 0:
                    continue

                E_cands = f[cand_pool] - y_sv[cand_pool]
                j = int(cand_pool[int(np.argmax(np.abs(E_i - E_cands)))])
                E_j = f[j] - y_sv[j]

                alpha_i_old = alphas[i]
                alpha_j_old = alphas[j]

                if y_sv[i] != y_sv[j]:
                    L = max(0.0, alpha_j_old - alpha_i_old)
                    H = min(self.C, self.C + alpha_j_old - alpha_i_old)
                else:
                    L = max(0.0, alpha_i_old + alpha_j_old - self.C)
                    H = min(self.C, alpha_i_old + alpha_j_old)

                if L >= H:
                    continue

                eta = K[i, i] + K[j, j] - 2.0 * K[i, j]
                if eta <= 1e-12:
                    continue

                alpha_j_new = alpha_j_old + y_sv[j] * (E_i - E_j) / eta
                alpha_j_new = float(np.clip(alpha_j_new, L, H))

                if abs(alpha_j_new - alpha_j_old) < 1e-6:
                    continue

                alpha_i_new = alpha_i_old + y_sv[i] * y_sv[j] * (alpha_j_old - alpha_j_new)

                b1 = (
                    b
                    - E_i
                    - y_sv[i] * (alpha_i_new - alpha_i_old) * K[i, i]
                    - y_sv[j] * (alpha_j_new - alpha_j_old) * K[i, j]
                )
                b2 = (
                    b
                    - E_j
                    - y_sv[i] * (alpha_i_new - alpha_i_old) * K[i, j]
                    - y_sv[j] * (alpha_j_new - alpha_j_old) * K[j, j]
                )

                if 0.0 < alpha_i_new < self.C:
                    b_new = b1
                elif 0.0 < alpha_j_new < self.C:
                    b_new = b2
                else:
                    b_new = 0.5 * (b1 + b2)

                delta_alpha_i = alpha_i_new - alpha_i_old
                delta_alpha_j = alpha_j_new - alpha_j_old
                delta_b = b_new - b

                f += (
                    delta_alpha_i * y_sv[i] * K[i, :]
                    + delta_alpha_j * y_sv[j] * K[j, :]
                    + delta_b
                )

                alphas[i] = alpha_i_new
                alphas[j] = alpha_j_new
                b = b_new
                num_changed += 1

            if num_changed == 0:
                break

        return alphas, float(b)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Melatih model SVM (mengonversi label {0, 1} ke {-1, +1} secara internal)
        menggunakan dual coordinate ascent / SMO from scratch dengan NumPy,
        lalu melatih parameter Platt Scaling pada seluruh data latih.
        """
        X = np.asarray(X, dtype=np.float64)
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
        rng = np.random.default_rng(self.random_state)

        self._gamma_ = self._resolve_gamma(X)

        if self.max_samples is not None and n_total > self.max_samples:
            qp_idx = []
            for label in (-1.0, 1.0):
                cls_idx = np.where(y_pm == label)[0]
                k = max(1, int(round(self.max_samples * len(cls_idx) / n_total)))
                qp_idx.append(rng.choice(cls_idx, size=min(k, len(cls_idx)), replace=False))
            qp_idx = np.sort(np.concatenate(qp_idx))
        else:
            qp_idx = np.arange(n_total)

        n_sub = len(qp_idx)
        if n_sub > _QP_WARN_SAMPLES:
            warnings.warn(
                f"Jumlah sampel pelatihan ({n_sub}) melebihi {_QP_WARN_SAMPLES}; "
                "matriks kernel penuh dapat memakan waktu dan memori lebih besar.",
                RuntimeWarning,
            )

        Xq, yq = X[qp_idx], y_pm[qp_idx]
        K = self._compute_kernel(Xq, Xq)
        alphas_q, b_val = self._solve_smo(K, yq, rng)

        sv_mask = alphas_q > 1e-7
        alphas_full = np.zeros(n_total, dtype=np.float64)
        alphas_full[qp_idx] = alphas_q
        sv_global = qp_idx[sv_mask]

        self.alphas_ = alphas_full
        self.b_ = float(b_val)
        self.support_vector_indices_ = sv_global
        self.support_vectors_ = X[sv_global]
        self.support_vector_labels_ = y_pm[sv_global]
        self.n_support_ = int(np.sum(sv_mask))
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
            if abs(det) < 1e-18:
                break
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
    """
    import matplotlib.pyplot as plt

    X_2d = np.asarray(X_2d, dtype=np.float64)
    y = np.asarray(y).ravel()

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.decision_function(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.contourf(xx, yy, zz, levels=25, cmap="RdBu", alpha=0.25)
    contours = ax.contour(
        xx,
        yy,
        zz,
        levels=[-1.0, 0.0, 1.0],
        colors=["#1f77b4", "#111111", "#d62728"],
        linestyles=["--", "-", "--"],
        linewidths=[1.3, 2.0, 1.3],
    )
    ax.clabel(contours, fmt={-1.0: "f=-1", 0.0: "f=0", 1.0: "f=+1"}, inline=True, fontsize=9)

    ax.scatter(
        X_2d[y == 0, 0],
        X_2d[y == 0, 1],
        c="#1f77b4",
        label="Class 0",
        edgecolors="k",
        s=30,
        alpha=0.8,
    )
    ax.scatter(
        X_2d[y == 1, 0],
        X_2d[y == 1, 1],
        c="#d62728",
        label="Class 1",
        edgecolors="k",
        s=30,
        alpha=0.8,
    )

    if model.support_vectors_ is not None and len(model.support_vectors_) > 0:
        sv = model.support_vectors_
        ax.scatter(
            sv[:, 0],
            sv[:, 1],
            s=110,
            facecolors="none",
            edgecolors="#f59e0b",
            linewidths=1.8,
            label=f"Support Vectors ({len(sv)})",
        )

    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path