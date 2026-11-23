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


def _laplacian_gram(X1: np.ndarray, X2: np.ndarray, gamma: float, dtype, block_bytes: int = 64 * 1024 * 1024) -> np.ndarray:
    """
    Matriks Gram kernel Laplacian: K_ij = exp(-gamma * sum_k |X1_ik - X2_jk|) (jarak L1).

    Dihitung per BLOK baris dan langsung ditulis ke buffer keluaran ber-dtype `dtype`, sehingga
    puncak memori ~ satu matriks n1 x n2 (+ satu blok ~64 MB), bukan dua matriks float64 penuh.
    Memakai scipy.cdist('cityblock') bila tersedia; jika tidak, fallback NumPy (loop per fitur).
    """
    n1, n2 = X1.shape[0], X2.shape[0]
    out = np.empty((n1, n2), dtype=dtype)
    X2_64 = np.ascontiguousarray(X2, dtype=np.float64)
    rows = max(1, block_bytes // (8 * max(n2, 1)))
    try:
        from scipy.spatial.distance import cdist
    except ImportError:  # pragma: no cover - scipy ada di requirements.txt
        cdist = None

    for start in range(0, n1, rows):
        a = np.ascontiguousarray(X1[start:start + rows], dtype=np.float64)
        if cdist is not None:
            D = cdist(a, X2_64, "cityblock")
        else:
            D = np.zeros((a.shape[0], n2), dtype=np.float64)
            for k in range(a.shape[1]):
                D += np.abs(a[:, k:k + 1] - X2_64[:, k][None, :])
        D *= -gamma
        np.exp(D, out=D)
        out[start:start + rows] = D
    return out


class SVM(BaseModel):
    """
    Implementasi Support Vector Machine (SVM) Classifier from Scratch (NumPy).

    Parameter:
        C (float): Parameter regularisasi soft-margin.
        kernel (str): Jenis kernel ('linear', 'poly', 'rbf', 'laplacian').
        degree (int): Derajat polinomial untuk kernel 'poly' (default=3).
        gamma (float | str | None): Koefisien kernel untuk 'poly', 'rbf', dan 'laplacian'
            (jika None atau 'scale': 1 / (n_features * var) untuk poly/rbf dan
            1 / (n_features * std) untuk laplacian; jika 'auto': 1 / n_features).
        coef0 (float): Konstanta independen r pada kernel polinomial (default=1.0).
        solver (str): Solver from-scratch ('smo' atau 'cvxopt' untuk kompatibilitas API).
        max_iter (int): Batas iterasi SMO dalam satuan 'putaran' data (maks. max_iter * n pembaruan).
        tol (float): Toleransi pelanggaran kondisi KKT.
        max_samples (int | None): Jika diisi dan n > max_samples, model dilatih pada subsampel
            berstrata sebesar max_samples (kelas dipertahankan proporsional). Model akhir
            tetap diskalakan Platt pada SELURUH data latih.
        random_state (int): Seed untuk subsampling berstrata (solver SMO sendiri deterministik).
        kernel_dtype (str): Presisi matriks Gram saat pelatihan ('float32' atau 'float64').
            float32 menghemat memori 2x (n=21k: ~1.8 GB vs ~3.5 GB); variabel dual, bias,
            dan decision_function tetap float64.
        probability (bool): Jika True (default), fit juga melatih Platt Scaling untuk predict_proba.
            Set False untuk melewatkannya (mis. saat tuning hiperparameter) agar fit lebih cepat.
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
        kernel_dtype: str = "float32",
        probability: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if C <= 0:
            raise ValueError("C harus lebih besar dari 0.")
        if kernel.lower() not in ("linear", "poly", "rbf", "laplacian"):
            raise ValueError("Kernel harus 'linear', 'poly', 'rbf', atau 'laplacian'.")
        if solver.lower() not in ("smo", "cvxopt"):
            raise ValueError("Solver harus 'smo' atau 'cvxopt'.")
        if max_samples is not None and max_samples < 2:
            raise ValueError("max_samples harus None atau >= 2.")
        if kernel_dtype not in ("float32", "float64"):
            raise ValueError("kernel_dtype harus 'float32' atau 'float64'.")

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
        self.kernel_dtype = kernel_dtype
        self.probability = bool(probability)

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
        # Default ('scale' atau None) dengan fallback aman:
        #   poly/rbf  : 1 / (n_features * var)   (jarak kuadrat ~ var)
        #   laplacian : 1 / (n_features * std)   (jarak L1 ~ std)
        var = float(np.var(X))
        scale = np.sqrt(var) if self.kernel == "laplacian" else var
        denom = n_features * scale
        return 1.0 / denom if denom > 0.0 else 1.0 / n_features

    # ------------------------------------------------------------------ kernel
    def _compute_kernel(self, X1: np.ndarray, X2: np.ndarray, dtype=np.float64) -> np.ndarray:
        """
        Menghitung matriks Gram K(X1, X2) sesuai pilihan kernel:
        1. Linear: K(x, x') = x^T x'
        2. Polynomial: K(x, x') = (gamma * x^T x' + coef0)^degree
        3. Gaussian (RBF): K(x, x') = exp(-gamma * ||x - x'||^2)
        4. Laplacian: K(x, x') = exp(-gamma * ||x - x'||_1)  (jarak L1, lebih tahan terhadap
           fitur ber-skala besar/berekor panjang dibanding RBF)

        Semua operasi setelah perkalian matriks dilakukan IN-PLACE pada satu buffer,
        sehingga puncak memori hanya satu matriks (bukan 3-4 temporary berukuran n x n).
        """
        gamma = self._gamma_
        if gamma is None:
            gamma = self._resolve_gamma(np.asarray(X1))
        if self.kernel == "laplacian":
            return _laplacian_gram(np.asarray(X1), np.asarray(X2), gamma, dtype)

        X1 = np.asarray(X1, dtype=dtype)
        X2 = np.asarray(X2, dtype=dtype)

        G = X1 @ X2.T
        if self.kernel == "linear":
            return G
        if self.kernel == "poly":
            G *= gamma
            G += self.coef0
            np.power(G, self.degree, out=G)
            return G
        if self.kernel == "rbf":
            # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b (dipotong >= 0 untuk galat pembulatan)
            x1_sq = np.einsum("ij,ij->i", X1, X1)[:, None]
            x2_sq = np.einsum("ij,ij->i", X2, X2)[None, :]
            G *= -2.0
            G += x1_sq
            G += x2_sq
            np.maximum(G, 0.0, out=G)
            G *= -gamma
            np.exp(G, out=G)
            return G
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

    def _solve_smo(self, K: np.ndarray, y_sv: np.ndarray, rng: np.random.Generator = None):
        """
        Menyelesaikan masalah dual SVM:
            min_a 0.5 a^T Q a - 1^T a,  0 <= a <= C,  y^T a = 0,  Q_ij = y_i y_j K_ij
        dengan SMO pemilihan pasangan orde-dua (maximal violating pair, Fan-Chen-Lin 2005,
        seperti libsvm). Tiap iterasi vektorial O(n); berhenti bila celah KKT < tol,
        sehingga konvergen deterministik ke solusi optimum (rng tidak dipakai).

        max_iter dihitung sebagai jumlah "putaran" data: batas total pembaruan pasangan
        adalah max_iter * n_samples.
        """
        n = len(y_sv)
        C, eps, tau = self.C, self.tol, 1e-12
        alphas = np.zeros(n, dtype=np.float64)
        grad = -np.ones(n, dtype=np.float64)      # gradien dual: Q a - 1
        diag = np.diag(K).astype(np.float64)
        pos, neg = y_sv > 0, y_sv < 0

        for _ in range(max(self.max_iter, 1) * n):
            up = (pos & (alphas < C)) | (neg & (alphas > 0.0))
            low = (pos & (alphas > 0.0)) | (neg & (alphas < C))
            if not up.any() or not low.any():
                break

            yG = y_sv * grad
            i = int(np.argmax(np.where(up, -yG, -np.inf)))
            g_max = -yG[i]
            g_max2 = yG[low].max()
            if g_max + g_max2 < eps:               # kondisi KKT terpenuhi dalam toleransi
                break

            # pilih j: penurunan objektif terbesar (orde-dua) di antara pelanggar
            yi = y_sv[i]
            Ki = K[i]
            b_t = g_max + yG                        # > 0 untuk pelanggar
            a_t = diag[i] + diag - 2.0 * yi * y_sv * Ki
            a_t = np.where(a_t > 0.0, a_t, tau)
            score = np.where(low & (b_t > 0.0), -(b_t * b_t) / a_t, np.inf)
            j = int(np.argmin(score))
            if not np.isfinite(score[j]):
                break

            yj = y_sv[j]
            quad = diag[i] + diag[j] - 2.0 * float(Ki[j])
            if quad <= 0.0:
                quad = tau
            ai_old, aj_old = alphas[i], alphas[j]

            if yi != yj:
                delta = (-grad[i] - grad[j]) / quad
                diff = ai_old - aj_old
                ai, aj = ai_old + delta, aj_old + delta
                if diff > 0.0:
                    if aj < 0.0:
                        aj, ai = 0.0, diff
                elif ai < 0.0:
                    ai, aj = 0.0, -diff
                if diff > 0.0:
                    if ai > C:
                        ai, aj = C, C - diff
                elif aj > C:
                    aj, ai = C, C + diff
            else:
                delta = (grad[i] - grad[j]) / quad
                s = ai_old + aj_old
                ai, aj = ai_old - delta, aj_old + delta
                if s > C:
                    if ai > C:
                        ai, aj = C, s - C
                elif aj < 0.0:
                    aj, ai = 0.0, s
                if s > C:
                    if aj > C:
                        aj, ai = C, s - C
                elif ai < 0.0:
                    ai, aj = 0.0, s

            d_ai, d_aj = ai - ai_old, aj - aj_old
            if d_ai == 0.0 and d_aj == 0.0:
                break
            alphas[i], alphas[j] = ai, aj
            grad += y_sv * ((yi * d_ai) * Ki + (yj * d_aj) * K[j])

        # Bias: rata-rata -y_t*grad_t pada SV bebas; jika tidak ada, titik tengah batasnya
        yG = y_sv * grad
        free = (alphas > 0.0) & (alphas < C)
        if free.any():
            b = -float(yG[free].mean())
        else:
            at_c, at_0 = alphas >= C, alphas <= 0.0
            ub = min(yG[at_c & neg].min(initial=np.inf), yG[at_0 & pos].min(initial=np.inf))
            lb = max(yG[at_c & pos].max(initial=-np.inf), yG[at_0 & neg].max(initial=-np.inf))
            r = 0.5 * (ub + lb) if np.isfinite(ub) and np.isfinite(lb) else 0.0
            b = -float(r)
        return alphas, b

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
        kdtype = np.dtype(self.kernel_dtype)
        if n_sub > _QP_WARN_SAMPLES:
            est_gb = n_sub * n_sub * kdtype.itemsize / 1024 ** 3
            warnings.warn(
                f"Jumlah sampel pelatihan ({n_sub}) melebihi {_QP_WARN_SAMPLES}; "
                f"matriks kernel penuh ~{est_gb:.2f} GB ({kdtype.name}). "
                "Pertimbangkan max_samples untuk subsampling berstrata.",
                RuntimeWarning,
            )

        Xq, yq = X[qp_idx], y_pm[qp_idx]
        K = self._compute_kernel(Xq, Xq, dtype=kdtype)
        alphas_q, b_val = self._solve_smo(K, yq, rng)
        del K  # bebaskan matriks Gram sebelum Platt scaling / prediksi

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

        if self.probability:
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
        if not self.probability:
            raise ValueError("predict_proba membutuhkan probability=True saat membuat model.")
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