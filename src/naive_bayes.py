"""
src/naive_bayes.py
Modul Naive Bayes / Bayesian Network Classifier, Graph Visualizer, & Cost-Sensitive Helper (Anggota 5).
Mengimplementasikan Categorical Naive Bayes (berbasis frekuensi + Laplace smoothing, sesuai slide
kuliah IF3170 "BayesNet") dalam ruang log-likelihood,
Bonus Visualisasi Graf DAG Bayesian Network (+1 poin),
Bonus Optimasi Tree-Augmented Naive Bayes / TAN (+2 poin),
dan Bonus Cost-Sensitive Classification (+2 poin).

Referensi Ilmiah:
- Friedman, N., Geiger, D., & Goldszmidt, M. (1997). Bayesian Network Classifiers.
  Machine Learning, 29(2-3), 131-163.
"""

import numbers

import numpy as np
from .base import BaseModel


class NaiveBayes(BaseModel):
    """
    Categorical Naive Bayes from Scratch (baseline sesuai slide kuliah); TAN menyusul pada fase berikutnya.

    Masukan X adalah matriks bilangan bulat non-negatif (n_samples, n_features) berisi kode kategori,
    misalnya keluaran `DiscreteNBPreprocessor`. Kelas target biner {0, 1}; kolom keluaran
    `predict_proba` berurutan [P(y=0|X), P(y=1|X)].

    Estimasi parameter (N = jumlah sampel, N_c = sampel kelas c, K_j = jumlah kategori fitur j):
        P(C=c)            = N_c / N
        P(X_j=v | C=c)    = (N_{j,v,c} + alpha) / (N_c + alpha * K_j)
    Semua probabilitas disimpan dalam bentuk log.

    Parameter:
        alpha (float): parameter Laplace smoothing, alpha >= 0. alpha=0 berarti tanpa smoothing
            (probabilitas 0 -> log = -inf, lihat predict_proba untuk penanganannya).
        use_tan (bool): struktur Tree-Augmented Naive Bayes (Bonus +2 poin). Belum tersedia pada
            fase ini; use_tan=True akan memunculkan NotImplementedError saat fit().
        smooth_class_prior (bool): jika True, prior memakai (N_c + alpha) / (N + alpha * n_classes)
            (varian yang ditampilkan di slide 49-50 sebagai "10/16, 6/16"). Default False mengikuti
            rumus P(C=c) = N_c / N.
    """

    def __init__(self, alpha: float = 1.0, use_tan: bool = False, smooth_class_prior: bool = False, **kwargs):
        super().__init__(**kwargs)
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, numbers.Real)
            or not np.isfinite(alpha)
            or alpha < 0
        ):
            raise ValueError(f"alpha harus bilangan real berhingga >= 0, diterima: {alpha!r}")
        self.alpha = float(alpha)
        self.use_tan = bool(use_tan)
        self.smooth_class_prior = bool(smooth_class_prior)

        self.classes_ = np.array([0, 1])
        self.n_features_in_ = None
        self.n_categories_ = None       # K_j per fitur, berukuran (n_features,)
        self.class_counts_ = None       # N_c, berukuran (2,)
        self.class_priors_ = None       # P(y=c), berukuran (2,)
        self.log_class_priors_ = None   # log P(y=c)
        self.feature_counts_ = None     # list panjang d; elemen ke-j berukuran (2, K_j): N_{j,v,c}
        self.feature_log_prob_ = None   # list panjang d; elemen ke-j berukuran (2, K_j): log P(X_j=v|C=c)
        self.tan_edges_ = []            # busur antar-fitur (X_i -> X_j) jika use_tan=True (fase TAN)

    @staticmethod
    def _validate_codes(X, name: str = "X") -> np.ndarray:
        """Memastikan X adalah matriks 2D tak-kosong berisi bilangan bulat non-negatif; mengembalikan int64."""
        try:
            arr = np.asarray(X)
            arr_f = arr.astype(np.float64)
        except (TypeError, ValueError):
            raise ValueError(f"{name} harus berupa array numerik.")
        if arr_f.ndim != 2:
            raise ValueError(f"{name} harus 2 dimensi (n_samples, n_features), diterima ndim={arr_f.ndim}.")
        if arr_f.shape[0] == 0 or arr_f.shape[1] == 0:
            raise ValueError(f"{name} kosong: shape={arr_f.shape}.")
        if not np.all(np.isfinite(arr_f)):
            raise ValueError(f"{name} memuat NaN/inf; gunakan kode MISSING dari DiscreteNBPreprocessor.")
        if np.any(arr_f < 0) or np.any(arr_f != np.floor(arr_f)):
            raise ValueError(f"{name} harus berisi kode kategori bilangan bulat non-negatif.")
        return arr_f.astype(np.int64)

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("NaiveBayes belum di-fit. Panggil fit(X, y) terlebih dahulu.")

    def fit(self, X: np.ndarray, y: np.ndarray, n_categories=None):
        """
        Menghitung prior kelas P(y=c) dan tabel P(X_j=v | y=c) dengan Laplace smoothing.

        Parameter:
            X: (n_samples, n_features) kode kategori bilangan bulat non-negatif.
            y: (n_samples,) label biner {0, 1}; kedua kelas harus muncul pada data latih.
            n_categories: opsional, jumlah kategori K_j per fitur (mis. `preprocessor.n_categories_`).
                Jika None, K_j = max(X[:, j]) + 1.
        Mengembalikan:
            self
        """
        if self.use_tan:
            raise NotImplementedError("use_tan=True belum tersedia; TAN diimplementasikan pada fase berikutnya.")

        X = self._validate_codes(X, "X")
        y_arr = np.asarray(y)
        if y_arr.ndim != 1 or y_arr.shape[0] != X.shape[0]:
            raise ValueError(f"y harus 1 dimensi dengan panjang {X.shape[0]}, diterima shape={y_arr.shape}.")
        try:
            y_f = y_arr.astype(np.float64)
        except (TypeError, ValueError):
            raise ValueError("y harus berupa label numerik biner {0, 1}.")
        if not np.all(np.isin(y_f, (0.0, 1.0))):
            raise ValueError("y hanya boleh berisi label 0 atau 1.")
        y_int = y_f.astype(np.int64)
        if len(np.unique(y_int)) < 2:
            raise ValueError("Data latih harus memuat kedua kelas (0 dan 1).")

        n_samples, n_features = X.shape
        min_k = X.max(axis=0) + 1
        if n_categories is None:
            k = min_k
        else:
            k = np.asarray(n_categories)
            if k.shape != (n_features,) or not np.issubdtype(k.dtype, np.integer):
                raise ValueError(f"n_categories harus array bilangan bulat berukuran ({n_features},).")
            if np.any(k < min_k):
                raise ValueError("n_categories lebih kecil dari kode kategori terbesar yang ada pada X.")
            k = k.astype(np.int64)

        n_classes = len(self.classes_)
        class_counts = np.array([np.sum(y_int == c) for c in self.classes_], dtype=np.int64)
        if self.smooth_class_prior:
            priors = (class_counts + self.alpha) / (n_samples + self.alpha * n_classes)
        else:
            priors = class_counts / n_samples

        feature_counts, feature_log_prob = [], []
        with np.errstate(divide="ignore"):
            for j in range(n_features):
                counts = np.zeros((n_classes, k[j]), dtype=np.int64)
                for ci, c in enumerate(self.classes_):
                    counts[ci] = np.bincount(X[y_int == c, j], minlength=k[j])
                prob = (counts + self.alpha) / (class_counts[:, None] + self.alpha * k[j])
                feature_counts.append(counts)
                feature_log_prob.append(np.log(prob))

            self.log_class_priors_ = np.log(priors)

        self.n_features_in_ = n_features
        self.n_categories_ = k
        self.class_counts_ = class_counts
        self.class_priors_ = priors
        self.feature_counts_ = feature_counts
        self.feature_log_prob_ = feature_log_prob
        self.tan_edges_ = []
        self.is_fitted = True
        return self

    def _validate_predict_input(self, X) -> np.ndarray:
        self._check_fitted()
        X = self._validate_codes(X, "X")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X harus memiliki {self.n_features_in_} fitur, diterima {X.shape[1]}.")
        over = X >= self.n_categories_[None, :]
        if np.any(over):
            j = int(np.argwhere(over)[0][1])
            raise ValueError(
                f"Kode kategori di luar rentang pada fitur indeks {j} (K={self.n_categories_[j]}). "
                "Petakan kategori baru ke kode UNKNOWN melalui DiscreteNBPreprocessor."
            )
        return X

    def _compute_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Menghitung log-posterior unnormalized untuk setiap kelas c:
        log P(y=c | X) ∝ log P(y=c) + sum_{j=1}^d log P(x_j | y=c)
        Mengembalikan array (n_samples, 2); kolom mengikuti urutan `classes_` = [0, 1].
        """
        X = self._validate_predict_input(X)
        scores = np.tile(self.log_class_priors_, (X.shape[0], 1))
        for j in range(self.n_features_in_):
            scores += self.feature_log_prob_[j][:, X[:, j]].T
        return scores

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Menghasilkan probabilitas posterior [P(y=0|X), P(y=1|X)] menggunakan softmax pada
        ruang log-posterior (log-sum-exp trick untuk mencegah underflow).

        Kasus khusus (hanya mungkin saat alpha=0): jika log-posterior kedua kelas = -inf
        (seperti slide 48), posterior tidak terdefinisi dan dikembalikan distribusi seragam [0.5, 0.5].
        """
        scores = self._compute_log_likelihood(X)
        row_max = np.max(scores, axis=1, keepdims=True)
        undefined = ~np.isfinite(row_max[:, 0])
        shifted = np.exp(scores - np.where(np.isfinite(row_max), row_max, 0.0))
        total = shifted.sum(axis=1, keepdims=True)
        proba = np.empty_like(shifted)
        proba[~undefined] = shifted[~undefined] / total[~undefined]
        proba[undefined] = 1.0 / len(self.classes_)
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Melakukan prediksi label kelas diskrit (0 atau 1) dengan ambang batas standar 0.5.
        """
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)


def predict_cost_sensitive(model: BaseModel, X: np.ndarray, cost_fp: float = 2.0, cost_fn: float = 1.0) -> np.ndarray:
    """
    Bonus Cost-Sensitive Classification (+2 Poin):
    Menghitung ambang batas keputusan optimal (Optimal Bayesian Decision Threshold):
        tau* = cost_fp / (cost_fp + cost_fn)
    Untuk cost_fp = 2.0 dan cost_fn = 1.0, tau* = 2/3 ≈ 0.6667.
    Dapat digunakan pada model apa pun yang mengimplementasikan `BaseModel.predict_proba(X)`.
    """
    optimal_threshold = cost_fp / (cost_fp + cost_fn)
    probs = model.predict_proba(X)
    return (probs[:, 1] >= optimal_threshold).astype(int)


def export_bayesian_network_graph(
    model: NaiveBayes,
    feature_names: list = None,
    target_name: str = "is_canceled (Y)",
    output_path: str = "output/plots/bayesian_network_graph.png",
):
    """
    Bonus Visualisasi Graf Bayesian Network (+1 Poin):
    Memvisualisasikan Directed Acyclic Graph (DAG) menggunakan `networkx` dan `matplotlib`:
    - Simpul sentral: Variabel Target Kelas (Y)
    - Simpul anak: Seluruh fitur masukan (X_1, ..., X_d)
    - Busur panah: Y -> X_j (serta X_i -> X_j apabila menggunakan TAN).
    """
    # TODO (Anggota 5): Implementasikan visualisasi DAG dengan networkx & matplotlib
    raise NotImplementedError("export_bayesian_network_graph belum diimplementasikan oleh Anggota 5.")

