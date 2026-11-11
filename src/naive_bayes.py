"""
src/naive_bayes.py
Modul Naive Bayes / Bayesian Network Classifier, Graph Visualizer, & Cost-Sensitive Helper (Anggota 5).
Mengimplementasikan Gaussian Naive Bayes dalam ruang log-likelihood,
Bonus Visualisasi Graf DAG Bayesian Network (+1 poin),
Bonus Optimasi Tree-Augmented Naive Bayes / TAN (+2 poin),
dan Bonus Cost-Sensitive Classification (+2 poin).

Referensi Ilmiah:
- Friedman, N., Geiger, D., & Goldszmidt, M. (1997). Bayesian Network Classifiers.
  Machine Learning, 29(2-3), 131-163.
"""

import numpy as np
from .base import BaseModel


class NaiveBayes(BaseModel):
    """
    Implementasi Gaussian Naive Bayes & Tree-Augmented Naive Bayes (TAN) from Scratch.

    Parameter:
        var_smoothing (float): Konstanta perataan variansi untuk mencegah pembagian dengan nol.
        use_tan (bool): Aktifkan struktur Tree-Augmented Naive Bayes (TAN) untuk Bonus Optimasi (+2 poin).
    """

    def __init__(self, var_smoothing: float = 1e-9, use_tan: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.var_smoothing = var_smoothing
        self.use_tan = use_tan

        self.class_priors_ = None  # P(y=c)
        self.means_ = None  # mu_{jc} berukuran (n_classes, n_features)
        self.vars_ = None  # sigma^2_{jc} berukuran (n_classes, n_features)
        self.tan_edges_ = []  # Daftar busur ketergantungan antar-fitur (X_i -> X_j) jika use_tan=True

    def _compute_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Menghitung log-posterior unnormalized untuk setiap kelas c:
        log P(y=c | X) = log P(y=c) + sum_{j=1}^d log P(x_j | y=c)
        """
        # TODO (Anggota 5): Implementasikan log-likelihood Gaussian (dan TAN jika use_tan=True)
        raise NotImplementedError("NaiveBayes._compute_log_likelihood belum diimplementasikan oleh Anggota 5.")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Menghitung prior kelas P(y=c), rata-rata mu_{jc}, dan variansi sigma^2_{jc}.
        Jika `use_tan=True`, bangun Maximum Spanning Tree berbasis Conditional Mutual Information I(X_i; X_j | Y).
        """
        # TODO (Anggota 5): Implementasikan estimasi parameter Gaussian NB & struktur TAN (Friedman et al., 1997)
        raise NotImplementedError("NaiveBayes.fit belum diimplementasikan oleh Anggota 5.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Menghasilkan probabilitas posterior [P(y=0|X), P(y=1|X)] menggunakan fungsi softmax
        pada ruang log-posterior (dengan log-sum-exp trick untuk mencegah underflow).
        """
        # TODO (Anggota 5): Implementasikan softmax atas log-posterior
        raise NotImplementedError("NaiveBayes.predict_proba belum diimplementasikan oleh Anggota 5.")

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

