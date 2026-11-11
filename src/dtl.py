"""
src/dtl.py
Modul Decision Tree Learning (DTL) from Scratch & Visualizer (Anggota 2).
Mendukung fitur numerik (continuous threshold), kategorikal, penanganan missing values (Quinlan C4.5),
visualisasi Top-N pohon keputusan (+1 poin bonus), dan post-pruning (+2 poin bonus).

Referensi Ilmiah:
- Quinlan, J. R. (1993). C4.5: Programs for Machine Learning. Morgan Kaufmann Publishers.
- Mitchell, T. M. (1997). Machine Learning. McGraw-Hill. (Reduced Error Pruning & Rule Post-Pruning).
"""

import numpy as np
from .base import BaseModel


class TreeNode:
    """Representasi simpul (node) pada pohon keputusan."""

    def __init__(
        self,
        feature_idx=None,
        threshold=None,
        is_categorical: bool = False,
        children: dict = None,
        left=None,
        right=None,
        value=None,
        class_distribution=None,
        n_samples: int = 0,
    ):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.is_categorical = is_categorical
        self.children = children or {}
        self.left = left
        self.right = right
        self.value = value  # Label kelas mayoritas pada simpul ini
        self.class_distribution = class_distribution  # Distribusi probabilitas [P(y=0), P(y=1)]
        self.n_samples = n_samples


class DecisionTree(BaseModel):
    """
    Implementasi Decision Tree Classifier (ID3 / C4.5 / CART) from Scratch.

    Parameter:
        criterion (str): 'c45' (Gain Ratio - direkomendasikan), 'id3' (Entropy), atau 'cart' (Gini).
        max_depth (int): Kedalaman maksimum pohon.
        min_samples_split (int): Jumlah minimum sampel untuk memecah simpul.
        use_pruning (bool): Aktifkan post-pruning (Reduced Error Pruning / CCP) untuk Bonus Optimasi (+2 poin).
    """

    def __init__(
        self,
        criterion: str = "c45",
        max_depth: int = None,
        min_samples_split: int = 2,
        use_pruning: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.criterion = criterion.lower()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.use_pruning = use_pruning
        self.root_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        Melatih pohon keputusan pada data X (mendukung campuran numerik, kategorikal, dan NaN).
        Jika `use_pruning=True` dan data validasi disediakan, jalankan post-pruning.
        """
        # TODO (Anggota 2): Implementasikan konstruksi pohon & penanganan missing value ala Quinlan (C4.5)
        raise NotImplementedError("DecisionTree.fit belum diimplementasikan oleh Anggota 2.")

    def prune(self, X_val: np.ndarray, y_val: np.ndarray):
        """
        Bonus Optimasi DTL (+2 Poin):
        Reduced Error Pruning (REP) / Cost-Complexity Pruning (CCP) secara bottom-up.
        """
        # TODO (Anggota 2): Implementasikan post-pruning dengan referensi Mitchell (1997)
        raise NotImplementedError("DecisionTree.prune belum diimplementasikan oleh Anggota 2.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Menghasilkan probabilitas posterior [P(y=0|X), P(y=1|X)] berukuran (n_samples, 2).
        Mendukung agregasi bobot pecahan (fractional weight) apabila sampel memiliki nilai NaN.
        """
        # TODO (Anggota 2): Implementasikan penelusuran pohon untuk prediksi probabilitas
        raise NotImplementedError("DecisionTree.predict_proba belum diimplementasikan oleh Anggota 2.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Melakukan prediksi label kelas diskrit (0 atau 1).
        """
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)


def export_tree_visualization(
    model: DecisionTree,
    top_n: int = 3,
    feature_names: list = None,
    output_path: str = "output/plots/dtl_tree_top_n.png",
):
    """
    Bonus Visualisasi Tree Top-N (+1 Poin):
    Menggambar struktur pohon keputusan dari root hingga kedalaman `top_n`,
    menampilkan aturan partisi (split rule), jumlah sampel, dan kelas dominan.
    """
    # TODO (Anggota 2): Implementasikan visualisasi pohon menggunakan matplotlib / graphviz
    raise NotImplementedError("export_tree_visualization belum diimplementasikan oleh Anggota 2.")

