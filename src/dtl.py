r"""
src/dtl.py
Modul Decision Tree Learning (DTL) from Scratch & Visualizer (Anggota 2).
Mendukung fitur numerik (continuous threshold), kategorikal, penanganan missing values (Quinlan C4.5),
visualisasi Top-N pohon keputusan (+1 poin bonus), dan post-pruning (+2 poin bonus).

 Dasar Teori & Formulasi Matematis:
1. Impurity & Information Gain (ID3 - Quinlan, 1986):
   $$\text{Entropy}(S) = -\sum_{c \in \{0, 1\}} p_c \log_2(p_c), \quad p_c = \frac{\sum_{i \in S, y_i = c} w_i}{\sum_{i \in S} w_i}$$

2. Penanganan Missing Values & Gain Ratio (C4.5 - Quinlan, 1993):
   Saat atribut $A$ memiliki nilai hilang (NaN/Null), himpunan sampel $S$ dipartisi menjadi
   $S_{\text{known}}$ (nilai diketahui) dan $S_{\text{missing}}$ (nilai hilang).
   Information Gain diskalakan dengan proporsi bobot sampel yang diketahui:
   $$\text{Gain}(S, A) = \frac{|S_{\text{known}}|}{|S|} \times \left( \text{Entropy}(S_{\text{known}}) - \sum_{v} \frac{|S_v|}{|S_{\text{known}}|} \text{Entropy}(S_v) \right)$$
   $$\text{SplitInfo}(S, A) = -\sum_{v} \frac{|S_v|}{|S|} \log_2\left(\frac{|S_v|}{|S|}\right) - \frac{|S_{\text{missing}}|}{|S|} \log_2\left(\frac{|S_{\text{missing}}|}{|S|}\right)$$
   $$\text{GainRatio}(S, A) = \frac{\text{Gain}(S, A)}{\text{SplitInfo}(S, A)}$$
   Saat memecah simpul ke cabang anak $v$, sampel dengan nilai missing didistribusikan ke seluruh
   cabang anak dengan bobot pecahan (fractional weight) sebanding dengan frekuensi cabang:
   $$w_i^{(v)} = w_i \times \frac{|S_v|}{|S_{\text{known}}|}$$

3. Gini Impurity (CART - Breiman et al., 1984):
   $$\text{Gini}(S) = 1 - \sum_{c \in \{0, 1\}} p_c^2$$

4. Post-Pruning (Bonus Optimasi DTL +2 Poin - Mitchell, 1997; Breiman et al., 1984):
   - Reduced Error Pruning (REP) (Mitchell, 1997, Bab 3): Menelusuri simpul internal secara
     bottom-up (post-order) dan memangkas (prune) sub-pohon menjadi simpul daun (leaf) apabila
     akurasi pada himpunan validasi tidak menurun:
     $$\text{Acc}(T_{\text{pruned}}, D_{\text{val}}) \ge \text{Acc}(T_{\text{current}}, D_{\text{val}})$$
   - Cost-Complexity Pruning (CCP) (Breiman et al., 1984): Meminimalkan fungsi biaya-kompleksitas
     $$R_\alpha(T) = R(T) + \alpha |L(T)|$$

Referensi Ilmiah (Format IEEE):
[1] J. R. Quinlan, "Induction of decision trees," Machine Learning, vol. 1, no. 1, pp. 81–106, 1986.
[2] J. R. Quinlan, C4.5: Programs for Machine Learning. San Mateo, CA, USA: Morgan Kaufmann, 1993.
[3] T. M. Mitchell, Machine Learning. New York, NY, USA: McGraw-Hill, 1997, ch. 3, pp. 52–80.
[4] L. Breiman, J. Friedman, C. J. Stone, and R. A. Olshen, Classification and Regression Trees.
    Belmont, CA, USA: Wadsworth International Group, 1984.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from .base import BaseModel
from .utils import RANDOM_SEED, accuracy_score


def _is_missing_array(arr: np.ndarray) -> np.ndarray:
    """
    Mendeteksi elemen missing (NaN, None, atau string kosong/'nan'/'null')
    secara vektorisasi maupun objek campuran.
    """
    arr = np.asarray(arr)
    if arr.dtype.kind == "f":
        return np.isnan(arr)
    if arr.dtype.kind in ("i", "u", "b"):
        return np.zeros(arr.shape, dtype=bool)

    # Array bertipe object atau string
    flat = arr.ravel()
    mask = np.zeros(flat.shape, dtype=bool)
    for idx, val in enumerate(flat):
        if val is None:
            mask[idx] = True
        elif isinstance(val, (float, np.floating)):
            if np.isnan(val):
                mask[idx] = True
        elif isinstance(val, str):
            if val.strip().lower() in ("", "nan", "none", "null", "na"):
                mask[idx] = True
        elif val != val:  # Menangani tipe NaN khusus lainnya
            mask[idx] = True
    return mask.reshape(arr.shape)


def _is_single_missing(val: Any) -> bool:
    """Mendeteksi apakah satu nilai skalar merupakan missing value (NaN/None)."""
    if val is None:
        return True
    if isinstance(val, (float, np.floating)):
        return bool(np.isnan(val))
    if isinstance(val, str):
        return val.strip().lower() in ("", "nan", "none", "null", "na")
    return bool(val != val)


class TreeNode:
    """
    Representasi simpul (node) pada pohon keputusan.

    Atribut:
        feature_idx (int | None): Indeks fitur yang digunakan untuk partisi pada simpul ini.
        threshold (float | Any | None): Nilai ambang batas untuk partisi kontinu (x <= T)
            atau kategori acuan untuk partisi kategorikal biner CART (x == T).
        is_categorical (bool): True jika partisi dilakukan pada fitur kategorikal.
        children (dict): Pemetaan {nilai_kategori: TreeNode} untuk partisi multi-cabang (ID3/C4.5).
        left (TreeNode | None): Cabang anak kiri (x <= threshold atau x == kategori pada CART).
        right (TreeNode | None): Cabang anak kanan (x > threshold atau x != kategori pada CART).
        value (int): Label kelas mayoritas (0 atau 1) pada simpul ini.
        class_distribution (np.ndarray): Distribusi probabilitas posterior [P(y=0), P(y=1)].
        n_samples (int): Jumlah sampel (dibulatkan dari bobot sampel) yang mencapai simpul ini.
        weighted_n_samples (float): Total bobot sampel eksak pada simpul ini.
        impurity (float): Nilai ketidakmurnian (Entropy atau Gini) pada simpul ini.
        gain (float): Nilai Information Gain / Gain Ratio / Gini Gain dari partisi simpul ini.
        branch_weights (dict): Proporsi bobot cabang {kunci_cabang: r_v} untuk penanganan NaN C4.5.
        is_binary_categorical (bool): True jika simpul kategorikal menggunakan partisi biner (CART).
        depth (int): Kedalaman simpul dari root (root = 0).
    """

    def __init__(
        self,
        feature_idx: Optional[int] = None,
        threshold: Optional[Any] = None,
        is_categorical: bool = False,
        children: Optional[Dict[Any, "TreeNode"]] = None,
        left: Optional["TreeNode"] = None,
        right: Optional["TreeNode"] = None,
        value: Optional[int] = None,
        class_distribution: Optional[np.ndarray] = None,
        n_samples: int = 0,
        weighted_n_samples: float = 0.0,
        impurity: float = 0.0,
        gain: float = 0.0,
        branch_weights: Optional[Dict[Any, float]] = None,
        is_binary_categorical: bool = False,
        depth: int = 0,
    ):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.is_categorical = is_categorical
        self.children = children if children is not None else {}
        self.left = left
        self.right = right
        self.value = value
        self.class_distribution = (
            np.asarray(class_distribution, dtype=float)
            if class_distribution is not None
            else np.array([0.5, 0.5], dtype=float)
        )
        self.n_samples = int(n_samples)
        self.weighted_n_samples = float(weighted_n_samples if weighted_n_samples > 0 else n_samples)
        self.impurity = float(impurity)
        self.gain = float(gain)
        self.branch_weights = branch_weights if branch_weights is not None else {}
        self.is_binary_categorical = bool(is_binary_categorical)
        self.depth = int(depth)

    @property
    def is_leaf(self) -> bool:
        """Mengembalikan True apabila simpul merupakan simpul daun (leaf node)."""
        return self.left is None and self.right is None and len(self.children) == 0


class DecisionTree(BaseModel):
    """
    Implementasi Decision Tree Classifier (ID3 / C4.5 / CART) from Scratch.

    Fitur Utama:
    1. Mendukung varian 'c45' (Gain Ratio - default), 'id3' (Information Gain), dan 'cart' (Gini).
    2. Menangani atribut kontinu menggunakan titik tengah (midpoint threshold):
       T_i = (v_i + v_{i+1}) / 2.
    3. Menangani atribut kategorikal (multi-way split pada ID3/C4.5, binary split pada CART).
    4. Menangani missing values (NaN/Null) secara mandiri sesuai metode Quinlan (C4.5):
       - Penskalaan Gain dengan proporsi sampel non-null |S_known| / |S|.
       - Distribusi sampel missing ke seluruh cabang anak menggunakan fractional weight r_v.
       - Agregasi probabilitas berbobot pecahan saat inferensi (`predict_proba`) jika uji mengandung NaN.
    5. Mendukung Post-Pruning (Reduced Error Pruning & Cost-Complexity Pruning) untuk Bonus Optimasi (+2 poin).

    Parameter:
        criterion (str): 'c45' (Gain Ratio - direkomendasikan), 'id3' (Entropy), atau 'cart' (Gini).
        max_depth (int | None): Kedalaman maksimum pohon (None = tidak dibatasi).
        min_samples_split (int): Jumlah minimum sampel untuk memecah simpul (default: 2).
        min_samples_leaf (int): Jumlah minimum bobot sampel pada setiap cabang anak (default: 1).
        min_impurity_decrease (float): Ambang batas minimum penurunan impurity/gain untuk memecah simpul.
        use_pruning (bool): Aktifkan post-pruning otomatis untuk Bonus Optimasi (+2 poin).
        pruning_method (str): Metode post-pruning: 'rep' (Reduced Error Pruning - Mitchell 1997)
            atau 'ccp' (Cost-Complexity Pruning - Breiman et al. 1984). Default: 'rep'.
        ccp_alpha (float): Parameter kompleksitas alpha untuk Cost-Complexity Pruning (default: 0.0).
        val_split_ratio (float): Proporsi data latih yang disisihkan untuk validasi pruning
            jika `use_pruning=True` tetapi `X_val` tidak diberikan secara eksplisit pada `fit`.
            Set ke 0.0 jika ingin melatih pada 100% data dan memanggil `prune(X_val, y_val)` manual.
        categorical_features (list | None): Daftar indeks atau mask boolean fitur kategorikal.
            Jika None, tipe fitur dideteksi otomatis dari data masukan X.
        max_categorical_cardinality (int): Batas maksimum nilai unik integer agar dideteksi
            sebagai fitur kategorikal apabila kolom bertipe objek/integer (default: 12).
        random_state (int): Seed pengacakan untuk pemisahan validasi internal (default: 42).
    """

    def __init__(
        self,
        criterion: str = "c45",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 1e-7,
        use_pruning: bool = False,
        pruning_method: str = "rep",
        ccp_alpha: float = 0.01,
        val_split_ratio: float = 0.0,
        categorical_features: Optional[List[Union[int, bool]]] = None,
        max_categorical_cardinality: int = 12,
        random_state: int = RANDOM_SEED,
        **kwargs,
    ):
        super().__init__(**kwargs)
        criterion_clean = criterion.lower().strip()
        if criterion_clean not in ("c45", "c4.5", "gain_ratio", "id3", "entropy", "cart", "gini"):
            raise ValueError(
                f"Criterion '{criterion}' tidak dikenali. Pilih 'c45', 'id3', atau 'cart'."
            )
        if criterion_clean in ("c45", "c4.5", "gain_ratio"):
            self.criterion = "c45"
        elif criterion_clean in ("id3", "entropy"):
            self.criterion = "id3"
        else:
            self.criterion = "cart"

        self.max_depth = max_depth
        self.min_samples_split = max(2, int(min_samples_split))
        self.min_samples_leaf = max(1, int(min_samples_leaf))
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.use_pruning = bool(use_pruning)
        self.pruning_method = pruning_method.lower().strip()
        self.ccp_alpha = float(ccp_alpha)
        self.val_split_ratio = float(val_split_ratio)
        self.categorical_features = categorical_features
        self.max_categorical_cardinality = int(max_categorical_cardinality)
        self.random_state = int(random_state)

        self.root_: Optional[TreeNode] = None
        self._unpruned_root_: Optional[TreeNode] = None
        self.n_features_: int = 0
        self.feature_types_: List[str] = []  # 'categorical' atau 'continuous'
        self.feature_names_: Optional[List[str]] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.pruning_history_: List[Dict[str, Any]] = []

    # =========================================================================
    # 1. DETEKSI TIPE FITUR & PRA-PEMROSESAN INTERNAL
    # =========================================================================

    def _prepare_input_X(self, X: Any, is_training: bool = False) -> np.ndarray:
        """Mengonversi DataFrame/array masukan ke np.ndarray 2D tanpa kehilangan tipe objek/NaN."""
        if hasattr(X, "columns") and is_training and self.feature_names_ is None:
            self.feature_names_ = [str(col) for col in X.columns]
        if hasattr(X, "to_numpy"):
            X_arr = X.to_numpy()
        else:
            X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        return X_arr

    def _detect_feature_types(self, X: np.ndarray) -> List[str]:
        """
        Mendeteksi apakah setiap kolom fitur merupakan 'categorical' atau 'continuous'.
        - Jika `categorical_features` diberikan oleh pengguna, gunakan spesifikasi tersebut.
        - Jika tidak:
          * Kolom dengan nilai string/boolean diklasifikasikan sebagai 'categorical'.
          * Kolom numerik float murni diklasifikasikan sebagai 'continuous'.
          * Kolom bertipe integer/objek yang seluruh nilai non-null-nya bilangan bulat
            dengan jumlah kategori unik <= `max_categorical_cardinality` diklasifikasikan
            sebagai 'categorical'.
        """
        n_features = X.shape[1]
        if self.categorical_features is not None:
            if len(self.categorical_features) == n_features and all(
                isinstance(v, (bool, np.bool_)) for v in self.categorical_features
            ):
                return [
                    "categorical" if is_cat else "continuous"
                    for is_cat in self.categorical_features
                ]
            cat_set = set(int(idx) for idx in self.categorical_features)
            return [
                "categorical" if j in cat_set else "continuous"
                for j in range(n_features)
            ]

        feature_types = []
        for j in range(n_features):
            col = X[:, j]
            miss_mask = _is_missing_array(col)
            known = col[~miss_mask]
            if len(known) == 0:
                feature_types.append("continuous")
                continue

            if col.dtype.kind in ("U", "S", "b"):
                feature_types.append("categorical")
                continue

            if col.dtype.kind == "f":
                feature_types.append("continuous")
                continue

            if col.dtype.kind in ("i", "u"):
                n_unique = len(np.unique(known))
                if n_unique <= self.max_categorical_cardinality:
                    feature_types.append("categorical")
                else:
                    feature_types.append("continuous")
                continue

            # Dtype object: periksa apakah ada nilai string/bool yang bukan angka
            is_cat = False
            numeric_vals = []
            for v in known:
                if isinstance(v, (bool, np.bool_)):
                    is_cat = True
                    break
                if isinstance(v, (int, float, np.number)):
                    numeric_vals.append(float(v))
                elif isinstance(v, str):
                    try:
                        numeric_vals.append(float(v))
                    except ValueError:
                        is_cat = True
                        break
                else:
                    is_cat = True
                    break

            if is_cat:
                feature_types.append("categorical")
            else:
                num_arr = np.asarray(numeric_vals, dtype=float)
                all_int = np.all(np.floor(num_arr) == num_arr)
                n_unique = len(np.unique(num_arr))
                if all_int and n_unique <= self.max_categorical_cardinality:
                    feature_types.append("categorical")
                else:
                    feature_types.append("continuous")

        return feature_types

    # =========================================================================
    # 2. PERHITUNGAN IMPURITY (ENTROPY & GINI) BERBOBOT
    # =========================================================================

    def _compute_impurity_from_counts(self, w0: float, w1: float) -> float:
        """
        Menghitung nilai Impurity (Shannon Entropy untuk ID3/C4.5 atau Gini untuk CART)
        dari bobot kelas negatif (w0) dan positif (w1).
        """
        total = w0 + w1
        if total <= 1e-12:
            return 0.0
        p0 = w0 / total
        p1 = w1 / total

        if self.criterion in ("id3", "c45"):
            entropy = 0.0
            if p0 > 1e-12:
                entropy -= p0 * np.log2(p0)
            if p1 > 1e-12:
                entropy -= p1 * np.log2(p1)
            return float(entropy)
        else:
            # CART: Gini Impurity = 1 - (p0^2 + p1^2) = 2 * p0 * p1
            return float(1.0 - (p0 * p0 + p1 * p1))

    def _compute_impurity_vectorized(
        self, w0_arr: np.ndarray, w1_arr: np.ndarray
    ) -> np.ndarray:
        """
        Menghitung Impurity (Entropy atau Gini) secara vektorisasi untuk banyak kandidat split sekaligus.
        """
        total = w0_arr + w1_arr
        safe_total = np.where(total > 1e-12, total, 1.0)
        p0 = np.where(total > 1e-12, w0_arr / safe_total, 0.0)
        p1 = np.where(total > 1e-12, w1_arr / safe_total, 0.0)

        if self.criterion in ("id3", "c45"):
            with np.errstate(divide="ignore", invalid="ignore"):
                term0 = np.where(p0 > 1e-12, -p0 * np.log2(np.maximum(p0, 1e-12)), 0.0)
                term1 = np.where(p1 > 1e-12, -p1 * np.log2(np.maximum(p1, 1e-12)), 0.0)
            return term0 + term1
        else:
            return 1.0 - (p0 * p0 + p1 * p1)

    # =========================================================================
    # 3. EVALUASI KANDIDAT PARTISI (NUMERIK, KATEGORIKAL, & MISSING VALUES C4.5)
    # =========================================================================

    def _evaluate_continuous_feature(
        self,
        col_known: np.ndarray,
        y_known: np.ndarray,
        w_known: np.ndarray,
        total_weight_all: float,
        weight_known: float,
        weight_missing: float,
        impurity_known: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Mengevaluasi seluruh titik batas tengah (midpoint threshold):
            T_i = (v_i + v_{i+1}) / 2
        pada atribut numerik kontinu menggunakan operasi kumulatif NumPy ter-vektorisasi.
        Mengacu pada Quinlan (1993) C4.5 untuk penskalaan Gain pada data dengan missing values.
        """
        vals = col_known.astype(float)
        unique_vals, inv_idx = np.unique(vals, return_inverse=True)
        m = len(unique_vals)
        if m <= 1:
            return None

        # Akumulasi bobot kelas 0 dan kelas 1 pada setiap nilai unik v_1 < v_2 < ... < v_m
        w0_per_unique = np.bincount(inv_idx, weights=w_known * (y_known == 0), minlength=m)
        w1_per_unique = np.bincount(inv_idx, weights=w_known * (y_known == 1), minlength=m)

        # Kumulatif untuk cabang kiri (x <= T_i) untuk i = 0 .. m-2
        w0_left = np.cumsum(w0_per_unique)[:-1]
        w1_left = np.cumsum(w1_per_unique)[:-1]
        w_left = w0_left + w1_left

        # Cabang kanan (x > T_i)
        total_w0 = w0_per_unique.sum()
        total_w1 = w1_per_unique.sum()
        w0_right = total_w0 - w0_left
        w1_right = total_w1 - w1_left
        w_right = w0_right + w1_right

        # Filter berdasarkan min_samples_leaf
        valid_mask = (w_left >= self.min_samples_leaf - 1e-9) & (
            w_right >= self.min_samples_leaf - 1e-9
        )
        if not np.any(valid_mask):
            return None

        # Hitung impurity cabang kiri dan kanan secara vektorisasi
        imp_left = self._compute_impurity_vectorized(w0_left, w1_left)
        imp_right = self._compute_impurity_vectorized(w0_right, w1_right)

        # Rumus Quinlan (C4.5):
        # Gain(S, A) = (|S_known| / |S|) * (Impurity(S_known) - sum_v (|S_v| / |S_known|) * Impurity(S_v))
        known_ratio = weight_known / total_weight_all
        weighted_child_imp = (w_left * imp_left + w_right * imp_right) / weight_known
        gains = known_ratio * (impurity_known - weighted_child_imp)

        # Terapkan filter valid_mask dan gain positif
        gains = np.where(valid_mask & (gains > self.min_impurity_decrease), gains, -1.0)
        if np.max(gains) <= 0.0:
            return None

        # Sesuai Quinlan (1993, Bab 2.4 - Tests on Continuous Attributes):
        # Ambang batas optimal T^* pada atribut kontinu dipilih yang memaksimalkan Information Gain
        # (agar tidak bias ke titik ekstrem di ujung rentang yang memiliki SplitInfo mendekati 0).
        best_idx = int(np.argmax(gains))
        best_gain = float(gains[best_idx])
        if best_gain <= self.min_impurity_decrease:
            return None

        if self.criterion == "c45":
            # Hitung SplitInfo(S, A, T^*) dan GainRatio(S, A, T^*) pada titik batas terbaik T^*
            p_left = float(w_left[best_idx] / total_weight_all)
            p_right = float(w_right[best_idx] / total_weight_all)
            split_info = 0.0
            if p_left > 1e-12:
                split_info -= p_left * float(np.log2(p_left))
            if p_right > 1e-12:
                split_info -= p_right * float(np.log2(p_right))
            if weight_missing > 1e-12:
                p_miss = float(weight_missing / total_weight_all)
                split_info -= p_miss * float(np.log2(p_miss))

            if split_info <= 1e-12:
                return None
            best_score = float(best_gain / split_info)
        else:
            best_score = best_gain

        # Titik batas tengah (midpoint threshold): T_i = (v_i + v_{i+1}) / 2
        threshold = float(0.5 * (unique_vals[best_idx] + unique_vals[best_idx + 1]))
        r_left = float(w_left[best_idx] / weight_known)
        r_right = float(w_right[best_idx] / weight_known)

        return {
            "score": best_score,
            "gain": best_gain,
            "threshold": threshold,
            "is_categorical": False,
            "is_binary_categorical": False,
            "branch_weights": {"left": r_left, "right": r_right},
        }

    def _evaluate_categorical_feature(
        self,
        col_known: np.ndarray,
        y_known: np.ndarray,
        w_known: np.ndarray,
        total_weight_all: float,
        weight_known: float,
        weight_missing: float,
        impurity_known: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Mengevaluasi partisi pada atribut kategorikal:
        - Untuk ID3 dan C4.5: Partisi multi-cabang (multi-way split) untuk setiap kategori unik.
        - Untuk CART: Partisi biner (one-vs-rest: x == c vs x != c) yang memaksimalkan Gini Gain.
        """
        # Ambil kategori unik dengan tetap menjaga tipe asli (str/int)
        str_vals = np.asarray([str(v) for v in col_known])
        unique_strs, first_indices, inv_idx = np.unique(
            str_vals, return_index=True, return_inverse=True
        )
        k = len(unique_strs)
        if k <= 1:
            return None

        orig_categories = [col_known[idx] for idx in first_indices]
        w0_per_cat = np.bincount(inv_idx, weights=w_known * (y_known == 0), minlength=k)
        w1_per_cat = np.bincount(inv_idx, weights=w_known * (y_known == 1), minlength=k)
        w_per_cat = w0_per_cat + w1_per_cat

        known_ratio = weight_known / total_weight_all

        if self.criterion == "cart":
            # CART: Evaluasi partisi biner x == c vs x != c untuk setiap kategori c
            total_w0 = w0_per_cat.sum()
            total_w1 = w1_per_cat.sum()
            w0_right = total_w0 - w0_per_cat
            w1_right = total_w1 - w1_per_cat
            w_right = w0_right + w1_right

            valid_mask = (w_per_cat >= self.min_samples_leaf - 1e-9) & (
                w_right >= self.min_samples_leaf - 1e-9
            )
            if not np.any(valid_mask):
                return None

            imp_left = self._compute_impurity_vectorized(w0_per_cat, w1_per_cat)
            imp_right = self._compute_impurity_vectorized(w0_right, w1_right)
            weighted_child_imp = (w_per_cat * imp_left + w_right * imp_right) / weight_known
            gains = known_ratio * (impurity_known - weighted_child_imp)
            gains = np.where(valid_mask & (gains > self.min_impurity_decrease), gains, -1.0)

            best_idx = int(np.argmax(gains))
            best_gain = float(gains[best_idx])
            if best_gain <= self.min_impurity_decrease:
                return None

            r_left = float(w_per_cat[best_idx] / weight_known)
            r_right = float(w_right[best_idx] / weight_known)
            return {
                "score": best_gain,
                "gain": best_gain,
                "threshold": orig_categories[best_idx],
                "is_categorical": True,
                "is_binary_categorical": True,
                "branch_weights": {"left": r_left, "right": r_right},
            }

        # ID3 & C4.5: Multi-way categorical split
        # Pastikan minimal ada 2 cabang yang memenuhi min_samples_leaf
        if np.sum(w_per_cat >= self.min_samples_leaf - 1e-9) < 2:
            return None

        imp_per_cat = self._compute_impurity_vectorized(w0_per_cat, w1_per_cat)
        weighted_child_imp = float(np.sum((w_per_cat / weight_known) * imp_per_cat))
        gain = float(known_ratio * (impurity_known - weighted_child_imp))

        if gain <= self.min_impurity_decrease:
            return None

        if self.criterion == "c45":
            p_cats = w_per_cat / total_weight_all
            with np.errstate(divide="ignore", invalid="ignore"):
                split_info = float(
                    -np.sum(
                        np.where(p_cats > 1e-12, p_cats * np.log2(np.maximum(p_cats, 1e-12)), 0.0)
                    )
                )
            if weight_missing > 1e-12:
                p_miss = weight_missing / total_weight_all
                split_info -= float(p_miss * np.log2(p_miss))

            if split_info <= 1e-12:
                return None
            score = float(gain / split_info)
        else:
            score = gain

        branch_weights = {
            orig_categories[idx]: float(w_per_cat[idx] / weight_known) for idx in range(k)
        }
        return {
            "score": score,
            "gain": gain,
            "threshold": None,
            "is_categorical": True,
            "is_binary_categorical": False,
            "branch_weights": branch_weights,
            "categories": orig_categories,
        }

    # =========================================================================
    # 4. KONSTRUKSI POHON KEPUTUSAN REKURSIF & DISTRIBUSI PECAHAN SAMPEL NAN
    # =========================================================================

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        depth: int,
        used_cat_features: Tuple[int, ...] = (),
    ) -> TreeNode:
        """
        Membangun pohon keputusan secara rekursif dari atas ke bawah (top-down induction).
        Menyimpan distribusi kelas pada setiap simpul internal maupun daun untuk mendukung
        prediksi probabilitas dan post-pruning.
        """
        w0 = float(np.sum(weights[y == 0]))
        w1 = float(np.sum(weights[y == 1]))
        total_weight = w0 + w1
        n_samples = int(round(total_weight))

        if total_weight > 1e-12:
            class_dist = np.array([w0 / total_weight, w1 / total_weight], dtype=float)
        else:
            class_dist = np.array([0.5, 0.5], dtype=float)

        majority_class = int(1 if class_dist[1] >= 0.5 else 0)
        node_impurity = self._compute_impurity_from_counts(w0, w1)

        node = TreeNode(
            value=majority_class,
            class_distribution=class_dist,
            n_samples=max(1, n_samples),
            weighted_n_samples=total_weight,
            impurity=node_impurity,
            depth=depth,
        )

        # Kriteria penghentian (Stopping Criteria):
        # 1. Simpul sudah murni (semua sampel berasal dari 1 kelas)
        # 2. Mencapai batas kedalaman maksimum (max_depth)
        # 3. Jumlah bobot sampel kurang dari min_samples_split
        if (
            node_impurity <= 1e-12
            or (self.max_depth is not None and depth >= self.max_depth)
            or total_weight < self.min_samples_split
            or len(y) < self.min_samples_split
        ):
            return node

        # Cari atribut dan titik partisi terbaik
        candidates: List[Tuple[int, Dict[str, Any]]] = []

        for j in range(self.n_features_):
            # Pada ID3/C4.5 multi-cabang, fitur kategorikal yang sudah dipakai penuh tidak diulang
            if (
                self.feature_types_[j] == "categorical"
                and self.criterion in ("id3", "c45")
                and j in used_cat_features
            ):
                continue

            col = X[:, j]
            miss_mask = _is_missing_array(col)
            known_mask = ~miss_mask

            if not np.any(known_mask):
                continue

            w_known = weights[known_mask]
            weight_known = float(np.sum(w_known))
            if weight_known < self.min_samples_split:
                continue

            weight_missing = total_weight - weight_known
            y_known = y[known_mask]
            w0_known = float(np.sum(w_known[y_known == 0]))
            w1_known = float(np.sum(w_known[y_known == 1]))
            impurity_known = self._compute_impurity_from_counts(w0_known, w1_known)

            if impurity_known <= 1e-12:
                continue

            col_known = col[known_mask]

            if self.feature_types_[j] == "continuous":
                candidate = self._evaluate_continuous_feature(
                    col_known=col_known,
                    y_known=y_known,
                    w_known=w_known,
                    total_weight_all=total_weight,
                    weight_known=weight_known,
                    weight_missing=weight_missing,
                    impurity_known=impurity_known,
                )
            else:
                candidate = self._evaluate_categorical_feature(
                    col_known=col_known,
                    y_known=y_known,
                    w_known=w_known,
                    total_weight_all=total_weight,
                    weight_known=weight_known,
                    weight_missing=weight_missing,
                    impurity_known=impurity_known,
                )

            if candidate is not None:
                candidates.append((j, candidate))

        if not candidates:
            return node

        if self.criterion == "c45":
            # Aturan Quinlan (1993) C4.5: Pilih atribut dengan Gain Ratio tertinggi
            # di antara kandidat yang memiliki Information Gain >= rata-rata Gain seluruh kandidat
            avg_gain = float(np.mean([cand["gain"] for _, cand in candidates]))
            above_avg = [
                (j, cand) for j, cand in candidates if cand["gain"] >= avg_gain - 1e-12
            ]
            pool = above_avg if above_avg else candidates
            best_feature_idx, best_split = max(pool, key=lambda item: item[1]["score"])
        else:
            best_feature_idx, best_split = max(candidates, key=lambda item: item[1]["score"])

        # Tetapkan metadata partisi terbaik pada simpul saat ini
        node.feature_idx = best_feature_idx
        node.threshold = best_split["threshold"]
        node.is_categorical = best_split["is_categorical"]
        node.is_binary_categorical = best_split["is_binary_categorical"]
        node.gain = best_split["gain"]
        node.branch_weights = best_split["branch_weights"]

        col_best = X[:, best_feature_idx]
        miss_mask = _is_missing_array(col_best)
        known_mask = ~miss_mask

        # Ambang batas bobot minimum agar pecahan sampel NaN mikroskopis tidak menggandakan ukuran array
        min_frac_weight = 1e-3

        if not node.is_categorical or node.is_binary_categorical:
            # Partisi biner (Continuous <= threshold ATAU CART Categorical == threshold)
            if not node.is_categorical:
                known_vals = col_best[known_mask].astype(float)
                go_left_known = known_vals <= node.threshold
            else:
                known_vals = col_best[known_mask]
                go_left_known = np.asarray([v == node.threshold for v in known_vals], dtype=bool)

            known_indices = np.where(known_mask)[0]
            miss_indices = np.where(miss_mask)[0]

            left_known_idx = known_indices[go_left_known]
            right_known_idx = known_indices[~go_left_known]

            r_left = node.branch_weights["left"]
            r_right = node.branch_weights["right"]

            # Distribusikan sampel missing (NaN) ke cabang kiri & kanan dengan bobot pecahan r_v
            if len(miss_indices) > 0:
                miss_w_left = weights[miss_indices] * r_left
                keep_miss_left = miss_w_left >= min_frac_weight
                left_idx = np.concatenate([left_known_idx, miss_indices[keep_miss_left]])
                left_w = np.concatenate([weights[left_known_idx], miss_w_left[keep_miss_left]])

                miss_w_right = weights[miss_indices] * r_right
                keep_miss_right = miss_w_right >= min_frac_weight
                right_idx = np.concatenate([right_known_idx, miss_indices[keep_miss_right]])
                right_w = np.concatenate([weights[right_known_idx], miss_w_right[keep_miss_right]])
            else:
                left_idx = left_known_idx
                left_w = weights[left_known_idx]
                right_idx = right_known_idx
                right_w = weights[right_known_idx]

            if len(left_idx) == 0 or len(right_idx) == 0:
                node.feature_idx = None
                node.threshold = None
                node.branch_weights = {}
                return node

            node.left = self._build_tree(
                X[left_idx], y[left_idx], left_w, depth + 1, used_cat_features
            )
            node.right = self._build_tree(
                X[right_idx], y[right_idx], right_w, depth + 1, used_cat_features
            )
        else:
            # Partisi multi-cabang untuk fitur kategorikal pada ID3 / C4.5
            known_indices = np.where(known_mask)[0]
            miss_indices = np.where(miss_mask)[0]
            known_vals = col_best[known_mask]
            new_used_cats = used_cat_features + (best_feature_idx,)

            for cat_val, r_cat in node.branch_weights.items():
                cat_mask_known = np.asarray([v == cat_val for v in known_vals], dtype=bool)
                cat_known_idx = known_indices[cat_mask_known]

                if len(miss_indices) > 0:
                    miss_w_cat = weights[miss_indices] * r_cat
                    keep_miss = miss_w_cat >= min_frac_weight
                    child_idx = np.concatenate([cat_known_idx, miss_indices[keep_miss]])
                    child_w = np.concatenate([weights[cat_known_idx], miss_w_cat[keep_miss]])
                else:
                    child_idx = cat_known_idx
                    child_w = weights[cat_known_idx]

                if len(child_idx) == 0:
                    node.children[cat_val] = TreeNode(
                        value=majority_class,
                        class_distribution=class_dist.copy(),
                        n_samples=0,
                        weighted_n_samples=0.0,
                        impurity=0.0,
                        depth=depth + 1,
                    )
                else:
                    node.children[cat_val] = self._build_tree(
                        X[child_idx], y[child_idx], child_w, depth + 1, new_used_cats
                    )

        return node

    # =========================================================================
    # 5. ANTARMUKA UTAMA: FIT, PREDICT_PROBA, & PREDICT
    # =========================================================================

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "DecisionTree":
        """
        Melatih pohon keputusan pada data X (mendukung campuran numerik, kategorikal, dan NaN).
        Jika `use_pruning=True`, jalankan post-pruning (menggunakan `X_val, y_val` jika diberikan,
        atau memisahkan himpunan validasi internal secara otomatis sesuai `val_split_ratio`).

        Parameter:
            X: Array fitur berukuran (n_samples, n_features)
            y: Array label biner berukuran (n_samples,) dengan nilai 0 atau 1
            X_val: (Opsional) Array fitur validasi untuk post-pruning
            y_val: (Opsional) Array label validasi untuk post-pruning

        Mengembalikan:
            self
        """
        X_arr = self._prepare_input_X(X, is_training=True)
        y_arr = np.asarray(y).astype(int).ravel()

        if len(X_arr) != len(y_arr):
            raise ValueError(
                f"Jumlah baris X ({len(X_arr)}) tidak sama dengan panjang y ({len(y_arr)})."
            )

        self.classes_ = np.unique(y_arr)
        self.n_features_ = X_arr.shape[1]
        if self.feature_names_ is None or len(self.feature_names_) != self.n_features_:
            self.feature_names_ = [f"Feature_{j}" for j in range(self.n_features_)]

        self.feature_types_ = self._detect_feature_types(X_arr)

        # Jika use_pruning=True, metode REP dipilih, dan X_val tidak diberikan secara eksplisit,
        # pisahkan sebagian kecil data latih secara stratified untuk validasi post-pruning
        X_train_use = X_arr
        y_train_use = y_arr
        X_val_use = self._prepare_input_X(X_val) if X_val is not None else None
        y_val_use = np.asarray(y_val).astype(int).ravel() if y_val is not None else None

        if (
            self.use_pruning
            and self.pruning_method == "rep"
            and X_val_use is None
            and 0.0 < self.val_split_ratio < 0.5
            and len(X_arr) >= 20
        ):
            rng = np.random.default_rng(self.random_state)
            idx_0 = np.where(y_arr == 0)[0]
            idx_1 = np.where(y_arr == 1)[0]
            rng.shuffle(idx_0)
            rng.shuffle(idx_1)

            n_val_0 = max(1, int(round(len(idx_0) * self.val_split_ratio)))
            n_val_1 = max(1, int(round(len(idx_1) * self.val_split_ratio)))

            val_idx = np.concatenate([idx_0[:n_val_0], idx_1[:n_val_1]])
            train_idx = np.concatenate([idx_0[n_val_0:], idx_1[n_val_1:]])

            X_train_use = X_arr[train_idx]
            y_train_use = y_arr[train_idx]
            X_val_use = X_arr[val_idx]
            y_val_use = y_arr[val_idx]

        init_weights = np.ones(len(y_train_use), dtype=float)
        self.root_ = self._build_tree(
            X=X_train_use,
            y=y_train_use,
            weights=init_weights,
            depth=0,
            used_cat_features=(),
        )
        self._unpruned_root_ = copy.deepcopy(self.root_)
        self.is_fitted = True

        # Jalankan Post-Pruning jika diaktifkan
        if self.use_pruning:
            if X_val_use is not None and y_val_use is not None:
                self.prune(X_val_use, y_val_use)
            elif self.pruning_method == "ccp" or self.ccp_alpha > 0.0:
                self._prune_cost_complexity(alpha=max(self.ccp_alpha, 0.005))

        self.feature_importances_ = self.compute_feature_importances()
        return self

    def _predict_row_proba(self, row: np.ndarray, node: TreeNode) -> np.ndarray:
        """
        Menghitung probabilitas posterior [P(y=0|x), P(y=1|x)] untuk satu sampel x.
        Mendukung agregasi bobot pecahan (fractional weight) ala Quinlan (C4.5) apabila
        nilai fitur pada simpul keputusan bernilai NaN/Null atau kategori belum pernah ditemui.
        """
        if node.is_leaf or node.feature_idx is None:
            return node.class_distribution

        val = row[node.feature_idx]

        # 1. Penanganan Missing Value (NaN/Null) saat Inferensi (Quinlan C4.5):
        #    Telusuri seluruh cabang anak dan agregasikan probabilitas dengan bobot r_v
        if _is_single_missing(val):
            if not node.is_categorical or node.is_binary_categorical:
                r_left = node.branch_weights.get("left", 0.5)
                r_right = node.branch_weights.get("right", 0.5)
                p_left = (
                    self._predict_row_proba(row, node.left)
                    if node.left is not None
                    else node.class_distribution
                )
                p_right = (
                    self._predict_row_proba(row, node.right)
                    if node.right is not None
                    else node.class_distribution
                )
                return r_left * p_left + r_right * p_right
            else:
                agg_prob = np.zeros(2, dtype=float)
                total_r = 0.0
                for cat_val, child_node in node.children.items():
                    r_cat = node.branch_weights.get(cat_val, 0.0)
                    agg_prob += r_cat * self._predict_row_proba(row, child_node)
                    total_r += r_cat
                if total_r > 1e-12:
                    return agg_prob / total_r
                return node.class_distribution

        # 2. Nilai fitur diketahui (Non-Null)
        if not node.is_categorical:
            try:
                num_val = float(val)
            except (ValueError, TypeError):
                # Jika gagal dikonversi ke float, perlakukan seperti nilai tidak diketahui
                r_left = node.branch_weights.get("left", 0.5)
                r_right = node.branch_weights.get("right", 0.5)
                return r_left * self._predict_row_proba(
                    row, node.left
                ) + r_right * self._predict_row_proba(row, node.right)

            if num_val <= node.threshold:
                return (
                    self._predict_row_proba(row, node.left)
                    if node.left is not None
                    else node.class_distribution
                )
            else:
                return (
                    self._predict_row_proba(row, node.right)
                    if node.right is not None
                    else node.class_distribution
                )

        if node.is_binary_categorical:
            if val == node.threshold or str(val) == str(node.threshold):
                return (
                    self._predict_row_proba(row, node.left)
                    if node.left is not None
                    else node.class_distribution
                )
            else:
                return (
                    self._predict_row_proba(row, node.right)
                    if node.right is not None
                    else node.class_distribution
                )

        # Multi-way categorical split (ID3 / C4.5)
        if val in node.children:
            return self._predict_row_proba(row, node.children[val])

        # Fallback pencocokan representasi string (misal int vs str)
        val_str = str(val)
        for cat_key, child_node in node.children.items():
            if str(cat_key) == val_str:
                return self._predict_row_proba(row, child_node)

        # Jika kategori pada data uji belum pernah muncul saat pelatihan (unseen category),
        # agregasikan seluruh cabang menggunakan distribusi bobot cabang C4.5
        agg_prob = np.zeros(2, dtype=float)
        total_r = 0.0
        for cat_val, child_node in node.children.items():
            r_cat = node.branch_weights.get(cat_val, 0.0)
            agg_prob += r_cat * self._predict_row_proba(row, child_node)
            total_r += r_cat
        if total_r > 1e-12:
            return agg_prob / total_r
        return node.class_distribution

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Menghasilkan probabilitas posterior [P(y=0|X), P(y=1|X)] berukuran (n_samples, 2).
        Mendukung agregasi bobot pecahan (fractional weight) apabila sampel memiliki nilai NaN.
        """
        if not self.is_fitted or self.root_ is None:
            raise RuntimeError("Model DecisionTree belum dilatih. Panggil `fit(X, y)` terlebih dahulu.")

        X_arr = self._prepare_input_X(X, is_training=False)
        n_samples = len(X_arr)
        probs = np.empty((n_samples, 2), dtype=float)

        for i in range(n_samples):
            p = self._predict_row_proba(X_arr[i], self.root_)
            s = p[0] + p[1]
            if s > 1e-12:
                probs[i, 0] = p[0] / s
                probs[i, 1] = p[1] / s
            else:
                probs[i, 0] = 0.5
                probs[i, 1] = 0.5

        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Melakukan prediksi label kelas diskrit (0 atau 1).
        """
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

    # =========================================================================
    # 6. BONUS OPTIMASI DTL (+2 POIN): POST-PRUNING (REP & CCP)
    # =========================================================================

    def prune(self, X_val: np.ndarray, y_val: np.ndarray) -> "DecisionTree":
        """
        Bonus Optimasi DTL (+2 Poin):
        Melakukan pemangkasan pohon keputusan pasca-pelatihan (Post-Pruning) secara bottom-up.

        Mekanisme Reduced Error Pruning (REP) — Tom M. Mitchell (1997), Machine Learning, Bab 3:
        1. Telusuri seluruh simpul internal pohon secara post-order (anak dipangkas lebih dulu
           sebelum induknya dievaluasi).
        2. Untuk setiap simpul internal $t$, uji penggantian sub-pohon $T_t$ dengan simpul daun
           yang memprediksi kelas mayoritas sampel latih di $t$.
        3. Hitung akurasi pada himpunan validasi $D_{\\text{val}} = (X_{\\text{val}}, y_{\\text{val}})$.
        4. Apabila akurasi validasi setelah pemangkasan tidak menurun:
               $\\text{Acc}(T_{\\text{pruned}}, D_{\\text{val}}) \\ge \\text{Acc}(T_{\\text{current}}, D_{\\text{val}})$
           maka pemangkasan dipertahankan secara permanen (prinsip Occam's Razor).

        Parameter:
            X_val: np.ndarray berukuran (n_val, n_features)
            y_val: np.ndarray berukuran (n_val,)

        Mengembalikan:
            self
        """
        if not self.is_fitted or self.root_ is None:
            raise RuntimeError("Model DecisionTree belum dilatih. Panggil `fit` sebelum `prune`.")

        X_val_arr = self._prepare_input_X(X_val, is_training=False)
        y_val_arr = np.asarray(y_val).astype(int).ravel()

        self.pruning_history_ = []

        if self.pruning_method == "ccp" and self.ccp_alpha > 0.0:
            self._prune_cost_complexity_with_val(X_val_arr, y_val_arr)
        else:
            self._prune_reduced_error(self.root_, X_val_arr, y_val_arr)

        self.feature_importances_ = self.compute_feature_importances()
        return self

    def _prune_reduced_error(
        self, node: Optional[TreeNode], X_val: np.ndarray, y_val: np.ndarray
    ) -> None:
        """
        Implementasi rekursif bottom-up (post-order) Reduced Error Pruning (Mitchell, 1997).
        """
        if node is None or node.is_leaf:
            return

        # Langkah 1: Telusuri cabang anak terlebih dahulu (Bottom-Up / Post-Order Traversal)
        if node.left is not None:
            self._prune_reduced_error(node.left, X_val, y_val)
        if node.right is not None:
            self._prune_reduced_error(node.right, X_val, y_val)
        if node.children:
            for child in list(node.children.values()):
                self._prune_reduced_error(child, X_val, y_val)

        # Langkah 2: Evaluasi akurasi sebelum simpul `node` dipangkas
        acc_before = float(accuracy_score(y_val, self.predict(X_val)))

        # Simpan cadangan atribut partisi dan anak cabang
        saved_state = (
            node.feature_idx,
            node.threshold,
            node.is_categorical,
            node.is_binary_categorical,
            node.left,
            node.right,
            node.children,
            node.branch_weights,
        )

        # Uji pemangkasan: ubah simpul internal menjadi daun (leaf)
        node.feature_idx = None
        node.threshold = None
        node.left = None
        node.right = None
        node.children = {}
        node.branch_weights = {}

        acc_after = float(accuracy_score(y_val, self.predict(X_val)))

        if acc_after >= acc_before - 1e-12:
            # Pertahankan pemangkasan karena akurasi validasi meningkat atau tetap sama
            self.pruning_history_.append(
                {
                    "depth": node.depth,
                    "n_samples": node.n_samples,
                    "acc_before": acc_before,
                    "acc_after": acc_after,
                }
            )
        else:
            # Batalkan pemangkasan (pulihkan cabang anak)
            (
                node.feature_idx,
                node.threshold,
                node.is_categorical,
                node.is_binary_categorical,
                node.left,
                node.right,
                node.children,
                node.branch_weights,
            ) = saved_state

    def _subtree_leaf_error_and_count(
        self, node: TreeNode, total_root_weight: float
    ) -> Tuple[float, int]:
        """
        Menghitung resubstitution error R(T_t) dan jumlah daun |L(T_t)| untuk Cost-Complexity Pruning.
        """
        if node.is_leaf:
            node_err = (node.weighted_n_samples / total_root_weight) * (
                1.0 - float(np.max(node.class_distribution))
            )
            return node_err, 1

        total_err = 0.0
        total_leaves = 0
        if node.left is not None:
            e_l, l_l = self._subtree_leaf_error_and_count(node.left, total_root_weight)
            total_err += e_l
            total_leaves += l_l
        if node.right is not None:
            e_r, l_r = self._subtree_leaf_error_and_count(node.right, total_root_weight)
            total_err += e_r
            total_leaves += l_r
        for child in node.children.values():
            e_c, l_c = self._subtree_leaf_error_and_count(child, total_root_weight)
            total_err += e_c
            total_leaves += l_c
        return total_err, max(1, total_leaves)

    def _prune_cost_complexity(self, alpha: float) -> None:
        """
        Implementasi Cost-Complexity Post-Pruning (Breiman et al., 1984) secara bottom-up.
        Memangkas sub-pohon $T_t$ apabila effective alpha:
            $\\alpha_{\\text{eff}}(t) = \\frac{R(t) - R(T_t)}{|L(T_t)| - 1} \\le \\alpha$
        """
        if self.root_ is None:
            return
        total_w = max(1.0, self.root_.weighted_n_samples)

        def _ccp_walk(node: Optional[TreeNode]) -> None:
            if node is None or node.is_leaf:
                return
            if node.left is not None:
                _ccp_walk(node.left)
            if node.right is not None:
                _ccp_walk(node.right)
            for child in list(node.children.values()):
                _ccp_walk(child)

            r_t = (node.weighted_n_samples / total_w) * (
                1.0 - float(np.max(node.class_distribution))
            )
            r_sub, n_leaves = self._subtree_leaf_error_and_count(node, total_w)
            if n_leaves > 1:
                alpha_eff = (r_t - r_sub) / (n_leaves - 1.0)
                if alpha_eff <= alpha + 1e-12:
                    node.feature_idx = None
                    node.threshold = None
                    node.left = None
                    node.right = None
                    node.children = {}
                    node.branch_weights = {}

        _ccp_walk(self.root_)

    def _prune_cost_complexity_with_val(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> None:
        """
        Mengevaluasi jalur Cost-Complexity Pruning dan memilih sub-pohon dengan akurasi validasi tertinggi,
        lalu menerapkan penyelesaian akhir dengan Reduced Error Pruning.
        """
        self._prune_cost_complexity(self.ccp_alpha)
        self._prune_reduced_error(self.root_, X_val, y_val)

    # =========================================================================
    # 7. UTILITAS INSPEKSI STRUKTUR POHON & FEATURE IMPORTANCE
    # =========================================================================

    def get_depth(self) -> int:
        """Mengembalikan kedalaman maksimum pohon keputusan saat ini."""
        if self.root_ is None:
            return 0

        def _depth(n: Optional[TreeNode]) -> int:
            if n is None or n.is_leaf:
                return 0
            child_depths = []
            if n.left is not None:
                child_depths.append(_depth(n.left))
            if n.right is not None:
                child_depths.append(_depth(n.right))
            for c in n.children.values():
                child_depths.append(_depth(c))
            return 1 + (max(child_depths) if child_depths else 0)

        return _depth(self.root_)

    def get_n_leaves(self) -> int:
        """Mengembalikan jumlah total simpul daun (leaf nodes) pada pohon."""
        if self.root_ is None:
            return 0

        def _count_leaves(n: Optional[TreeNode]) -> int:
            if n is None:
                return 0
            if n.is_leaf:
                return 1
            total = _count_leaves(n.left) + _count_leaves(n.right)
            for c in n.children.values():
                total += _count_leaves(c)
            return total

        return _count_leaves(self.root_)

    def get_n_nodes(self) -> int:
        """Mengembalikan jumlah total seluruh simpul (internal + daun) pada pohon."""
        if self.root_ is None:
            return 0

        def _count_nodes(n: Optional[TreeNode]) -> int:
            if n is None:
                return 0
            total = 1 + _count_nodes(n.left) + _count_nodes(n.right)
            for c in n.children.values():
                total += _count_nodes(c)
            return total

        return _count_nodes(self.root_)

    def compute_feature_importances(self) -> np.ndarray:
        """
        Menghitung tingkat kepentingan fitur (feature importances) ternormalisasi
        berdasarkan total penurunan ketidakmurnian berbobot (weighted impurity decrease).
        """
        if self.root_ is None or self.n_features_ == 0:
            return np.zeros(0, dtype=float)

        importances = np.zeros(self.n_features_, dtype=float)

        def _accumulate(n: Optional[TreeNode]) -> None:
            if n is None or n.is_leaf or n.feature_idx is None:
                return
            importances[n.feature_idx] += n.weighted_n_samples * max(0.0, n.gain)
            _accumulate(n.left)
            _accumulate(n.right)
            for c in n.children.values():
                _accumulate(c)

        _accumulate(self.root_)
        total_imp = float(np.sum(importances))
        if total_imp > 1e-12:
            importances /= total_imp
        return importances


# =============================================================================
# 8. BONUS VISUALISASI TREE TOP-N (+1 POIN)
# =============================================================================


def export_tree_visualization(
    model: DecisionTree,
    top_n: int = 3,
    feature_names: Optional[List[str]] = None,
    output_path: str = "output/plots/dtl_tree_top_n.png",
) -> str:
    """
    Bonus Visualisasi Tree Top-N (+1 Poin):
    Menggambar struktur pohon keputusan dari root hingga kedalaman `top_n`,
    menampilkan aturan partisi (split rule), nilai impurity, jumlah sampel (`n_samples`),
    distribusi probabilitas kelas, dan kelas dominan.

    Parameter:
        model (DecisionTree): Instance model DecisionTree yang telah dilatih (`is_fitted=True`).
        top_n (int): Kedalaman maksimum pohon yang divisualisasikan dari root (default: 3).
        feature_names (list[str] | None): Nama-nama fitur untuk ditampilkan pada kotak simpul.
        output_path (str): Jalur berkas gambar keluaran (default: 'output/plots/dtl_tree_top_n.png').

    Mengembalikan:
        str: Jalur berkas gambar yang telah disimpan.
    """
    if not model.is_fitted or model.root_ is None:
        raise RuntimeError(
            "Model DecisionTree belum dilatih. Panggil `fit(X, y)` sebelum visualisasi."
        )

    feat_names = (
        feature_names
        if feature_names is not None
        else (
            model.feature_names_
            if model.feature_names_ is not None
            else [f"Feature_{j}" for j in range(model.n_features_)]
        )
    )

    def _get_visible_children(
        node: TreeNode, current_depth: int
    ) -> List[Tuple[str, TreeNode]]:
        """Mengambil daftar (label_cabang, simpul_anak) apabila kedalaman < top_n."""
        if node.is_leaf or current_depth >= top_n:
            return []
        result: List[Tuple[str, TreeNode]] = []
        if not node.is_categorical:
            if node.left is not None:
                result.append((f"≤ {node.threshold:.2f}", node.left))
            if node.right is not None:
                result.append((f"> {node.threshold:.2f}", node.right))
        elif node.is_binary_categorical:
            if node.left is not None:
                result.append((f"== {node.threshold}", node.left))
            if node.right is not None:
                result.append((f"≠ {node.threshold}", node.right))
        else:
            for cat_val, child_node in node.children.items():
                result.append((f"= {cat_val}", child_node))
        return result

    # 1. Hitung tata letak koordinat (x, y) bebas tumpang-tindih berbasis urutan slot daun
    positions: Dict[int, Tuple[float, float]] = {}
    leaf_counter = [0]
    max_visible_depth = [0]

    def _assign_coordinates(node: TreeNode, depth: int) -> float:
        max_visible_depth[0] = max(max_visible_depth[0], depth)
        vis_children = _get_visible_children(node, depth)
        if not vis_children:
            x_coord = float(leaf_counter[0])
            leaf_counter[0] += 1
            positions[id(node)] = (x_coord, -float(depth))
            return x_coord

        child_xs = [_assign_coordinates(child, depth + 1) for _, child in vis_children]
        x_coord = float(np.mean(child_xs))
        positions[id(node)] = (x_coord, -float(depth))
        return x_coord

    _assign_coordinates(model.root_, 0)

    total_slots = max(1, leaf_counter[0])
    actual_depth = max(1, max_visible_depth[0])

    fig_width = max(13.0, min(34.0, total_slots * 2.4))
    fig_height = max(7.5, (actual_depth + 1) * 2.4)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=200)
    ax.set_facecolor("#F8FAFC")
    fig.patch.set_facecolor("#FFFFFF")

    imp_label = "gini" if model.criterion == "cart" else "entropy"

    def _format_node_label(node: TreeNode, depth: int) -> str:
        """Menyusun teks informasi simpul sesuai spesifikasi bonus visualisasi."""
        lines = []
        if not node.is_leaf and node.feature_idx is not None:
            fname = (
                feat_names[node.feature_idx]
                if node.feature_idx < len(feat_names)
                else f"Feature_{node.feature_idx}"
            )
            if not node.is_categorical:
                rule_str = f"{fname} ≤ {node.threshold:.3f}"
            elif node.is_binary_categorical:
                rule_str = f"{fname} == '{node.threshold}'"
            else:
                rule_str = f"{fname} (Categorical)"
            if depth >= top_n:
                lines.append(f"[Subtree: {rule_str}]")
            else:
                lines.append(f"{rule_str}")
        else:
            lines.append("LEAF NODE")

        lines.append(f"{imp_label} = {node.impurity:.3f} | n = {node.n_samples}")
        p0, p1 = node.class_distribution[0], node.class_distribution[1]
        lines.append(f"probs = [{p0:.2f}, {p1:.2f}]")
        lines.append(f"dominant class = {node.value}")
        return "\n".join(lines)

    def _get_node_colors(node: TreeNode) -> Tuple[str, str]:
        """Menentukan warna latar & garis tepi kotak berdasarkan kelas dominan dan kemurnian."""
        p1 = float(node.class_distribution[1])
        if node.value == 1:
            # Nuansa Coral/Oranye lembut untuk Kelas 1 (Canceled / Positif)
            alpha = 0.16 + 0.42 * max(0.0, (p1 - 0.5) * 2.0)
            cmap = matplotlib.colormaps["Oranges"]
            rgba = cmap(alpha)
            edge = "#C2410C"
        else:
            # Nuansa Biru lembut untuk Kelas 0 (Not Canceled / Negatif)
            p0 = 1.0 - p1
            alpha = 0.16 + 0.42 * max(0.0, (p0 - 0.5) * 2.0)
            cmap = matplotlib.colormaps["Blues"]
            rgba = cmap(alpha)
            edge = "#1D4ED8"
        return matplotlib.colors.to_hex(rgba), edge

    # 2. Gambar garis busur (edges) dan kotak simpul (nodes) secara rekursif
    def _draw_subtree(node: TreeNode, depth: int) -> None:
        x_parent, y_parent = positions[id(node)]
        vis_children = _get_visible_children(node, depth)

        for edge_label, child in vis_children:
            x_child, y_child = positions[id(child)]

            # Gambar panah penghubung induk -> anak
            arrow = FancyArrowPatch(
                posA=(x_parent, y_parent - 0.18),
                posB=(x_child, y_child + 0.18),
                arrowstyle="-|>,head_length=6,head_width=4",
                color="#475569",
                linewidth=1.5,
                zorder=1,
            )
            ax.add_patch(arrow)

            # Label kondisi cabang pada titik tengah busur
            mid_x = 0.52 * x_parent + 0.48 * x_child
            mid_y = 0.52 * y_parent + 0.48 * y_child
            ax.text(
                mid_x,
                mid_y,
                edge_label,
                fontsize=8.5,
                fontweight="bold",
                color="#1E293B",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.22",
                    facecolor="#FFFFFF",
                    edgecolor="#CBD5E1",
                    alpha=0.92,
                ),
                zorder=3,
            )

            _draw_subtree(child, depth + 1)

        # Gambar kotak simpul
        fill_hex, edge_hex = _get_node_colors(node)
        label_text = _format_node_label(node, depth)
        is_terminal_box = node.is_leaf or depth >= top_n

        ax.text(
            x_parent,
            y_parent,
            label_text,
            fontsize=8.5 if total_slots <= 10 else 7.5,
            fontfamily="sans-serif",
            fontweight="bold" if not is_terminal_box else "normal",
            color="#0F172A",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.45,rounding_size=0.18"
                if not is_terminal_box
                else "round,pad=0.42,rounding_size=0.35",
                facecolor=fill_hex,
                edgecolor=edge_hex,
                linewidth=2.0 if depth == 0 else 1.5,
            ),
            zorder=4,
        )

    _draw_subtree(model.root_, 0)

    # Atur batas sumbu dan judul
    ax.set_xlim(-0.8, max(0.8, total_slots - 0.2))
    ax.set_ylim(-float(actual_depth) - 0.65, 0.65)
    ax.axis("off")

    pruning_tag = " + Post-Pruning (REP)" if model.use_pruning else ""
    ax.set_title(
        f"Decision Tree Visualization — Top-{top_n} Levels "
        f"(Criterion: {model.criterion.upper()}{pruning_tag})\n"
        f"Total Nodes: {model.get_n_nodes()} | Total Leaves: {model.get_n_leaves()} | "
        f"Full Tree Depth: {model.get_depth()}",
        fontsize=13,
        fontweight="bold",
        color="#0F172A",
        pad=16,
    )

    # Tambahkan legenda kelas
    legend_handles = [
        FancyBboxPatch(
            (0, 0), 1, 1, facecolor="#BFDBFE", edgecolor="#1D4ED8", label="Dominant Class 0"
        ),
        FancyBboxPatch(
            (0, 0), 1, 1, facecolor="#FED7AA", edgecolor="#C2410C", label="Dominant Class 1"
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        fontsize=9.5,
    )

    plt.tight_layout()
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Tree Visualization Saved] Berhasil disimpan di: {output_path}")
    return output_path


def get_sklearn_dtl(criterion: str = "c45", random_state: int = RANDOM_SEED, **kwargs):
    """
    Mengembalikan instance `sklearn.tree.DecisionTreeClassifier` dengan parameter wajib
    sesuai spesifikasi FOUNDATION.md Bagian 4 (Anggota 2, Poin 4):
    - ID3  : DecisionTreeClassifier(criterion='entropy')
    - C4.5 : DecisionTreeClassifier(criterion='entropy', ccp_alpha=0.01)
    - CART : DecisionTreeClassifier(criterion='gini')
    """
    from sklearn.tree import DecisionTreeClassifier

    crit = criterion.lower().strip()
    if crit in ("id3", "entropy"):
        return DecisionTreeClassifier(criterion="entropy", random_state=random_state, **kwargs)
    elif crit in ("c45", "c4.5", "gain_ratio"):
        return DecisionTreeClassifier(
            criterion="entropy", ccp_alpha=0.01, random_state=random_state, **kwargs
        )
    elif crit in ("cart", "gini"):
        return DecisionTreeClassifier(criterion="gini", random_state=random_state, **kwargs)
    else:
        raise ValueError(f"Criterion '{criterion}' tidak dikenali.")


if __name__ == "__main__":
    import time
    from .dummy_generator import generate_clean_synthetic_data, generate_dtl_synthetic_data
    from .utils import (
        f1_score,
        k_fold_split,
        precision_score,
        recall_score,
        set_global_seed,
    )

    set_global_seed(RANDOM_SEED)
    print("=" * 78)
    print(" PENGUJIAN MODUL DECISION TREE LEARNING (DTL) — ANGGOTA 2")
    print("=" * 78)

    # 1. Uji pada dataset campuran DTL (Numerik + Kategorikal + Missing Values NaN)
    X_dtl, y_dtl = generate_dtl_synthetic_data(n_samples=500, random_state=RANDOM_SEED)
    # Buat target yang memiliki korelasi kuat dengan fitur campuran untuk menguji pembelajaran pohon
    rng = np.random.default_rng(RANDOM_SEED)
    num0 = np.where(_is_missing_array(X_dtl[:, 0]), 50.0, X_dtl[:, 0]).astype(float)
    num1 = np.where(_is_missing_array(X_dtl[:, 1]), 50.0, X_dtl[:, 1]).astype(float)
    cat2 = X_dtl[:, 2]
    rule_score = (
        (num0 > 48.0).astype(float) * 1.2
        + (num1 < 55.0).astype(float) * 0.9
        + np.isin(cat2, ["A", "B"]).astype(float) * 1.1
        + rng.normal(0, 0.45, size=len(X_dtl))
    )
    y_dtl = (rule_score >= 1.6).astype(int)

    X_train_d, X_val_d, X_test_d = X_dtl[:350], X_dtl[350:425], X_dtl[425:]
    y_train_d, y_val_d, y_test_d = y_dtl[:350], y_dtl[350:425], y_dtl[425:]

    for crit in ("id3", "c45", "cart"):
        t0 = time.perf_counter()
        dt_raw = DecisionTree(criterion=crit, use_pruning=False)
        dt_raw.fit(X_train_d, y_train_d)
        preds_raw = dt_raw.predict(X_test_d)
        acc_raw = accuracy_score(y_test_d, preds_raw)
        f1_raw = f1_score(y_test_d, preds_raw)

        dt_pruned = DecisionTree(criterion=crit, use_pruning=True, val_split_ratio=0.0)
        dt_pruned.fit(X_train_d, y_train_d, X_val=X_val_d, y_val=y_val_d)
        preds_pruned = dt_pruned.predict(X_test_d)
        acc_pruned = accuracy_score(y_test_d, preds_pruned)
        f1_pruned = f1_score(y_test_d, preds_pruned)
        elapsed = (time.perf_counter() - t0) * 1000.0

        print(
            f"[{crit.upper():4s}] Unpruned (Nodes={dt_raw.get_n_nodes():3d}, Depth={dt_raw.get_depth():2d}) "
            f"Test Acc={acc_raw:.4f}, F1={f1_raw:.4f} | "
            f"Pruned REP (Nodes={dt_pruned.get_n_nodes():3d}, Depth={dt_pruned.get_depth():2d}) "
            f"Test Acc={acc_pruned:.4f}, F1={f1_pruned:.4f} ({elapsed:.1f} ms)"
        )

    # 2. Simpan model C4.5 terbaik & Ekspor Visualisasi Top-N Tree
    best_dtl = DecisionTree(criterion="c45", use_pruning=True, val_split_ratio=0.0)
    best_dtl.fit(
        X_train_d,
        y_train_d,
        X_val=X_val_d,
        y_val=y_val_d,
    )
    best_dtl.feature_names_ = ["lead_time", "adr_price", "market_segment", "deposit_type"]
    best_dtl.save("output/models/dtl_model.pkl")
    loaded_dtl = DecisionTree.load("output/models/dtl_model.pkl")
    assert np.allclose(best_dtl.predict_proba(X_test_d), loaded_dtl.predict_proba(X_test_d))

    export_tree_visualization(
        loaded_dtl,
        top_n=3,
        feature_names=loaded_dtl.feature_names_,
        output_path="output/plots/dtl_tree_top_n.png",
    )

