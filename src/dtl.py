r"""
src/dtl.py
Modul Decision Tree Learning (DTL) from Scratch & Visualizer (Anggota 2).
Mendukung fitur numerik (continuous threshold), kategorikal, penanganan missing values (Quinlan C4.5),
visualisasi Top-N pohon keputusan (+1 poin bonus), dan algoritma optimasi model di luar salindia kuliah
(+2 poin bonus): Histogram-Based Quantile Split Optimization & Minimal Cost-Complexity Weakest-Link Pruning.

Dasar Teori & Formulasi Matematis:
1. Impurity & Information Gain (ID3 - Quinlan, 1986 [1]):
   $$\text{Entropy}(S) = -\sum_{c \in \{0, 1\}} p_c \log_2(p_c), \quad p_c = \frac{\sum_{i \in S, y_i = c} w_i}{\sum_{i \in S} w_i}$$

2. Penanganan Missing Values & Gain Ratio (C4.5 - Quinlan, 1993 [2]):
   Saat atribut $A$ memiliki nilai hilang (NaN/Null), himpunan sampel $S$ dipartisi menjadi
   $S_{\text{known}}$ (nilai diketahui) dan $S_{\text{missing}}$ (nilai hilang).
   Information Gain diskalakan dengan proporsi bobot sampel yang diketahui:
   $$\text{Gain}(S, A) = \frac{|S_{\text{known}}|}{|S|} \times \left( \text{Entropy}(S_{\text{known}}) - \sum_{v} \frac{|S_v|}{|S_{\text{known}}|} \text{Entropy}(S_v) \right)$$
   $$\text{SplitInfo}(S, A) = -\sum_{v} \frac{|S_v|}{|S|} \log_2\left(\frac{|S_v|}{|S|}\right) - \frac{|S_{\text{missing}}|}{|S|} \log_2\left(\frac{|S_{\text{missing}}|}{|S|}\right)$$
   $$\text{GainRatio}(S, A) = \frac{\text{Gain}(S, A)}{\text{SplitInfo}(S, A)}$$
   Saat memecah simpul ke cabang anak $v$, sampel dengan nilai missing didistribusikan ke seluruh
   cabang anak dengan bobot pecahan (fractional weight) sebanding dengan frekuensi cabang:
   $$w_i^{(v)} = w_i \times \frac{|S_v|}{|S_{\text{known}}|}$$

3. Gini Impurity (CART - Breiman et al., 1984 [4]):
   $$\text{Gini}(S) = 1 - \sum_{c \in \{0, 1\}} p_c^2$$

4. Baseline Pruning Materi Kuliah — Reduced Error Pruning (REP - Mitchell, 1997 [3]):
   Menelusuri simpul internal secara bottom-up (post-order) dan memangkas sub-pohon menjadi daun
   apabila akurasi pada himpunan validasi tidak menurun:
   $$\text{Acc}(T_{\text{pruned}}, D_{\text{val}}) \ge \text{Acc}(T_{\text{current}}, D_{\text{val}})$$

5. Bonus Optimasi Model DTL (+2 Poin — Di Luar Salindia Kuliah IF3170):
   a) Histogram-Based Quantile Split Optimization (Ke et al., 2017 [5]; Chen & Guestrin, 2016 [6]):
      Pada algoritma C4.5/CART standar (exact split), setiap simpul harus mengurutkan seluruh nilai
      unik atribut kontinu ($O(N \log N)$ per simpul) dan menguji hingga $N - 1$ kandidat midpoint.
      Optimasi berbasis histogram melakukan pra-diskretisasi kuantil (quantile binning) sekali di awal
      pelatihan (hanya pada data latih untuk mencegah data leakage) menjadi $B$ bin ($B = \text{max\_bins}$):
      $$q_k = \text{Quantile}\left(X_{\text{train}, j}, \frac{k}{B}\right), \quad k = 1, \dots, B-1$$
      Pada setiap simpul, agregasi bobot kelas per bin dihitung dalam $O(N)$ menggunakan operasi
      akumulasi histogram tanpa pengurutan ulang, dan pencarian split optimal berkurang menjadi $O(B)$
      sekaligus memberikan efek regularisasi alami terhadap nilai ekstrem/noise numerik.
   b) Minimal Cost-Complexity Weakest-Link Pruning Path (Breiman et al., 1984 [4], Bab 3):
      Meminimalkan fungsi objektif biaya-kompleksitas berbasis weighted impurity:
      $$R_\alpha(T) = R(T) + \alpha |L(T)|, \quad R(t) = \frac{W(t)}{W(\text{root})} \cdot \text{Impurity}(t)$$
      Pada setiap iterasi, simpul internal dengan tautan terlemah (weakest link) dipilih:
      $$\alpha_{\text{eff}}(t) = \frac{R(t) - R(T_t)}{|L(T_t)| - 1}, \quad t^* = \arg\min_{t \in \text{Internal}(T)} \alpha_{\text{eff}}(t)$$
      dan dipangkas selama $\alpha_{\text{eff}}(t^*) \le \alpha$.
   c) Surrogate Splits untuk Missing Values (Breiman et al., 1984 [4], Bab 5.3):
      Ketika atribut utama $A^*$ bernilai NaN, simpul mencari atribut pengganti $A' \neq A^*$
      yang memaksimalkan ukuran asosiasi prediktif (predictive measure of association):
      $$\lambda(s^* \mid s') = \frac{\min(p_L, p_R) - (1 - P_{\text{agree}}(s^*, s'))}{\min(p_L, p_R)} > 0$$
      untuk mengarahkan sampel bernilai hilang secara deterministik.
   d) Particle Swarm Optimization / PSO untuk Optimasi Struktur & Hyperparameter DTL
      (Kennedy & Eberhart, 1995 [7]; Shi & Eberhart, 1998 [8]):
      Mengoptimasi vektor hyperparameter $(max\_depth, min\_samples\_leaf, min\_samples\_split, \log_{10}\alpha)$
      terhadap fitness Macro F1 validasi internal (bebas leakage) menggunakan persamaan kecepatan & posisi partikel.

Referensi Ilmiah (Format IEEE):
[1] J. R. Quinlan, "Induction of decision trees," Machine Learning, vol. 1, no. 1, pp. 81–106, 1986.
    DOI: 10.1007/BF00116251
[2] J. R. Quinlan, C4.5: Programs for Machine Learning. San Mateo, CA, USA: Morgan Kaufmann, 1993.
[3] T. M. Mitchell, Machine Learning. New York, NY, USA: McGraw-Hill, 1997, ch. 3, pp. 52–80.
[4] L. Breiman, J. H. Friedman, R. A. Olshen, and C. J. Stone, Classification and Regression Trees.
    Belmont, CA, USA: Wadsworth International Group, 1984, ch. 3 & ch. 5.
[5] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T. Y. Liu, "LightGBM: A highly
    efficient gradient boosting decision tree," in Advances in Neural Information Processing Systems
    (NeurIPS), vol. 30, 2017, pp. 3146–3154.
[6] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in Proc. 22nd ACM SIGKDD
    Int. Conf. Knowledge Discovery and Data Mining (KDD), 2016, pp. 785–794.
    DOI: 10.1145/2939672.2939785
[7] J. Kennedy and R. Eberhart, "Particle swarm optimization," in Proc. IEEE Int. Conf. Neural
    Networks (ICNN), vol. 4, 1995, pp. 1942–1948. DOI: 10.1109/ICNN.1995.488968
[8] Y. Shi and R. Eberhart, "A modified particle swarm optimizer," in Proc. IEEE Int. Conf.
    Evolutionary Computation (ICEC), 1998, pp. 69–73. DOI: 10.1109/ICEC.1998.699146
"""

from __future__ import annotations

import copy
import heapq
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from .base import BaseModel
from .utils import RANDOM_SEED


def _is_val_missing(v: Any) -> bool:
    """Mendeteksi nilai missing skalar (None, NaN, pandas.NA/NaT, atau string kosong/'nan')."""
    if v is None:
        return True
    if isinstance(v, (float, np.floating)):
        return bool(np.isnan(v))
    if isinstance(v, str):
        return v.strip().lower() in ("", "nan", "none", "null", "na", "<na>")
    if type(v).__name__ in ("NAType", "NaTType"):
        return True
    try:
        return bool(v != v)
    except (TypeError, ValueError):
        return True


def _is_missing_array(arr: np.ndarray) -> np.ndarray:
    """
    Mendeteksi elemen missing (NaN, None, pandas.NA, atau string kosong/'nan'/'null')
    secara vektorisasi maupun objek campuran.
    """
    arr = np.asarray(arr)
    if arr.dtype.kind == "f":
        return np.isnan(arr)
    if arr.dtype.kind in ("i", "u", "b"):
        return np.zeros(arr.shape, dtype=bool)

    flat = arr.ravel()
    mask = np.fromiter((_is_val_missing(v) for v in flat), dtype=bool, count=len(flat))
    return mask.reshape(arr.shape)


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
        threshold_code: Optional[int] = None,
        children_codes: Optional[Dict[int, Any]] = None,
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
        self.threshold_code = threshold_code
        self.children_codes = children_codes if children_codes is not None else {}
        self.surrogates: List[Dict[str, Any]] = []

    @property
    def is_leaf(self) -> bool:
        """Mengembalikan True apabila simpul merupakan simpul daun (leaf node)."""
        return self.left is None and self.right is None and len(self.children) == 0


class DecisionTree(BaseModel):
    """
    Implementasi Decision Tree Classifier (ID3 / C4.5 / CART) from Scratch.

    Fitur Utama:
    1. Mendukung varian 'c45' (Gain Ratio - default), 'id3' (Information Gain), dan 'cart' (Gini).
    2. Menangani atribut kontinu menggunakan titik tengah exact (T_i = (v_i + v_{i+1}) / 2)
       ATAU optimasi Histogram-Based Quantile Split (Ke et al., 2017; Chen & Guestrin, 2016)
       melalui parameter `split_strategy='histogram'` dan `max_bins`.
    3. Menangani atribut kategorikal (multi-way split pada ID3/C4.5, binary split pada CART).
    4. Menangani missing values (NaN/Null) secara mandiri sesuai metode Quinlan (C4.5):
       - Penskalaan Gain dengan proporsi sampel non-null |S_known| / |S|.
       - Distribusi sampel missing ke seluruh cabang anak menggunakan fractional weight r_v.
       - Agregasi probabilitas berbobot pecahan saat inferensi (`predict_proba`) jika uji mengandung NaN.
    5. Mendukung Post-Pruning:
       - Reduced Error Pruning (REP - baseline materi kuliah, Mitchell 1997).
       - Minimal Cost-Complexity Weakest-Link Pruning Path (CCP - Breiman et al., 1984)
         berbasis weighted impurity.

    Parameter:
        criterion (str): 'c45' (Gain Ratio - direkomendasikan), 'id3' (Entropy), atau 'cart' (Gini).
        max_depth (int | None): Kedalaman maksimum pohon (None = tidak dibatasi).
        min_samples_split (int): Jumlah minimum sampel untuk memecah simpul (default: 2).
        min_samples_leaf (int): Jumlah minimum bobot sampel pada setiap cabang anak (default: 1).
        min_impurity_decrease (float): Ambang batas minimum penurunan impurity/gain untuk memecah simpul.
        split_strategy (str): Strategi pencarian titik batas kontinu:
            - 'exact': Menguji seluruh titik tengah nilai unik O(N log N) (baseline salindia kuliah).
            - 'histogram': Optimasi Histogram-Based Quantile Split O(B) (Bonus 6, Ke et al., 2017).
        max_bins (int): Jumlah bin kuantil B untuk `split_strategy='histogram'` (default: 64).
        use_pruning (bool): Aktifkan post-pruning (default: False).
        pruning_method (str): 'ccp' (Minimal Cost-Complexity Weakest-Link Pruning - Breiman et al., 1984)
            atau 'rep' (Reduced Error Pruning - Mitchell, 1997). Default: 'ccp'.
        ccp_alpha (float): Parameter kompleksitas alpha untuk Minimal CCP (default: 0.0005).
        val_split_ratio (float): Proporsi data latih yang disisihkan untuk validasi REP internal
            jika `pruning_method='rep'` dan `X_val` tidak diberikan secara eksplisit.
        categorical_features (list | None): Daftar indeks atau mask boolean fitur kategorikal.
        max_categorical_cardinality (int): Batas maksimum nilai unik integer agar dideteksi otomatis
            sebagai fitur kategorikal saat `categorical_features=None` (default: 12).
        random_state (int): Seed pengacakan deterministik untuk reproducibility (default: 42).
    """

    def __init__(
        self,
        criterion: str = "c45",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 1e-7,
        split_strategy: str = "exact",
        max_bins: int = 64,
        missing_strategy: str = "fractional",
        max_surrogates: int = 3,
        use_pruning: bool = False,
        pruning_method: str = "ccp",
        ccp_alpha: float = 0.0005,
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

        strategy_clean = split_strategy.lower().strip()
        if strategy_clean not in ("exact", "histogram"):
            raise ValueError(
                f"split_strategy '{split_strategy}' tidak dikenali. Pilih 'exact' atau 'histogram'."
            )
        self.split_strategy = strategy_clean
        self.max_bins = max(2, int(max_bins))

        miss_clean = missing_strategy.lower().strip()
        if miss_clean not in ("fractional", "surrogate"):
            raise ValueError(
                f"missing_strategy '{missing_strategy}' tidak dikenali. Pilih 'fractional' atau 'surrogate'."
            )
        self.missing_strategy = miss_clean
        self.max_surrogates = max(1, int(max_surrogates))

        pruning_clean = pruning_method.lower().strip()
        if pruning_clean not in ("ccp", "rep"):
            raise ValueError(
                f"pruning_method '{pruning_method}' tidak dikenali. Pilih 'ccp' atau 'rep'."
            )
        self.pruning_method = pruning_clean

        self.max_depth = max_depth
        self.min_samples_split = max(2, int(min_samples_split))
        self.min_samples_leaf = max(1, int(min_samples_leaf))
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.use_pruning = bool(use_pruning)
        self.ccp_alpha = float(ccp_alpha)
        self.val_split_ratio = float(val_split_ratio)
        self.categorical_features = categorical_features
        self.max_categorical_cardinality = int(max_categorical_cardinality)
        self.random_state = int(random_state)

        self.root_: Optional[TreeNode] = None
        self._unpruned_root_: Optional[TreeNode] = None
        self.n_features_: int = 0
        self.feature_types_: List[str] = []
        self.feature_names_: Optional[List[str]] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.pruning_history_: List[Dict[str, Any]] = []
        self._cat_to_code_: Dict[int, Dict[Any, int]] = {}
        self._code_to_cat_: Dict[int, List[Any]] = {}
        self._hist_bin_edges_: Dict[int, np.ndarray] = {}

    # =========================================================================
    # 1. DETEKSI TIPE FITUR, ENCODING INTERNAL, & QUANTILE BINNING (HISTOGRAM)
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

    def _detect_feature_types_and_encode(self, X: np.ndarray) -> np.ndarray:
        """
        Mendeteksi tipe fitur ('categorical' vs 'continuous') dan mengodekan matriks X
        ke dalam matriks float64 kontigu di mana nilai missing direpresentasikan sebagai np.nan.
        """
        n_samples, n_features = X.shape
        X_num = np.full((n_samples, n_features), np.nan, dtype=np.float64)

        user_cat_set = None
        if self.categorical_features is not None:
            if len(self.categorical_features) == n_features and all(
                isinstance(v, (bool, np.bool_)) for v in self.categorical_features
            ):
                user_cat_set = {
                    j for j, is_cat in enumerate(self.categorical_features) if is_cat
                }
            else:
                user_cat_set = {int(idx) for idx in self.categorical_features}

        self.feature_types_ = []
        self._cat_to_code_ = {}
        self._code_to_cat_ = {}

        for j in range(n_features):
            col = X[:, j]
            miss_mask = _is_missing_array(col)
            known_idx = np.where(~miss_mask)[0]
            known = col[known_idx]

            is_cat = False
            if user_cat_set is not None:
                is_cat = j in user_cat_set
            elif len(known) == 0:
                is_cat = False
            elif col.dtype.kind in ("U", "S", "b"):
                is_cat = True
            elif col.dtype.kind == "f":
                is_cat = False
            elif col.dtype.kind in ("i", "u"):
                is_cat = len(np.unique(known)) <= self.max_categorical_cardinality
            else:
                try:
                    num_vals = np.asarray(known, dtype=np.float64)
                    all_int = np.all(np.floor(num_vals) == num_vals)
                    is_cat = bool(
                        all_int and len(np.unique(num_vals)) <= self.max_categorical_cardinality
                    )
                except (ValueError, TypeError):
                    is_cat = True

            if not is_cat:
                self.feature_types_.append("continuous")
                if len(known_idx) > 0:
                    X_num[known_idx, j] = np.asarray(known, dtype=np.float64)
            else:
                self.feature_types_.append("categorical")
                cat_map: Dict[Any, int] = {}
                rev_list: List[Any] = []
                codes = np.empty(len(known_idx), dtype=np.float64)
                for idx_pos, val in enumerate(known):
                    if isinstance(val, (float, np.floating)) and float(val).is_integer():
                        val = int(val)
                    code = cat_map.get(val)
                    if code is None:
                        code = len(rev_list)
                        cat_map[val] = code
                        cat_map[str(val)] = code
                        rev_list.append(val)
                    codes[idx_pos] = float(code)
                self._cat_to_code_[j] = cat_map
                self._code_to_cat_[j] = rev_list
                if len(known_idx) > 0:
                    X_num[known_idx, j] = codes

        return X_num

    def _build_quantile_bins(self, X_train_num: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        Bonus Optimasi 6 — Pra-diskretisasi Kuantil (Histogram-Based Quantile Binning):
        Menghitung batas-batas bin kuantil (hanya dari data latih agar bebas data leakage)
        dan memetakan seluruh fitur ke matriks integer int32 (0 = NaN/missing, 1..B = bin/kategori).
        Matriks disimpan dalam urutan Fortran-contiguous agar kolom `X_bins[:, j]` bersifat stride-1
        untuk operasi `np.bincount` berkecepatan tinggi tanpa masking boolean berulang.

        Referensi:
        - Ke et al. (2017), LightGBM (NeurIPS 30), Section 3 (Histogram-based algorithm).
        - Chen & Guestrin (2016), XGBoost (ACM KDD), Section 3.3 (Quantile Sketch).
        """
        n_samples, n_features = X_train_num.shape
        X_bins = np.zeros((n_samples, n_features), dtype=np.int32, order="F")
        n_buckets_per_feat: List[int] = []
        self._hist_bin_edges_ = {}

        for j in range(n_features):
            col = X_train_num[:, j]
            known_mask = ~np.isnan(col)
            known_vals = col[known_mask]

            if self.feature_types_[j] == "categorical":
                n_cats = len(self._code_to_cat_.get(j, []))
                n_buckets_per_feat.append(max(1, n_cats))
                if len(known_vals) > 0:
                    X_bins[known_mask, j] = known_vals.astype(np.int32) + 1
                continue

            if len(known_vals) == 0:
                self._hist_bin_edges_[j] = np.empty(0, dtype=np.float64)
                n_buckets_per_feat.append(1)
                continue

            unique_vals = np.unique(known_vals)
            if len(unique_vals) <= 1:
                self._hist_bin_edges_[j] = np.empty(0, dtype=np.float64)
                X_bins[known_mask, j] = 1
                n_buckets_per_feat.append(1)
                continue

            if len(unique_vals) <= self.max_bins:
                edges = 0.5 * (unique_vals[:-1] + unique_vals[1:])
            else:
                quantiles = np.quantile(
                    known_vals, np.linspace(0.0, 1.0, self.max_bins + 1)[1:-1]
                )
                edges = np.unique(quantiles)
                edges = edges[(edges >= unique_vals[0]) & (edges < unique_vals[-1])]

            self._hist_bin_edges_[j] = edges.astype(np.float64)
            n_buckets_per_feat.append(len(edges) + 1)
            if len(edges) > 0:
                X_bins[known_mask, j] = (
                    np.searchsorted(edges, known_vals, side="right").astype(np.int32) + 1
                )
            else:
                X_bins[known_mask, j] = 1

        return X_bins, n_buckets_per_feat

    def _encode_inference_X(self, X: np.ndarray) -> np.ndarray:
        """Mengonversi matriks X saat inferensi/validasi ke matriks float64 sesuai kamus fit."""
        n_samples, n_features = X.shape
        if X.dtype == np.float64 and all(t == "continuous" for t in self.feature_types_):
            return X

        X_num = np.full((n_samples, n_features), np.nan, dtype=np.float64)
        for j in range(n_features):
            col = X[:, j]
            if self.feature_types_[j] == "continuous":
                if col.dtype.kind in ("f", "i", "u"):
                    X_num[:, j] = col.astype(np.float64)
                else:
                    miss_mask = _is_missing_array(col)
                    known_idx = np.where(~miss_mask)[0]
                    for idx_pos in known_idx:
                        try:
                            X_num[idx_pos, j] = float(col[idx_pos])
                        except (ValueError, TypeError):
                            X_num[idx_pos, j] = np.nan
            else:
                cat_map = self._cat_to_code_.get(j, {})
                miss_mask = _is_missing_array(col)
                known_idx = np.where(~miss_mask)[0]
                for idx_pos in known_idx:
                    val = col[idx_pos]
                    if isinstance(val, (float, np.floating)) and float(val).is_integer():
                        val = int(val)
                    code = cat_map.get(val)
                    if code is None:
                        code = cat_map.get(str(val))
                    if code is not None:
                        X_num[idx_pos, j] = float(code)
        return X_num

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
                entropy -= p0 * float(np.log2(p0))
            if p1 > 1e-12:
                entropy -= p1 * float(np.log2(p1))
            return float(entropy)
        else:
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
    # 3. EVALUASI KANDIDAT PARTISI (EXACT, HISTOGRAM QUANTILE, KATEGORIKAL)
    # =========================================================================

    def _score_cumulative_binary_splits(
        self,
        w0_per_bucket: np.ndarray,
        w1_per_bucket: np.ndarray,
        candidate_thresholds: np.ndarray,
        total_weight_all: float,
        weight_known: float,
        weight_missing: float,
        impurity_known: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Fungsi bersama untuk mengevaluasi seluruh kandidat split kumulatif (baik exact midpoint
        maupun histogram quantile bins) menggunakan skala Gain Quinlan C4.5 untuk missing values.
        """
        w0_left = np.cumsum(w0_per_bucket)[:-1]
        w1_left = np.cumsum(w1_per_bucket)[:-1]
        w_left = w0_left + w1_left

        total_w0 = w0_per_bucket.sum()
        total_w1 = w1_per_bucket.sum()
        w0_right = total_w0 - w0_left
        w1_right = total_w1 - w1_left
        w_right = w0_right + w1_right

        valid_mask = (w_left >= self.min_samples_leaf - 1e-9) & (
            w_right >= self.min_samples_leaf - 1e-9
        )
        if not np.any(valid_mask):
            return None

        imp_left = self._compute_impurity_vectorized(w0_left, w1_left)
        imp_right = self._compute_impurity_vectorized(w0_right, w1_right)

        known_ratio = weight_known / total_weight_all
        weighted_child_imp = (w_left * imp_left + w_right * imp_right) / weight_known
        gains = known_ratio * (impurity_known - weighted_child_imp)

        gains = np.where(valid_mask & (gains > self.min_impurity_decrease), gains, -1.0)
        best_idx = int(np.argmax(gains))
        best_gain = float(gains[best_idx])
        if best_gain <= self.min_impurity_decrease:
            return None

        if self.criterion == "c45":
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

        threshold = float(candidate_thresholds[best_idx])
        r_left = float(w_left[best_idx] / weight_known)
        r_right = float(w_right[best_idx] / weight_known)

        return {
            "score": best_score,
            "gain": best_gain,
            "threshold": threshold,
            "threshold_code": None,
            "best_bucket_idx": best_idx,
            "is_categorical": False,
            "is_binary_categorical": False,
            "branch_weights": {"left": r_left, "right": r_right},
        }

    def _evaluate_continuous_feature_exact(
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
        Baseline Salindia Kuliah (Materi 10c):
        Mengurutkan nilai unik O(N log N) dan mengevaluasi seluruh midpoint T_i = (v_i + v_{i+1}) / 2.
        """
        unique_vals, inv_idx = np.unique(col_known, return_inverse=True)
        m = len(unique_vals)
        if m <= 1:
            return None

        w0_per_unique = np.bincount(inv_idx, weights=w_known * (y_known == 0), minlength=m)
        w1_per_unique = np.bincount(inv_idx, weights=w_known * (y_known == 1), minlength=m)
        midpoints = 0.5 * (unique_vals[:-1] + unique_vals[1:])

        return self._score_cumulative_binary_splits(
            w0_per_bucket=w0_per_unique,
            w1_per_bucket=w1_per_unique,
            candidate_thresholds=midpoints,
            total_weight_all=total_weight_all,
            weight_known=weight_known,
            weight_missing=weight_missing,
            impurity_known=impurity_known,
        )

    def _evaluate_continuous_feature_histogram(
        self,
        feature_idx: int,
        w0_per_bin: np.ndarray,
        w1_per_bin: np.ndarray,
        total_weight_all: float,
        weight_known: float,
        weight_missing: float,
        impurity_known: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Bonus Optimasi 6 — Histogram-Based Quantile Split Finding (Ke et al., 2017 [5]):
        Menggunakan histogram bobot kelas per bin kuantil (dihitung sekali dalam O(N) via bincount)
        untuk mencari titik partisi optimal dalam O(B) atas batas-batas bin kuantil `edges`.
        """
        edges = self._hist_bin_edges_.get(feature_idx)
        if edges is None or len(edges) == 0:
            return None

        return self._score_cumulative_binary_splits(
            w0_per_bucket=w0_per_bin,
            w1_per_bucket=w1_per_bin,
            candidate_thresholds=edges,
            total_weight_all=total_weight_all,
            weight_known=weight_known,
            weight_missing=weight_missing,
            impurity_known=impurity_known,
        )

    def _evaluate_categorical_from_hist(
        self,
        feature_idx: int,
        w0_all_cats: np.ndarray,
        w1_all_cats: np.ndarray,
        total_weight_all: float,
        weight_known: float,
        weight_missing: float,
        impurity_known: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Mengevaluasi partisi fitur kategorikal langsung dari histogram bobot kelas per kode kategori
        (tanpa np.unique / masking ulang).
        """
        w_all = w0_all_cats + w1_all_cats
        present_codes = np.where(w_all > 1e-12)[0]
        k = len(present_codes)
        if k <= 1:
            return None

        rev_map = self._code_to_cat_[feature_idx]
        orig_categories = [rev_map[int(c)] for c in present_codes]
        w0_per_cat = w0_all_cats[present_codes]
        w1_per_cat = w1_all_cats[present_codes]
        w_per_cat = w_all[present_codes]
        known_ratio = weight_known / total_weight_all

        if self.criterion == "cart":
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
                "threshold_code": int(present_codes[best_idx]),
                "is_categorical": True,
                "is_binary_categorical": True,
                "branch_weights": {"left": r_left, "right": r_right},
            }

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
        code_to_cat_local = {
            int(present_codes[idx]): orig_categories[idx] for idx in range(k)
        }
        return {
            "score": score,
            "gain": gain,
            "threshold": None,
            "threshold_code": None,
            "is_categorical": True,
            "is_binary_categorical": False,
            "branch_weights": branch_weights,
            "code_to_cat_local": code_to_cat_local,
        }

    def _evaluate_categorical_feature(
        self,
        feature_idx: int,
        col_known: np.ndarray,
        y_known: np.ndarray,
        w_known: np.ndarray,
        total_weight_all: float,
        weight_known: float,
        weight_missing: float,
        impurity_known: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Mengevaluasi partisi pada atribut kategorikal menggunakan kode integer ter-vektorisasi:
        - Untuk ID3 dan C4.5: Partisi multi-cabang (multi-way split) untuk setiap kategori unik.
        - Untuk CART: Partisi biner (one-vs-rest: x == c vs x != c) yang memaksimalkan Gini Gain.
        """
        int_codes = col_known.astype(np.int64)
        unique_codes, inv_idx = np.unique(int_codes, return_inverse=True)
        k = len(unique_codes)
        if k <= 1:
            return None

        rev_map = self._code_to_cat_[feature_idx]
        orig_categories = [rev_map[c] for c in unique_codes]

        w0_per_cat = np.bincount(inv_idx, weights=w_known * (y_known == 0), minlength=k)
        w1_per_cat = np.bincount(inv_idx, weights=w_known * (y_known == 1), minlength=k)
        w_per_cat = w0_per_cat + w1_per_cat

        known_ratio = weight_known / total_weight_all

        if self.criterion == "cart":
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
                "threshold_code": int(unique_codes[best_idx]),
                "is_categorical": True,
                "is_binary_categorical": True,
                "branch_weights": {"left": r_left, "right": r_right},
            }

        # ID3 & C4.5: Multi-way categorical split
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
        code_to_cat_local = {
            int(unique_codes[idx]): orig_categories[idx] for idx in range(k)
        }
        return {
            "score": score,
            "gain": gain,
            "threshold": None,
            "threshold_code": None,
            "is_categorical": True,
            "is_binary_categorical": False,
            "branch_weights": branch_weights,
            "code_to_cat_local": code_to_cat_local,
        }

    # =========================================================================
    # 3b. BONUS 6 OPSI C: SURROGATE SPLITS UNTUK MISSING VALUES (BREIMAN ET AL., 1984)
    # =========================================================================

    def _find_surrogate_splits(
        self,
        primary_feat_idx: int,
        known_indices: np.ndarray,
        go_left_known: np.ndarray,
        weights: np.ndarray,
        X_num: Optional[np.ndarray],
        X_bins: Optional[np.ndarray],
        n_buckets_per_feat: Optional[List[int]],
    ) -> List[Dict[str, Any]]:
        """
        Mencari hingga `max_surrogates` atribut pengganti (surrogate splits) terbaik
        sesuai Breiman et al. (1984 [4], Bab 5.3) berdasarkan ukuran asosiasi prediktif:
            lambda(s* | s') = (min(p_L, p_R) - (1 - P_agree(s*, s'))) / min(p_L, p_R) > 0
        """
        if len(known_indices) < 4:
            return []

        w_known = weights[known_indices]
        w_left_vec = np.where(go_left_known, w_known, 0.0)
        w_right_vec = np.where(go_left_known, 0.0, w_known)
        total_known_w = float(w_known.sum())
        if total_known_w <= 1e-12:
            return []

        p_l = float(w_left_vec.sum()) / total_known_w
        p_r = float(w_right_vec.sum()) / total_known_w
        p_min = min(p_l, p_r)
        if p_min <= 1e-9:
            return []

        use_hist = X_bins is not None and n_buckets_per_feat is not None
        surrogates: List[Dict[str, Any]] = []

        for j in range(self.n_features_):
            if j == primary_feat_idx:
                continue

            if use_hist:
                bcol = X_bins[known_indices, j]
                n_b = n_buckets_per_feat[j]
                h_l = np.bincount(bcol, weights=w_left_vec, minlength=n_b + 1)
                h_r = np.bincount(bcol, weights=w_right_vec, minlength=n_b + 1)
                w_both = float(h_l[1:].sum() + h_r[1:].sum())
                if w_both <= 1e-9:
                    continue

                if self.feature_types_[j] == "continuous":
                    edges = self._hist_bin_edges_.get(j)
                    if edges is None or len(edges) == 0:
                        continue
                    cum_l = np.cumsum(h_l[1:])[:-1]
                    cum_r = np.cumsum(h_r[1:])[:-1]
                    tot_l = float(h_l[1:].sum())
                    tot_r = float(h_r[1:].sum())
                    agree_normal = (cum_l + (tot_r - cum_r)) / w_both
                    agree_inv = (cum_r + (tot_l - cum_l)) / w_both
                    best_n = int(np.argmax(agree_normal))
                    best_i = int(np.argmax(agree_inv))
                    if agree_normal[best_n] >= agree_inv[best_i]:
                        p_agree = float(agree_normal[best_n])
                        best_idx, invert = best_n, False
                    else:
                        p_agree = float(agree_inv[best_i])
                        best_idx, invert = best_i, True
                    lam = (p_min - (1.0 - p_agree)) / p_min
                    if lam > 1e-6:
                        surrogates.append(
                            {
                                "feature_idx": j,
                                "is_categorical": False,
                                "threshold": float(edges[best_idx]),
                                "bucket_idx": best_idx,
                                "code": None,
                                "invert": invert,
                                "lambda": float(lam),
                            }
                        )
                else:
                    tot_l = float(h_l[1:].sum())
                    tot_r = float(h_r[1:].sum())
                    agree_normal = (h_l[1:] + (tot_r - h_r[1:])) / w_both
                    agree_inv = (h_r[1:] + (tot_l - h_l[1:])) / w_both
                    best_n = int(np.argmax(agree_normal))
                    best_i = int(np.argmax(agree_inv))
                    if agree_normal[best_n] >= agree_inv[best_i]:
                        p_agree = float(agree_normal[best_n])
                        best_code, invert = best_n, False
                    else:
                        p_agree = float(agree_inv[best_i])
                        best_code, invert = best_i, True
                    lam = (p_min - (1.0 - p_agree)) / p_min
                    if lam > 1e-6:
                        surrogates.append(
                            {
                                "feature_idx": j,
                                "is_categorical": True,
                                "threshold": None,
                                "bucket_idx": None,
                                "code": best_code,
                                "invert": invert,
                                "lambda": float(lam),
                            }
                        )
            else:
                col_j = X_num[known_indices, j]
                valid_j = ~np.isnan(col_j)
                if not np.any(valid_j):
                    continue
                vals_j = col_j[valid_j]
                wl_j = w_left_vec[valid_j]
                wr_j = w_right_vec[valid_j]
                w_both = float(wl_j.sum() + wr_j.sum())
                if w_both <= 1e-9:
                    continue

                if self.feature_types_[j] == "continuous":
                    uvals, inv_idx = np.unique(vals_j, return_inverse=True)
                    m = len(uvals)
                    if m <= 1:
                        continue
                    hl = np.bincount(inv_idx, weights=wl_j, minlength=m)
                    hr = np.bincount(inv_idx, weights=wr_j, minlength=m)
                    cum_l = np.cumsum(hl)[:-1]
                    cum_r = np.cumsum(hr)[:-1]
                    tot_l, tot_r = float(hl.sum()), float(hr.sum())
                    agree_normal = (cum_l + (tot_r - cum_r)) / w_both
                    agree_inv = (cum_r + (tot_l - cum_l)) / w_both
                    best_n = int(np.argmax(agree_normal))
                    best_i = int(np.argmax(agree_inv))
                    if agree_normal[best_n] >= agree_inv[best_i]:
                        p_agree = float(agree_normal[best_n])
                        best_idx, invert = best_n, False
                    else:
                        p_agree = float(agree_inv[best_i])
                        best_idx, invert = best_i, True
                    lam = (p_min - (1.0 - p_agree)) / p_min
                    if lam > 1e-6:
                        surrogates.append(
                            {
                                "feature_idx": j,
                                "is_categorical": False,
                                "threshold": float(0.5 * (uvals[best_idx] + uvals[best_idx + 1])),
                                "bucket_idx": None,
                                "code": None,
                                "invert": invert,
                                "lambda": float(lam),
                            }
                        )
                else:
                    codes_j = vals_j.astype(np.int64)
                    ucodes, inv_idx = np.unique(codes_j, return_inverse=True)
                    k = len(ucodes)
                    if k <= 1:
                        continue
                    hl = np.bincount(inv_idx, weights=wl_j, minlength=k)
                    hr = np.bincount(inv_idx, weights=wr_j, minlength=k)
                    tot_l, tot_r = float(hl.sum()), float(hr.sum())
                    agree_normal = (hl + (tot_r - hr)) / w_both
                    agree_inv = (hr + (tot_l - hl)) / w_both
                    best_n = int(np.argmax(agree_normal))
                    best_i = int(np.argmax(agree_inv))
                    if agree_normal[best_n] >= agree_inv[best_i]:
                        p_agree = float(agree_normal[best_n])
                        best_code, invert = int(ucodes[best_n]), False
                    else:
                        p_agree = float(agree_inv[best_i])
                        best_code, invert = int(ucodes[best_i]), True
                    lam = (p_min - (1.0 - p_agree)) / p_min
                    if lam > 1e-6:
                        surrogates.append(
                            {
                                "feature_idx": j,
                                "is_categorical": True,
                                "threshold": None,
                                "bucket_idx": None,
                                "code": best_code,
                                "invert": invert,
                                "lambda": float(lam),
                            }
                        )

        surrogates.sort(key=lambda s: s["lambda"], reverse=True)
        return surrogates[: self.max_surrogates]

    @staticmethod
    def _route_missing_with_surrogates(
        surrogates: List[Dict[str, Any]],
        miss_indices: np.ndarray,
        X_num: Optional[np.ndarray],
        X_bins: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Mengarahkan sampel yang missing pada fitur utama memakai urutan surrogate splits."""
        n_miss = len(miss_indices)
        go_left = np.zeros(n_miss, dtype=bool)
        resolved = np.zeros(n_miss, dtype=bool)

        for sur in surrogates:
            pending_pos = np.where(~resolved)[0]
            if len(pending_pos) == 0:
                break
            row_ids = miss_indices[pending_pos]
            j = sur["feature_idx"]
            if X_bins is not None:
                bvals = X_bins[row_ids, j]
                avail = bvals > 0
                if not np.any(avail):
                    continue
                if not sur["is_categorical"]:
                    cond = bvals[avail] <= (sur["bucket_idx"] + 1)
                else:
                    cond = bvals[avail] == (sur["code"] + 1)
            else:
                vals = X_num[row_ids, j]
                avail = ~np.isnan(vals)
                if not np.any(avail):
                    continue
                if not sur["is_categorical"]:
                    cond = vals[avail] <= sur["threshold"]
                else:
                    cond = vals[avail] == float(sur["code"])

            if sur["invert"]:
                cond = ~cond
            hit_pos = pending_pos[avail]
            go_left[hit_pos] = cond
            resolved[hit_pos] = True

        return go_left, resolved

    # =========================================================================
    # 4. KONSTRUKSI POHON KEPUTUSAN REKURSIF & DISTRIBUSI PECAHAN SAMPEL NAN
    # =========================================================================

    def _build_tree(
        self,
        X_num: Optional[np.ndarray],
        y: np.ndarray,
        weights: np.ndarray,
        depth: int,
        X_bins: Optional[np.ndarray] = None,
        n_buckets_per_feat: Optional[List[int]] = None,
        used_cat_features: Tuple[int, ...] = (),
    ) -> TreeNode:
        """
        Membangun pohon keputusan secara rekursif dari atas ke bawah (top-down induction).
        Pada mode `histogram`, seluruh evaluasi split & partisi bekerja langsung pada matriks
        integer `X_bins` (stride-1 bincount tanpa pengurutan ulang maupun penyalinan matriks float64).
        """
        is_c0 = y == 0
        w0_vec = np.where(is_c0, weights, 0.0)
        w1_vec = np.where(is_c0, 0.0, weights)
        w0 = float(w0_vec.sum())
        w1 = float(w1_vec.sum())
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

        if (
            node_impurity <= 1e-12
            or (self.max_depth is not None and depth >= self.max_depth)
            or total_weight < self.min_samples_split
            or len(y) < self.min_samples_split
        ):
            return node

        candidates: List[Tuple[int, Dict[str, Any]]] = []
        use_hist = X_bins is not None and n_buckets_per_feat is not None

        for j in range(self.n_features_):
            if (
                self.feature_types_[j] == "categorical"
                and self.criterion in ("id3", "c45")
                and j in used_cat_features
            ):
                continue

            if use_hist:
                bcol = X_bins[:, j]
                n_b = n_buckets_per_feat[j]
                h0 = np.bincount(bcol, weights=w0_vec, minlength=n_b + 1)
                h1 = np.bincount(bcol, weights=w1_vec, minlength=n_b + 1)
                weight_missing = float(h0[0] + h1[0])
                w0_known = float(h0[1:].sum())
                w1_known = float(h1[1:].sum())
                weight_known = w0_known + w1_known
                if weight_known < self.min_samples_split:
                    continue
                impurity_known = self._compute_impurity_from_counts(w0_known, w1_known)
                if impurity_known <= 1e-12:
                    continue

                if self.feature_types_[j] == "continuous":
                    candidate = self._evaluate_continuous_feature_histogram(
                        feature_idx=j,
                        w0_per_bin=h0[1:],
                        w1_per_bin=h1[1:],
                        total_weight_all=total_weight,
                        weight_known=weight_known,
                        weight_missing=weight_missing,
                        impurity_known=impurity_known,
                    )
                else:
                    candidate = self._evaluate_categorical_from_hist(
                        feature_idx=j,
                        w0_all_cats=h0[1:],
                        w1_all_cats=h1[1:],
                        total_weight_all=total_weight,
                        weight_known=weight_known,
                        weight_missing=weight_missing,
                        impurity_known=impurity_known,
                    )
            else:
                col = X_num[:, j]
                known_mask = ~np.isnan(col)
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

                if self.feature_types_[j] == "continuous":
                    candidate = self._evaluate_continuous_feature_exact(
                        col_known=col[known_mask],
                        y_known=y_known,
                        w_known=w_known,
                        total_weight_all=total_weight,
                        weight_known=weight_known,
                        weight_missing=weight_missing,
                        impurity_known=impurity_known,
                    )
                else:
                    candidate = self._evaluate_categorical_feature(
                        feature_idx=j,
                        col_known=col[known_mask],
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
            avg_gain = float(np.mean([cand["gain"] for _, cand in candidates]))
            above_avg = [
                (j, cand) for j, cand in candidates if cand["gain"] >= avg_gain - 1e-12
            ]
            pool = above_avg if above_avg else candidates
            best_feature_idx, best_split = max(pool, key=lambda item: item[1]["score"])
        else:
            best_feature_idx, best_split = max(candidates, key=lambda item: item[1]["score"])

        node.feature_idx = best_feature_idx
        node.threshold = best_split["threshold"]
        node.threshold_code = best_split["threshold_code"]
        node.is_categorical = best_split["is_categorical"]
        node.is_binary_categorical = best_split["is_binary_categorical"]
        node.gain = best_split["gain"]
        node.branch_weights = best_split["branch_weights"]

        min_frac_weight = 1e-3

        if use_hist:
            bcol_best = X_bins[:, best_feature_idx]
            miss_mask = bcol_best == 0
            known_mask = ~miss_mask
        else:
            col_best = X_num[:, best_feature_idx]
            miss_mask = np.isnan(col_best)
            known_mask = ~miss_mask

        if not node.is_categorical or node.is_binary_categorical:
            if use_hist:
                known_bvals = bcol_best[known_mask]
                if not node.is_categorical:
                    go_left_known = known_bvals <= (best_split["best_bucket_idx"] + 1)
                else:
                    go_left_known = known_bvals == (node.threshold_code + 1)
            else:
                known_vals = col_best[known_mask]
                if not node.is_categorical:
                    go_left_known = known_vals <= node.threshold
                else:
                    go_left_known = known_vals == float(node.threshold_code)

            known_indices = np.where(known_mask)[0]
            miss_indices = np.where(miss_mask)[0]

            left_known_idx = known_indices[go_left_known]
            right_known_idx = known_indices[~go_left_known]

            r_left = node.branch_weights["left"]
            r_right = node.branch_weights["right"]

            if len(miss_indices) > 0:
                if self.missing_strategy == "surrogate":
                    # Bonus 6 Opsi C — Surrogate Splits (Breiman et al., 1984 [4], Bab 5.3):
                    # Cari atribut pengganti dengan asosiasi prediktif lambda > 0 terhadap primary split
                    node.surrogates = self._find_surrogate_splits(
                        primary_feat_idx=best_feature_idx,
                        known_indices=known_indices,
                        go_left_known=go_left_known,
                        weights=weights,
                        X_num=X_num,
                        X_bins=X_bins,
                        n_buckets_per_feat=n_buckets_per_feat,
                    )
                    miss_go_left, miss_resolved = self._route_missing_with_surrogates(
                        surrogates=node.surrogates,
                        miss_indices=miss_indices,
                        X_num=X_num,
                        X_bins=X_bins,
                    )
                    resolved_left = miss_indices[miss_resolved & miss_go_left]
                    resolved_right = miss_indices[miss_resolved & ~miss_go_left]
                    unresolved = miss_indices[~miss_resolved]

                    # Jika semua surrogate juga NaN, arahkan ke cabang mayoritas (aturan standar CART)
                    if len(unresolved) > 0:
                        if r_left >= r_right:
                            resolved_left = np.concatenate([resolved_left, unresolved])
                        else:
                            resolved_right = np.concatenate([resolved_right, unresolved])

                    left_idx = np.concatenate([left_known_idx, resolved_left])
                    left_w = weights[left_idx]
                    right_idx = np.concatenate([right_known_idx, resolved_right])
                    right_w = weights[right_idx]
                else:
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
                node.threshold_code = None
                node.branch_weights = {}
                return node

            node.left = self._build_tree(
                None if use_hist else X_num[left_idx],
                y[left_idx],
                left_w,
                depth + 1,
                np.asfortranarray(X_bins[left_idx]) if use_hist else None,
                n_buckets_per_feat,
                used_cat_features,
            )
            node.right = self._build_tree(
                None if use_hist else X_num[right_idx],
                y[right_idx],
                right_w,
                depth + 1,
                np.asfortranarray(X_bins[right_idx]) if use_hist else None,
                n_buckets_per_feat,
                used_cat_features,
            )
        else:
            known_indices = np.where(known_mask)[0]
            miss_indices = np.where(miss_mask)[0]
            if use_hist:
                known_codes = (bcol_best[known_mask] - 1).astype(np.int64)
            else:
                known_codes = col_best[known_mask].astype(np.int64)
            new_used_cats = used_cat_features + (best_feature_idx,)
            code_to_cat_local: Dict[int, Any] = best_split["code_to_cat_local"]
            node.children_codes = code_to_cat_local

            for code_int, cat_val in code_to_cat_local.items():
                r_cat = node.branch_weights[cat_val]
                cat_known_idx = known_indices[known_codes == code_int]

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
                        None if use_hist else X_num[child_idx],
                        y[child_idx],
                        child_w,
                        depth + 1,
                        np.asfortranarray(X_bins[child_idx]) if use_hist else None,
                        n_buckets_per_feat,
                        new_used_cats,
                    )

        return node

    # =========================================================================
    # 5. ANTARMUKA UTAMA: FIT, PREDICT_PROBA (TER-VEKTORISASI), & PREDICT
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
        - Jika `split_strategy='histogram'`, jalankan pra-diskretisasi kuantil (Ke et al., 2017)
          hanya pada data latih (bebas data leakage).
        - Jika `use_pruning=True`, jalankan post-pruning sesuai `pruning_method` ('ccp' atau 'rep').
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

        X_num = self._detect_feature_types_and_encode(X_arr)

        X_train_use = X_num
        y_train_use = y_arr
        X_val_raw = self._prepare_input_X(X_val) if X_val is not None else None
        y_val_use = np.asarray(y_val).astype(int).ravel() if y_val is not None else None

        if (
            self.use_pruning
            and self.pruning_method == "rep"
            and X_val_raw is None
            and 0.0 < self.val_split_ratio < 0.5
            and len(X_num) >= 20
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

            X_train_use = X_num[train_idx]
            y_train_use = y_arr[train_idx]
            X_val_num = X_num[val_idx]
            y_val_use = y_arr[val_idx]
        else:
            X_val_num = self._encode_inference_X(X_val_raw) if X_val_raw is not None else None

        # Hitung quantile histogram bins HANYA pada data latih X_train_use (bebas data leakage)
        if self.split_strategy == "histogram":
            X_bins, n_buckets_per_feat = self._build_quantile_bins(X_train_use)
            X_num_build = None
        else:
            X_bins, n_buckets_per_feat = None, None
            X_num_build = X_train_use

        init_weights = np.ones(len(y_train_use), dtype=float)
        self.root_ = self._build_tree(
            X_num=X_num_build,
            y=y_train_use,
            weights=init_weights,
            depth=0,
            X_bins=X_bins,
            n_buckets_per_feat=n_buckets_per_feat,
            used_cat_features=(),
        )
        self._unpruned_root_ = copy.deepcopy(self.root_)
        self.is_fitted = True

        if self.use_pruning:
            if self.pruning_method == "ccp":
                self._prune_cost_complexity_weakest_link(alpha=self.ccp_alpha)
            elif X_val_num is not None and y_val_use is not None:
                self._prune_from_encoded_val(X_val_num, y_val_use)
            else:
                self._prune_cost_complexity_weakest_link(alpha=max(self.ccp_alpha, 0.0005))

        self.feature_importances_ = self.compute_feature_importances()
        return self

    def _predict_batch_proba(self, X_sub: np.ndarray, node: TreeNode) -> np.ndarray:
        """
        Penelusuran pohon ter-vektorisasi (batch routing) untuk menghitung probabilitas
        posterior [P(y=0|X), P(y=1|X)]. Mendukung agregasi bobot pecahan (fractional weight)
        ala Quinlan (C4.5) apabila sampel memiliki nilai NaN atau unseen category.
        """
        n_rows = len(X_sub)
        if n_rows == 0:
            return np.empty((0, 2), dtype=float)

        if node.is_leaf or node.feature_idx is None:
            out = np.empty((n_rows, 2), dtype=float)
            out[:, 0] = node.class_distribution[0]
            out[:, 1] = node.class_distribution[1]
            return out

        col = X_sub[:, node.feature_idx]
        miss_mask = np.isnan(col)
        known_mask = ~miss_mask

        out = np.empty((n_rows, 2), dtype=float)

        if not node.is_categorical or node.is_binary_categorical:
            if np.any(known_mask):
                known_vals = col[known_mask]
                if not node.is_categorical:
                    go_left = known_vals <= node.threshold
                else:
                    go_left = known_vals == float(node.threshold_code)

                known_idx = np.where(known_mask)[0]
                left_idx = known_idx[go_left]
                right_idx = known_idx[~go_left]

                if len(left_idx) > 0:
                    out[left_idx] = (
                        self._predict_batch_proba(X_sub[left_idx], node.left)
                        if node.left is not None
                        else node.class_distribution
                    )
                if len(right_idx) > 0:
                    out[right_idx] = (
                        self._predict_batch_proba(X_sub[right_idx], node.right)
                        if node.right is not None
                        else node.class_distribution
                    )

            if np.any(miss_mask):
                miss_idx = np.where(miss_mask)[0]
                if self.missing_strategy == "surrogate" and node.surrogates:
                    miss_go_left, miss_resolved = self._route_missing_with_surrogates(
                        surrogates=node.surrogates,
                        miss_indices=miss_idx,
                        X_num=X_sub,
                        X_bins=None,
                    )
                    res_left = miss_idx[miss_resolved & miss_go_left]
                    res_right = miss_idx[miss_resolved & ~miss_go_left]
                    unres = miss_idx[~miss_resolved]

                    if len(res_left) > 0:
                        out[res_left] = (
                            self._predict_batch_proba(X_sub[res_left], node.left)
                            if node.left is not None
                            else node.class_distribution
                        )
                    if len(res_right) > 0:
                        out[res_right] = (
                            self._predict_batch_proba(X_sub[res_right], node.right)
                            if node.right is not None
                            else node.class_distribution
                        )
                    if len(unres) > 0:
                        r_left = node.branch_weights.get("left", 0.5)
                        r_right = node.branch_weights.get("right", 0.5)
                        target_child = node.left if r_left >= r_right else node.right
                        out[unres] = (
                            self._predict_batch_proba(X_sub[unres], target_child)
                            if target_child is not None
                            else node.class_distribution
                        )
                else:
                    X_miss = X_sub[miss_idx]
                    r_left = node.branch_weights.get("left", 0.5)
                    r_right = node.branch_weights.get("right", 0.5)
                    p_left = (
                        self._predict_batch_proba(X_miss, node.left)
                        if node.left is not None
                        else node.class_distribution
                    )
                    p_right = (
                        self._predict_batch_proba(X_miss, node.right)
                        if node.right is not None
                        else node.class_distribution
                    )
                    out[miss_idx] = r_left * p_left + r_right * p_right

            return out

        unmatched_mask = miss_mask.copy()
        if np.any(known_mask):
            known_idx = np.where(known_mask)[0]
            known_codes = col[known_idx].astype(np.int64)
            matched_known = np.zeros(len(known_idx), dtype=bool)

            for code_int, cat_val in node.children_codes.items():
                child_node = node.children.get(cat_val)
                if child_node is None:
                    continue
                mask_c = known_codes == code_int
                if np.any(mask_c):
                    matched_known |= mask_c
                    c_idx = known_idx[mask_c]
                    out[c_idx] = self._predict_batch_proba(X_sub[c_idx], child_node)

            if not np.all(matched_known):
                unmatched_mask[known_idx[~matched_known]] = True

        if np.any(unmatched_mask):
            unmatched_idx = np.where(unmatched_mask)[0]
            X_unm = X_sub[unmatched_idx]
            agg_prob = np.zeros((len(unmatched_idx), 2), dtype=float)
            total_r = 0.0
            for cat_val, child_node in node.children.items():
                r_cat = node.branch_weights.get(cat_val, 0.0)
                if r_cat > 0.0:
                    agg_prob += r_cat * self._predict_batch_proba(X_unm, child_node)
                    total_r += r_cat
            if total_r > 1e-12:
                out[unmatched_idx] = agg_prob / total_r
            else:
                out[unmatched_idx] = node.class_distribution

        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Menghasilkan probabilitas posterior [P(y=0|X), P(y=1|X)] berukuran (n_samples, 2).
        Mendukung agregasi bobot pecahan (fractional weight) apabila sampel memiliki nilai NaN.
        """
        if not self.is_fitted or self.root_ is None:
            raise RuntimeError(
                "Model DecisionTree belum dilatih. Panggil `fit(X, y)` terlebih dahulu."
            )

        X_arr = self._prepare_input_X(X, is_training=False)
        X_num = self._encode_inference_X(X_arr)
        probs = self._predict_batch_proba(X_num, self.root_)

        row_sums = probs.sum(axis=1, keepdims=True)
        valid_sums = row_sums > 1e-12
        probs = np.where(valid_sums, probs / np.where(valid_sums, row_sums, 1.0), 0.5)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Melakukan prediksi label kelas diskrit (0 atau 1).
        """
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

    # =========================================================================
    # 6. POST-PRUNING: MINIMAL COST-COMPLEXITY WEAKEST-LINK (CCP) & REP
    # =========================================================================

    def prune(
        self,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        ccp_alpha: Optional[float] = None,
    ) -> "DecisionTree":
        r"""
        Melakukan pemangkasan pohon pasca-pelatihan (Post-Pruning):
        - Jika `ccp_alpha` diberikan atau `pruning_method == 'ccp'`:
          Menjalankan **Minimal Cost-Complexity Weakest-Link Pruning** (Breiman et al., 1984 [4], Bab 3)
          berbasis weighted impurity $R_\alpha(T) = R(T) + \alpha |L(T)|$.
        - Jika `pruning_method == 'rep'` dan `(X_val, y_val)` diberikan:
          Menjalankan **Reduced Error Pruning (REP)** secara bottom-up (Mitchell, 1997 [3], Bab 3).
        """
        if not self.is_fitted or self.root_ is None:
            raise RuntimeError("Model DecisionTree belum dilatih. Panggil `fit` sebelum `prune`.")

        if ccp_alpha is not None:
            self.use_pruning = True
            self.pruning_method = "ccp"
            self.ccp_alpha = float(ccp_alpha)
            self._prune_cost_complexity_weakest_link(alpha=self.ccp_alpha)
        elif self.pruning_method == "ccp" and X_val is None:
            self.use_pruning = True
            self._prune_cost_complexity_weakest_link(alpha=self.ccp_alpha)
        elif X_val is not None and y_val is not None:
            self.use_pruning = True
            self.pruning_method = "rep"
            X_val_arr = self._prepare_input_X(X_val, is_training=False)
            X_val_num = self._encode_inference_X(X_val_arr)
            y_val_arr = np.asarray(y_val).astype(int).ravel()
            self._prune_from_encoded_val(X_val_num, y_val_arr)
        else:
            self.use_pruning = True
            self._prune_cost_complexity_weakest_link(alpha=self.ccp_alpha)

        self.feature_importances_ = self.compute_feature_importances()
        return self

    def _prune_from_encoded_val(self, X_val_num: np.ndarray, y_val: np.ndarray) -> None:
        """Menjalankan Reduced Error Pruning pada matriks validasi ter-encode."""
        self.pruning_history_ = []
        if self.root_ is not None:
            self._prune_rep_single_pass(self.root_, X_val_num, y_val)

    def _prune_rep_single_pass(
        self, node: TreeNode, X_sub: np.ndarray, y_sub: np.ndarray
    ) -> np.ndarray:
        """
        Reduced Error Pruning (Mitchell, 1997 [3]) bottom-up dalam satu lintasan ter-vektorisasi.
        """
        n_rows = len(X_sub)
        if node.is_leaf or node.feature_idx is None:
            out = np.empty((n_rows, 2), dtype=float)
            out[:, 0] = node.class_distribution[0]
            out[:, 1] = node.class_distribution[1]
            return out

        if n_rows == 0:
            if node.left is not None:
                self._prune_rep_single_pass(node.left, X_sub, y_sub)
            if node.right is not None:
                self._prune_rep_single_pass(node.right, X_sub, y_sub)
            for child in list(node.children.values()):
                self._prune_rep_single_pass(child, X_sub, y_sub)
            return np.empty((0, 2), dtype=float)

        col = X_sub[:, node.feature_idx]
        miss_mask = np.isnan(col)
        known_mask = ~miss_mask
        probs_subtree = np.empty((n_rows, 2), dtype=float)

        if not node.is_categorical or node.is_binary_categorical:
            known_idx = np.where(known_mask)[0]
            if len(known_idx) > 0:
                known_vals = col[known_idx]
                if not node.is_categorical:
                    go_left = known_vals <= node.threshold
                else:
                    go_left = known_vals == float(node.threshold_code)
                left_idx = known_idx[go_left]
                right_idx = known_idx[~go_left]
            else:
                left_idx = np.empty(0, dtype=int)
                right_idx = np.empty(0, dtype=int)

            if node.left is not None:
                probs_left = self._prune_rep_single_pass(
                    node.left, X_sub[left_idx], y_sub[left_idx]
                )
                if len(left_idx) > 0:
                    probs_subtree[left_idx] = probs_left
            elif len(left_idx) > 0:
                probs_subtree[left_idx] = node.class_distribution

            if node.right is not None:
                probs_right = self._prune_rep_single_pass(
                    node.right, X_sub[right_idx], y_sub[right_idx]
                )
                if len(right_idx) > 0:
                    probs_subtree[right_idx] = probs_right
            elif len(right_idx) > 0:
                probs_subtree[right_idx] = node.class_distribution

            if np.any(miss_mask):
                miss_idx = np.where(miss_mask)[0]
                X_miss = X_sub[miss_idx]
                r_left = node.branch_weights.get("left", 0.5)
                r_right = node.branch_weights.get("right", 0.5)
                p_l = (
                    self._predict_batch_proba(X_miss, node.left)
                    if node.left is not None
                    else node.class_distribution
                )
                p_r = (
                    self._predict_batch_proba(X_miss, node.right)
                    if node.right is not None
                    else node.class_distribution
                )
                probs_subtree[miss_idx] = r_left * p_l + r_right * p_r
        else:
            unmatched_mask = miss_mask.copy()
            known_idx = np.where(known_mask)[0]
            known_codes = (
                col[known_idx].astype(np.int64)
                if len(known_idx) > 0
                else np.empty(0, dtype=np.int64)
            )
            matched_known = np.zeros(len(known_idx), dtype=bool)

            for code_int, cat_val in list(node.children_codes.items()):
                child_node = node.children.get(cat_val)
                if child_node is None:
                    continue
                mask_c = known_codes == code_int
                c_idx = known_idx[mask_c]
                if len(c_idx) > 0:
                    matched_known |= mask_c
                probs_c = self._prune_rep_single_pass(
                    child_node, X_sub[c_idx], y_sub[c_idx]
                )
                if len(c_idx) > 0:
                    probs_subtree[c_idx] = probs_c

            if len(known_idx) > 0 and not np.all(matched_known):
                unmatched_mask[known_idx[~matched_known]] = True

            if np.any(unmatched_mask):
                unm_idx = np.where(unmatched_mask)[0]
                X_unm = X_sub[unm_idx]
                agg_prob = np.zeros((len(unm_idx), 2), dtype=float)
                total_r = 0.0
                for cat_val, child_node in node.children.items():
                    r_cat = node.branch_weights.get(cat_val, 0.0)
                    if r_cat > 0.0:
                        agg_prob += r_cat * self._predict_batch_proba(X_unm, child_node)
                        total_r += r_cat
                probs_subtree[unm_idx] = (
                    agg_prob / total_r if total_r > 1e-12 else node.class_distribution
                )

        preds_subtree = (probs_subtree[:, 1] >= 0.5).astype(int)
        acc_subtree = float(np.mean(preds_subtree == y_sub))
        acc_leaf = float(np.mean(y_sub == node.value))

        if acc_leaf >= acc_subtree - 1e-12:
            self._collapse_to_leaf(node)
            self.pruning_history_.append(
                {
                    "method": "rep",
                    "depth": node.depth,
                    "n_samples": node.n_samples,
                    "acc_before": acc_subtree,
                    "acc_after": acc_leaf,
                }
            )
            out_leaf = np.empty((n_rows, 2), dtype=float)
            out_leaf[:, 0] = node.class_distribution[0]
            out_leaf[:, 1] = node.class_distribution[1]
            return out_leaf

        return probs_subtree

    @staticmethod
    def _collapse_to_leaf(node: TreeNode) -> None:
        """Mengubah simpul internal menjadi simpul daun (leaf node)."""
        node.feature_idx = None
        node.threshold = None
        node.threshold_code = None
        node.left = None
        node.right = None
        node.children = {}
        node.children_codes = {}
        node.branch_weights = {}
        node.surrogates = []

    def _compute_weakest_link_stats(
        self, root: TreeNode, total_w: float
    ) -> Tuple[float, List[TreeNode], float]:
        """
        Menghitung R(T_t), |L(T_t)|, dan alpha_eff(t) untuk seluruh simpul internal dalam
        satu penelusuran post-order O(|Nodes|), lalu mengembalikan (min_alpha, weakest_nodes, root_R).
        Menggunakan weighted impurity: R(t) = (W(t) / W(root)) * Impurity(t) (Breiman et al., 1984 [4]).
        """
        min_alpha = float("inf")
        weakest_nodes: List[TreeNode] = []

        def _walk(node: TreeNode) -> Tuple[float, int]:
            nonlocal min_alpha, weakest_nodes
            r_node = (node.weighted_n_samples / total_w) * node.impurity
            if node.is_leaf:
                return r_node, 1

            r_sub = 0.0
            n_leaves = 0
            if node.left is not None:
                rl, ll = _walk(node.left)
                r_sub += rl
                n_leaves += ll
            if node.right is not None:
                rr, lr = _walk(node.right)
                r_sub += rr
                n_leaves += lr
            for child in node.children.values():
                rc, lc = _walk(child)
                r_sub += rc
                n_leaves += lc

            if n_leaves > 1:
                alpha_eff = max(0.0, (r_node - r_sub) / (n_leaves - 1.0))
                if alpha_eff < min_alpha - 1e-12:
                    min_alpha = alpha_eff
                    weakest_nodes = [node]
                elif abs(alpha_eff - min_alpha) <= 1e-12:
                    weakest_nodes.append(node)

            return r_sub, max(1, n_leaves)

        root_r, _ = _walk(root)
        return min_alpha, weakest_nodes, root_r

    def _prune_cost_complexity_weakest_link(self, alpha: float) -> None:
        """
        Minimal Cost-Complexity Weakest-Link Pruning (Breiman et al., 1984 [4], Bab 3)
        menggunakan priority queue (min-heap) + pembaruan jalur leluhur O(|Nodes| * Depth * log|Nodes|):
        Secara matematis 100% identik dengan weakest-link pruning iteratif, namun jauh lebih cepat
        karena hanya memperbarui statistik R(T_t) dan |L(T_t)| pada leluhur simpul yang dipangkas.
        """
        if self.root_ is None or alpha <= 0.0 or self.root_.is_leaf:
            return
        total_w = max(1.0, self.root_.weighted_n_samples)
        self.pruning_history_ = []

        parent_map: Dict[int, Optional[TreeNode]] = {}
        r_node_map: Dict[int, float] = {}
        r_sub_map: Dict[int, float] = {}
        n_leaves_map: Dict[int, int] = {}
        alive_map: Dict[int, bool] = {}
        heap: List[Tuple[float, int, TreeNode]] = []
        tie_counter = 0

        def _init_walk(node: TreeNode, parent: Optional[TreeNode]) -> Tuple[float, int]:
            nonlocal tie_counter
            nid = id(node)
            parent_map[nid] = parent
            alive_map[nid] = True
            rn = (node.weighted_n_samples / total_w) * node.impurity
            r_node_map[nid] = rn

            if node.is_leaf:
                r_sub_map[nid] = rn
                n_leaves_map[nid] = 1
                return rn, 1

            r_sub = 0.0
            n_l = 0
            if node.left is not None:
                rl, ll = _init_walk(node.left, node)
                r_sub += rl
                n_l += ll
            if node.right is not None:
                rr, lr = _init_walk(node.right, node)
                r_sub += rr
                n_l += lr
            for child in node.children.values():
                rc, lc = _init_walk(child, node)
                r_sub += rc
                n_l += lc

            r_sub_map[nid] = r_sub
            n_leaves_map[nid] = max(1, n_l)
            if n_l > 1:
                a_eff = max(0.0, (rn - r_sub) / (n_l - 1.0))
                tie_counter += 1
                heapq.heappush(heap, (a_eff, tie_counter, node))
            return r_sub, max(1, n_l)

        def _mark_descendants_dead(node: TreeNode) -> None:
            if node.left is not None:
                alive_map[id(node.left)] = False
                _mark_descendants_dead(node.left)
            if node.right is not None:
                alive_map[id(node.right)] = False
                _mark_descendants_dead(node.right)
            for child in node.children.values():
                alive_map[id(child)] = False
                _mark_descendants_dead(child)

        _init_walk(self.root_, None)

        while heap and not self.root_.is_leaf:
            a_eff, _, node = heapq.heappop(heap)
            nid = id(node)
            if not alive_map.get(nid, False) or node.is_leaf:
                continue
            n_l = n_leaves_map[nid]
            if n_l <= 1:
                continue
            current_a_eff = max(0.0, (r_node_map[nid] - r_sub_map[nid]) / (n_l - 1.0))
            if abs(current_a_eff - a_eff) > 1e-12:
                continue
            if current_a_eff > alpha + 1e-12:
                break

            # Pangkas simpul terlemah (weakest link)
            delta_r = r_node_map[nid] - r_sub_map[nid]
            delta_l = 1 - n_l
            _mark_descendants_dead(node)
            self._collapse_to_leaf(node)
            r_sub_map[nid] = r_node_map[nid]
            n_leaves_map[nid] = 1

            # Perbarui hanya jalur leluhur menuju root O(depth)
            anc = parent_map.get(nid)
            while anc is not None:
                aid = id(anc)
                if not alive_map.get(aid, False):
                    break
                r_sub_map[aid] += delta_r
                n_leaves_map[aid] += delta_l
                if n_leaves_map[aid] > 1:
                    new_a = max(
                        0.0,
                        (r_node_map[aid] - r_sub_map[aid]) / (n_leaves_map[aid] - 1.0),
                    )
                    tie_counter += 1
                    heapq.heappush(heap, (new_a, tie_counter, anc))
                anc = parent_map.get(aid)

            self.pruning_history_.append(
                {
                    "method": "ccp",
                    "effective_alpha": float(current_a_eff),
                    "pruned_count": 1,
                    "tree_impurity": float(r_sub_map[id(self.root_)]),
                }
            )

    def cost_complexity_pruning_path(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Menghasilkan jalur lengkap Minimal Cost-Complexity Weakest-Link Pruning (Breiman et al., 1984 [4]):
        Mengembalikan (ccp_alphas, total_leaf_impurities) dari pohon awal hingga root.
        """
        if not self.is_fitted or self.root_ is None:
            raise RuntimeError("Model DecisionTree belum dilatih.")

        tree_copy = copy.deepcopy(self._unpruned_root_ or self.root_)
        total_w = max(1.0, tree_copy.weighted_n_samples)
        alphas = [0.0]
        _, _, init_r = self._compute_weakest_link_stats(tree_copy, total_w)
        impurities = [float(init_r)]

        while not tree_copy.is_leaf:
            min_alpha, weakest_nodes, _ = self._compute_weakest_link_stats(tree_copy, total_w)
            if not weakest_nodes:
                break
            for target_node in weakest_nodes:
                self._collapse_to_leaf(target_node)
            _, _, new_r = self._compute_weakest_link_stats(tree_copy, total_w)
            alphas.append(float(min_alpha))
            impurities.append(float(new_r))

        return np.asarray(alphas, dtype=float), np.asarray(impurities, dtype=float)

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


def _short_text(text: str, max_len: int = 20) -> str:
    """Memendekkan nama fitur/label yang terlalu panjang agar kotak visualisasi tetap rapi."""
    s = str(text)
    return s if len(s) <= max_len else s[: max_len - 2] + ".."


def export_tree_visualization(
    model: DecisionTree,
    top_n: int = 3,
    feature_names: Optional[List[str]] = None,
    output_path: str = "output/plots/dtl_tree_top_n.png",
    max_branches_per_node: int = 3,
) -> str:
    """
    Bonus Visualisasi Tree Top-N (+1 Poin):
    Menggambar struktur pohon keputusan dari root hingga kedalaman `top_n`,
    menampilkan aturan partisi (split rule), nilai impurity, jumlah sampel (`n`),
    distribusi probabilitas kelas, dan kelas dominan tanpa tumpang tindih kotak.

    Parameter:
        model (DecisionTree): Model DecisionTree yang sudah dilatih (`is_fitted=True`).
        top_n (int): Kedalaman maksimum percabangan yang ditampilkan dari root (default: 3).
        feature_names (list[str] | None): Daftar nama fitur.
        output_path (str): Jalur penyimpanan berkas gambar PNG.
        max_branches_per_node (int): Batas maksimum cabang per simpul kategorikal multi-cabang
            yang digambar agar tata letak bebas tumpang tindih (default: 3).
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

    hidden_branch_counts: Dict[int, int] = {}

    def _get_visible_children(
        node: TreeNode, current_depth: int
    ) -> List[Tuple[str, TreeNode]]:
        if node.is_leaf or current_depth >= top_n:
            return []
        result: List[Tuple[str, TreeNode]] = []
        if not node.is_categorical:
            if node.left is not None:
                result.append((f"≤ {node.threshold:,.2f}", node.left))
            if node.right is not None:
                result.append((f"> {node.threshold:,.2f}", node.right))
        elif node.is_binary_categorical:
            if node.left is not None:
                result.append((f"== {_short_text(node.threshold, 14)}", node.left))
            if node.right is not None:
                result.append((f"≠ {_short_text(node.threshold, 14)}", node.right))
        else:
            items = list(node.children.items())
            if len(items) > max_branches_per_node:
                items = sorted(items, key=lambda kv: kv[1].n_samples, reverse=True)
                hidden_branch_counts[id(node)] = len(items) - max_branches_per_node
                items = items[:max_branches_per_node]
            for cat_val, child_node in items:
                result.append((f"= {_short_text(cat_val, 14)}", child_node))
        return result

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

    # Skalakan lebar kanvas secara proporsional terhadap jumlah slot daun agar kotak tidak bersenggolan
    slot_width_inch = 3.2 if total_slots <= 10 else 2.7
    fig_width = max(14.0, min(44.0, total_slots * slot_width_inch))
    fig_height = max(8.0, (actual_depth + 1) * 2.8)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=200)
    ax.set_facecolor("#F8FAFC")
    fig.patch.set_facecolor("#FFFFFF")

    imp_label = "gini" if model.criterion == "cart" else "entropy"

    def _format_node_label(node: TreeNode, depth: int) -> str:
        lines = []
        if not node.is_leaf and node.feature_idx is not None:
            fname = _short_text(
                feat_names[node.feature_idx]
                if node.feature_idx < len(feat_names)
                else f"Feature_{node.feature_idx}",
                max_len=20,
            )
            if not node.is_categorical:
                rule_str = f"{fname} ≤ {node.threshold:,.2f}"
            elif node.is_binary_categorical:
                rule_str = f"{fname} == '{_short_text(node.threshold, 12)}'"
            else:
                hidden = hidden_branch_counts.get(id(node), 0)
                suffix = f" (+{hidden} lainnya)" if hidden > 0 else ""
                rule_str = f"{fname} [Cat]{suffix}"
            if depth >= top_n:
                lines.append(f"Subtree: {rule_str}")
            else:
                lines.append(rule_str)
        else:
            lines.append("LEAF NODE")

        lines.append(f"{imp_label}={node.impurity:.3f} | n={node.n_samples:,}")
        p0, p1 = node.class_distribution[0], node.class_distribution[1]
        lines.append(f"P=[{p0:.2f}, {p1:.2f}] | class={node.value}")
        return "\n".join(lines)

    def _get_node_colors(node: TreeNode) -> Tuple[str, str]:
        p1 = float(node.class_distribution[1])
        if node.value == 1:
            alpha = 0.16 + 0.40 * max(0.0, (p1 - 0.5) * 2.0)
            rgba = matplotlib.colormaps["Oranges"](alpha)
            edge = "#C2410C"
        else:
            p0 = 1.0 - p1
            alpha = 0.16 + 0.40 * max(0.0, (p0 - 0.5) * 2.0)
            rgba = matplotlib.colormaps["Blues"](alpha)
            edge = "#1D4ED8"
        return matplotlib.colors.to_hex(rgba), edge

    node_fontsize = 8.5 if total_slots <= 8 else (7.8 if total_slots <= 12 else 7.0)

    def _draw_subtree(node: TreeNode, depth: int) -> None:
        x_parent, y_parent = positions[id(node)]
        vis_children = _get_visible_children(node, depth)

        for edge_label, child in vis_children:
            x_child, y_child = positions[id(child)]

            arrow = FancyArrowPatch(
                posA=(x_parent, y_parent - 0.16),
                posB=(x_child, y_child + 0.16),
                arrowstyle="-|>,head_length=6,head_width=4",
                color="#475569",
                linewidth=1.4,
                zorder=1,
            )
            ax.add_patch(arrow)

            mid_x = 0.52 * x_parent + 0.48 * x_child
            mid_y = 0.52 * y_parent + 0.48 * y_child
            ax.text(
                mid_x,
                mid_y,
                edge_label,
                fontsize=max(6.8, node_fontsize - 0.5),
                fontweight="bold",
                color="#1E293B",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="#FFFFFF",
                    edgecolor="#CBD5E1",
                    alpha=0.94,
                ),
                zorder=3,
            )

            _draw_subtree(child, depth + 1)

        fill_hex, edge_hex = _get_node_colors(node)
        label_text = _format_node_label(node, depth)
        is_terminal_box = node.is_leaf or depth >= top_n

        ax.text(
            x_parent,
            y_parent,
            label_text,
            fontsize=node_fontsize,
            fontfamily="sans-serif",
            fontweight="bold" if not is_terminal_box else "normal",
            color="#0F172A",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.42,rounding_size=0.16"
                if not is_terminal_box
                else "round,pad=0.38,rounding_size=0.30",
                facecolor=fill_hex,
                edgecolor=edge_hex,
                linewidth=2.0 if depth == 0 else 1.4,
            ),
            zorder=4,
        )

    _draw_subtree(model.root_, 0)

    ax.set_xlim(-0.75, max(0.75, total_slots - 0.25))
    ax.set_ylim(-float(actual_depth) - 0.60, 0.60)
    ax.axis("off")

    opt_tags = []
    if model.split_strategy == "histogram":
        opt_tags.append(f"Histogram Quantile B={model.max_bins}")
    if model.missing_strategy == "surrogate":
        opt_tags.append(f"Surrogate Splits (K={model.max_surrogates})")
    if model.use_pruning:
        if model.pruning_method == "ccp":
            opt_tags.append(f"Minimal CCP α={model.ccp_alpha:.6g}")
        else:
            opt_tags.append("REP")
    opt_str = (" + " + " + ".join(opt_tags)) if opt_tags else ""

    ax.set_title(
        f"Decision Tree Visualization — Top-{top_n} Levels "
        f"(Criterion: {model.criterion.upper()}{opt_str})\n"
        f"Total Nodes: {model.get_n_nodes()} | Total Leaves: {model.get_n_leaves()} | "
        f"Full Tree Depth: {model.get_depth()}",
        fontsize=13,
        fontweight="bold",
        color="#0F172A",
        pad=16,
    )

    legend_handles = [
        FancyBboxPatch(
            (0, 0), 1, 1, facecolor="#BFDBFE", edgecolor="#1D4ED8", label="Dominant Class 0 (Not Canceled)"
        ),
        FancyBboxPatch(
            (0, 0), 1, 1, facecolor="#FED7AA", edgecolor="#C2410C", label="Dominant Class 1 (Canceled)"
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


# =============================================================================
# 9. BONUS 6 OPSI A: PARTICLE SWARM OPTIMIZATION (PSO) UNTUK STRUKTUR DTL
# =============================================================================


class PSODecisionTreeOptimizer(BaseModel):
    r"""
    Bonus Optimasi 6 (Opsi A) — Particle Swarm Optimization (PSO) untuk Optimasi Struktur
    dan Hyperparameter Decision Tree (100% Pure NumPy from Scratch).

    Algoritma ini mengoptimasi vektor parameter struktur & pruning:
        $\mathbf{z} = [\text{max\_depth}, \text{min\_samples\_leaf}, \text{min\_samples\_split}, \log_{10}(\alpha_{\text{ccp}})]$
    dengan fungsi objektif (fitness) berupa Macro F1-Score pada validasi internal
    (stratified holdout / K-Fold CV di dalam data latih — bebas data leakage terhadap validation set luar).

    Persamaan Pembaruan Partikel (Kennedy & Eberhart, 1995 [7]; Shi & Eberhart, 1998 [8]):
        $$v_i^{(t+1)} = w^{(t)} v_i^{(t)} + c_1 r_1 (p_i^{\text{best}} - z_i^{(t)}) + c_2 r_2 (g^{\text{best}} - z_i^{(t)})$$
        $$z_i^{(t+1)} = \text{clip}(z_i^{(t)} + v_i^{(t+1)}, \mathbf{z}_{\min}, \mathbf{z}_{\max})$$
    dengan bobot inersia meluruh linier $w^{(t)} = w_{\max} - \frac{t}{T}(w_{\max} - w_{\min})$.

    Referensi Ilmiah (IEEE):
    [7] J. Kennedy and R. Eberhart, "Particle swarm optimization," in Proc. IEEE Int. Conf.
        Neural Networks (ICNN), vol. 4, 1995, pp. 1942–1948. DOI: 10.1109/ICNN.1995.488968
    [8] Y. Shi and R. Eberhart, "A modified particle swarm optimizer," in Proc. IEEE Int. Conf.
        Evolutionary Computation (ICEC), 1998, pp. 69–73. DOI: 10.1109/ICEC.1998.699146
    """

    def __init__(
        self,
        criterion: str = "c45",
        split_strategy: str = "histogram",
        max_bins: int = 64,
        missing_strategy: str = "fractional",
        n_particles: int = 8,
        n_iters: int = 6,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1: float = 1.5,
        c2: float = 1.5,
        inner_val_ratio: float = 0.2,
        patience: int = 3,
        categorical_features: Optional[List[Union[int, bool]]] = None,
        random_state: int = RANDOM_SEED,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.criterion = criterion
        self.split_strategy = split_strategy
        self.max_bins = int(max_bins)
        self.missing_strategy = missing_strategy
        self.n_particles = max(3, int(n_particles))
        self.n_iters = max(2, int(n_iters))
        self.w_max = float(w_max)
        self.w_min = float(w_min)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.inner_val_ratio = float(inner_val_ratio)
        self.patience = max(1, int(patience))
        self.categorical_features = categorical_features
        self.random_state = int(random_state)

        # Ruang pencarian: [max_depth, min_samples_leaf, min_samples_split, log10(ccp_alpha)]
        self.bounds_low_ = np.array([6.0, 1.0, 2.0, -4.5], dtype=float)
        self.bounds_high_ = np.array([26.0, 25.0, 60.0, -2.5], dtype=float)

        self.best_params_: Dict[str, Any] = {}
        self.best_score_: float = -1.0
        self.best_model_: Optional[DecisionTree] = None
        self.history_: List[Dict[str, Any]] = []

    @staticmethod
    def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        def _f1_bin(yt: np.ndarray, yp: np.ndarray) -> float:
            tp = np.sum((yt == 1) & (yp == 1))
            fp = np.sum((yt == 0) & (yp == 1))
            fn = np.sum((yt == 1) & (yp == 0))
            denom = 2 * tp + fp + fn
            return float(2 * tp / denom) if denom > 0 else 0.0

        return 0.5 * (_f1_bin(y_true, y_pred) + _f1_bin(1 - y_true, 1 - y_pred))

    def _decode_particle(self, pos: np.ndarray) -> Dict[str, Any]:
        max_depth = int(round(float(pos[0])))
        min_leaf = max(1, int(round(float(pos[1]))))
        min_split = max(2 * min_leaf, int(round(float(pos[2]))))
        ccp_alpha = float(10.0 ** float(pos[3]))
        return {
            "max_depth": max_depth,
            "min_samples_leaf": min_leaf,
            "min_samples_split": min_split,
            "ccp_alpha": ccp_alpha,
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PSODecisionTreeOptimizer":
        """
        Menjalankan pencarian PSO pada data latih (menggunakan stratified inner holdout
        agar bebas data leakage terhadap validation/test set eksternal), lalu melatih ulang
        model DecisionTree terbaik pada 100% data latih (X, y).
        """
        y_arr = np.asarray(y).astype(int).ravel()
        self.classes_ = np.unique(y_arr)
        rng = np.random.default_rng(self.random_state)

        # Stratified inner split khusus untuk evaluasi fitness PSO di dalam X_train
        idx_0 = np.where(y_arr == 0)[0]
        idx_1 = np.where(y_arr == 1)[0]
        rng.shuffle(idx_0)
        rng.shuffle(idx_1)
        n_v0 = max(1, int(round(len(idx_0) * self.inner_val_ratio)))
        n_v1 = max(1, int(round(len(idx_1) * self.inner_val_ratio)))
        in_val_idx = np.concatenate([idx_0[:n_v0], idx_1[:n_v1]])
        in_tr_idx = np.concatenate([idx_0[n_v0:], idx_1[n_v1:]])

        X_arr = X.to_numpy(dtype=object) if hasattr(X, "to_numpy") else np.asarray(X)
        X_in_tr, y_in_tr = X_arr[in_tr_idx], y_arr[in_tr_idx]
        X_in_va, y_in_va = X_arr[in_val_idx], y_arr[in_val_idx]

        # Inisialisasi posisi & kecepatan partikel
        dim = len(self.bounds_low_)
        span = self.bounds_high_ - self.bounds_low_
        positions = self.bounds_low_ + rng.random((self.n_particles, dim)) * span
        # Seed 1 partikel di titik awal yang baik agar konvergensi cepat
        positions[0] = np.array([18.0, 5.0, 10.0, np.log10(0.0003)], dtype=float)
        velocities = rng.uniform(-0.15, 0.15, size=(self.n_particles, dim)) * span

        p_best_pos = positions.copy()
        p_best_scores = np.full(self.n_particles, -1.0, dtype=float)
        g_best_pos = positions[0].copy()
        g_best_score = -1.0

        eval_cache: Dict[Tuple[int, int, int, int], float] = {}
        self.history_ = []
        no_improve = 0

        for it in range(self.n_iters):
            improved_this_iter = False
            iter_scores = []

            for p in range(self.n_particles):
                params = self._decode_particle(positions[p])
                cache_key = (
                    params["max_depth"],
                    params["min_samples_leaf"],
                    params["min_samples_split"],
                    int(round(np.log10(params["ccp_alpha"]) * 20)),
                )
                if cache_key in eval_cache:
                    score = eval_cache[cache_key]
                else:
                    clf = DecisionTree(
                        criterion=self.criterion,
                        split_strategy=self.split_strategy,
                        max_bins=self.max_bins,
                        missing_strategy=self.missing_strategy,
                        max_depth=params["max_depth"],
                        min_samples_leaf=params["min_samples_leaf"],
                        min_samples_split=params["min_samples_split"],
                        use_pruning=True,
                        pruning_method="ccp",
                        ccp_alpha=params["ccp_alpha"],
                        categorical_features=self.categorical_features,
                        random_state=self.random_state,
                    )
                    clf.fit(X_in_tr, y_in_tr)
                    preds = clf.predict(X_in_va)
                    score = self._macro_f1(y_in_va, preds)
                    eval_cache[cache_key] = score

                iter_scores.append(score)
                if score > p_best_scores[p]:
                    p_best_scores[p] = score
                    p_best_pos[p] = positions[p].copy()
                if score > g_best_score + 1e-6:
                    g_best_score = score
                    g_best_pos = positions[p].copy()
                    improved_this_iter = True

            self.history_.append(
                {
                    "iter": it + 1,
                    "g_best_macro_f1": float(g_best_score),
                    "mean_macro_f1": float(np.mean(iter_scores)),
                    "best_params": self._decode_particle(g_best_pos),
                }
            )

            if improved_this_iter:
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

            # Pembaruan kecepatan & posisi partikel (Kennedy & Eberhart, 1995; Shi & Eberhart, 1998)
            w = self.w_max - (it / max(1, self.n_iters - 1)) * (self.w_max - self.w_min)
            r1 = rng.random((self.n_particles, dim))
            r2 = rng.random((self.n_particles, dim))
            velocities = (
                w * velocities
                + self.c1 * r1 * (p_best_pos - positions)
                + self.c2 * r2 * (g_best_pos - positions)
            )
            positions = np.clip(positions + velocities, self.bounds_low_, self.bounds_high_)

        self.best_score_ = float(g_best_score)
        self.best_params_ = self._decode_particle(g_best_pos)

        # Latih ulang model DTL terbaik pada 100% data latih (X, y)
        self.best_model_ = DecisionTree(
            criterion=self.criterion,
            split_strategy=self.split_strategy,
            max_bins=self.max_bins,
            missing_strategy=self.missing_strategy,
            max_depth=self.best_params_["max_depth"],
            min_samples_leaf=self.best_params_["min_samples_leaf"],
            min_samples_split=self.best_params_["min_samples_split"],
            use_pruning=True,
            pruning_method="ccp",
            ccp_alpha=self.best_params_["ccp_alpha"],
            categorical_features=self.categorical_features,
            random_state=self.random_state,
        )
        if hasattr(X, "columns"):
            self.best_model_.feature_names_ = [str(c) for c in X.columns]
        self.best_model_.fit(X_arr, y_arr)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.best_model_ is None:
            raise RuntimeError("PSODecisionTreeOptimizer belum dilatih.")
        return self.best_model_.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.best_model_ is None:
            raise RuntimeError("PSODecisionTreeOptimizer belum dilatih.")
        return self.best_model_.predict(X)


if __name__ == "__main__":
    import time
    from .dummy_generator import generate_dtl_synthetic_data
    from .utils import accuracy_score, f1_score, set_global_seed

    set_global_seed(RANDOM_SEED)
    print("=" * 80)
    print(" VERIFIKASI MODUL DECISION TREE LEARNING (DTL) — ANGGOTA 2 (PURE NUMPY)")
    print("=" * 80)

    X_dtl, _ = generate_dtl_synthetic_data(n_samples=600, random_state=RANDOM_SEED)
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

    X_tr, X_te = X_dtl[:450], X_dtl[450:]
    y_tr, y_te = y_dtl[:450], y_dtl[450:]

    configs = [
        ("C4.5 Exact (Baseline)", dict(criterion="c45", split_strategy="exact", use_pruning=False)),
        (
            "C4.5 Exact + REP (Kuliah)",
            dict(criterion="c45", split_strategy="exact", use_pruning=True, pruning_method="rep", val_split_ratio=0.2),
        ),
        (
            "C4.5 Histogram B=32 + CCP (Bonus 6)",
            dict(criterion="c45", split_strategy="histogram", max_bins=32, use_pruning=True, pruning_method="ccp", ccp_alpha=0.008),
        ),
        (
            "CART Histogram B=32 + CCP (Bonus 6)",
            dict(criterion="cart", split_strategy="histogram", max_bins=32, use_pruning=True, pruning_method="ccp", ccp_alpha=0.005),
        ),
    ]

    for label, params in configs:
        t0 = time.perf_counter()
        clf = DecisionTree(**params, random_state=RANDOM_SEED)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        elapsed = (time.perf_counter() - t0) * 1000.0
        macro_f1 = 0.5 * (f1_score(y_te, preds) + f1_score(1 - y_te, 1 - preds))
        print(
            f"{label:38s} | Acc={accuracy_score(y_te, preds):.4f} | MacroF1={macro_f1:.4f} "
            f"| Nodes={clf.get_n_nodes():3d} | Depth={clf.get_depth():2d} | {elapsed:.1f} ms"
        )
