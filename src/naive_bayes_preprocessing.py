"""
src/naive_bayes_preprocessing.py
Preprocessor diskret khusus Categorical Naive Bayes / TAN (Anggota 5).

Mengubah dataset penginapan (format `*_dtl.csv`: NaN dipertahankan, kategori masih string)
menjadi matriks bilangan bulat non-negatif (n_samples, n_features) yang siap dipakai
`NaiveBayes` dan `sklearn.naive_bayes.CategoricalNB`.

Aturan representasi (per fitur):
  - Fitur non-numerik, atau numerik dengan <= `max_categorical_unique` nilai unik non-missing
    -> kategorikal: nilai unik terurut (dari data latih) dipetakan ke kode 0..K-1.
  - Fitur numerik dengan > `max_categorical_unique` nilai unik -> diskretisasi quantile
    (`n_bins` bin; batas bin dipelajari dari data latih saja, batas duplikat dibuang).
    Interval bin berbentuk (e_{i-1}, e_i]; nilai di luar rentang latih masuk bin batas
    (bin pertama / terakhir).
  - Dua kode khusus selalu dicadangkan di akhir setiap fitur, walau tidak muncul saat latih:
        MISSING = K       -> nilai NaN/NA
        UNKNOWN = K + 1   -> kategori yang tidak pernah terlihat saat latih / nilai tak terbaca
    sehingga `n_categories_[j] = K_j + 2`.

Seluruh parameter (pemetaan kategori, batas bin) hanya dipelajari pada `fit()`;
`transform()` bersifat read-only terhadap state hasil fit (bebas data leakage).
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .preprocessing import ID_COL, TARGET_COL

KIND_CATEGORICAL = "categorical"
KIND_BINNED = "binned"


class DiscreteNBPreprocessor:
    """
    Parameter:
        n_bins (int): jumlah bin quantile untuk fitur numerik kontinu (default 5).
        max_categorical_unique (int): fitur numerik dengan nilai unik non-missing <= ini
            diperlakukan langsung sebagai kategorikal (default 12).
        feature_cols (Sequence[str] | None): daftar fitur yang dipakai (urutan dipertahankan).
            None -> semua kolom kecuali `id` dan `is_canceled`.

    Atribut hasil fit:
        feature_names_  : list nama fitur sesuai urutan kolom keluaran.
        feature_kinds_  : dict nama -> "categorical" | "binned".
        categories_     : dict nama -> np.ndarray nilai kategori terurut (fitur kategorikal).
        bin_edges_      : dict nama -> np.ndarray batas bin (fitur binned).
        n_categories_   : np.ndarray (d,) jumlah kode per fitur (termasuk MISSING & UNKNOWN).
        missing_codes_  : np.ndarray (d,) kode MISSING per fitur.
        unknown_codes_  : np.ndarray (d,) kode UNKNOWN per fitur.
    """

    def __init__(
        self,
        n_bins: int = 5,
        max_categorical_unique: int = 12,
        feature_cols: Optional[Sequence[str]] = None,
    ):
        if not isinstance(n_bins, (int, np.integer)) or isinstance(n_bins, bool) or n_bins < 2:
            raise ValueError(f"n_bins harus bilangan bulat >= 2, diterima: {n_bins!r}")
        if (
            not isinstance(max_categorical_unique, (int, np.integer))
            or isinstance(max_categorical_unique, bool)
            or max_categorical_unique < 1
        ):
            raise ValueError(
                f"max_categorical_unique harus bilangan bulat >= 1, diterima: {max_categorical_unique!r}"
            )
        self.n_bins = int(n_bins)
        self.max_categorical_unique = int(max_categorical_unique)
        self.feature_cols = None if feature_cols is None else list(feature_cols)

        self.is_fitted = False
        self.feature_names_: List[str] = []
        self.feature_kinds_: Dict[str, str] = {}
        self.is_numeric_: Dict[str, bool] = {}
        self.categories_: Dict[str, np.ndarray] = {}
        self.bin_edges_: Dict[str, np.ndarray] = {}
        self.n_categories_: Optional[np.ndarray] = None
        self.missing_codes_: Optional[np.ndarray] = None
        self.unknown_codes_: Optional[np.ndarray] = None

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("DiscreteNBPreprocessor belum di-fit. Panggil fit() pada data latih terlebih dahulu.")

    @staticmethod
    def _check_frame(X) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input harus berupa pandas DataFrame.")
        if len(X) == 0:
            raise ValueError("Input kosong (0 baris).")

    def _resolve_feature_cols(self, X: pd.DataFrame) -> List[str]:
        if self.feature_cols is None:
            cols = [c for c in X.columns if c not in (ID_COL, TARGET_COL)]
        else:
            cols = list(self.feature_cols)
            forbidden = [c for c in cols if c in (ID_COL, TARGET_COL)]
            if forbidden:
                raise ValueError(f"feature_cols tidak boleh memuat kolom non-fitur: {forbidden}")
            if len(set(cols)) != len(cols):
                raise ValueError("feature_cols memuat nama kolom duplikat.")
        if not cols:
            raise ValueError("Tidak ada fitur yang dipilih.")
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise ValueError(f"Kolom fitur hilang pada input: {missing}")
        return cols

    @staticmethod
    def _to_float(col: pd.Series):
        """Mengembalikan (nilai float64, mask NA asli, mask gagal-parse)."""
        na = col.isna().to_numpy()
        v = pd.to_numeric(col, errors="coerce").astype("float64").to_numpy()
        bad = np.isnan(v) & ~na
        return v, na, bad

    @staticmethod
    def _to_str(col: pd.Series):
        """Mengembalikan (array str dengan '' pada posisi NA, mask NA asli)."""
        na = col.isna().to_numpy()
        arr = col.to_numpy(dtype=object)
        out = np.array(["" if n else str(x) for x, n in zip(arr, na)], dtype=str)
        return out, na

    @staticmethod
    def _lookup(values: np.ndarray, cats: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Indeks nilai pada `cats` terurut; -1 jika tidak ditemukan / tidak valid."""
        out = np.full(len(values), -1, dtype=np.int64)
        k = len(cats)
        if k == 0:
            return out
        pos = np.clip(np.searchsorted(cats, values), 0, k - 1)
        found = valid & (cats[pos] == values)
        out[found] = pos[found]
        return out

    def fit(self, X: pd.DataFrame, y=None):
        """Mempelajari pemetaan kategori & batas bin dari DATA LATIH saja."""
        self._check_frame(X)
        cols = self._resolve_feature_cols(X)

        kinds, is_num, cats, edges = {}, {}, {}, {}
        n_cat, miss, unk = [], [], []
        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]

        for name in cols:
            col = X[name]
            numeric = bool(pd.api.types.is_numeric_dtype(col))
            is_num[name] = numeric
            if numeric:
                v, na, _ = self._to_float(col)
                observed = v[~na & ~np.isnan(v)]
                uniq = np.unique(observed)
                if len(uniq) <= self.max_categorical_unique:
                    kinds[name] = KIND_CATEGORICAL
                    cats[name] = uniq
                    k = len(uniq)
                else:
                    kinds[name] = KIND_BINNED
                    finite = observed[np.isfinite(observed)]
                    e = np.unique(np.quantile(finite, quantiles))
                    e = e[e < finite.max()]  # buang batas yang menghasilkan bin atas kosong
                    edges[name] = e.astype("float64")
                    k = len(e) + 1
            else:
                s, na = self._to_str(col)
                uniq = np.unique(s[~na])
                kinds[name] = KIND_CATEGORICAL
                cats[name] = uniq
                k = len(uniq)
            n_cat.append(k + 2)
            miss.append(k)
            unk.append(k + 1)

        self.feature_names_ = cols
        self.feature_kinds_ = kinds
        self.is_numeric_ = is_num
        self.categories_ = cats
        self.bin_edges_ = edges
        self.n_categories_ = np.asarray(n_cat, dtype=np.int64)
        self.missing_codes_ = np.asarray(miss, dtype=np.int64)
        self.unknown_codes_ = np.asarray(unk, dtype=np.int64)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Mengubah X menjadi matriks int64 (n_samples, n_features) memakai parameter hasil fit.
        Tidak mengubah state objek maupun DataFrame masukan.
        """
        self._check_fitted()
        self._check_frame(X)
        missing = [c for c in self.feature_names_ if c not in X.columns]
        if missing:
            raise ValueError(f"Kolom fitur hilang pada input: {missing}")

        n = len(X)
        out = np.empty((n, len(self.feature_names_)), dtype=np.int64)

        for j, name in enumerate(self.feature_names_):
            col = X[name]
            miss_code = int(self.missing_codes_[j])
            unk_code = int(self.unknown_codes_[j])
            codes = np.full(n, unk_code, dtype=np.int64)

            if self.feature_kinds_[name] == KIND_BINNED:
                v, na, bad = self._to_float(col)
                ok = ~na & ~bad
                codes[ok] = np.searchsorted(self.bin_edges_[name], v[ok], side="left")
            elif self.is_numeric_[name]:
                v, na, bad = self._to_float(col)
                idx = self._lookup(v, self.categories_[name], ~na & ~bad)
                found = idx >= 0
                codes[found] = idx[found]
            else:
                s, na = self._to_str(col)
                idx = self._lookup(s, self.categories_[name], ~na)
                found = idx >= 0
                codes[found] = idx[found]

            codes[na] = miss_code
            out[:, j] = codes
        return out

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_feature_info(self) -> pd.DataFrame:
        """Ringkasan per fitur: jenis, jumlah kode, dan kode MISSING/UNKNOWN."""
        self._check_fitted()
        return pd.DataFrame({
            "feature": self.feature_names_,
            "kind": [self.feature_kinds_[c] for c in self.feature_names_],
            "n_value_codes": self.n_categories_ - 2,
            "n_categories": self.n_categories_,
            "missing_code": self.missing_codes_,
            "unknown_code": self.unknown_codes_,
        })

    def save(self, filepath: str):
        """Menyimpan preprocessor terlatih (pickle), mengikuti konvensi BaseModel.save."""
        self._check_fitted()
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"[Preprocessor Saved] Berhasil disimpan di: {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Memuat preprocessor terlatih dari berkas pickle."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File preprocessor tidak ditemukan: {filepath}")
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Objek yang dimuat bukan instansiasi dari {cls.__name__}")
        print(f"[Preprocessor Loaded] Berhasil dimuat dari: {filepath}")
        return obj