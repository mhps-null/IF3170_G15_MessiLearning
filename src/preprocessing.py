"""
src/preprocessing.py
Modul Pipeline Pemrosesan Data (Anggota 1).
Mengimplementasikan pembersihan, imputasi, encoding, standardisasi, penanganan imbalanced data,
dan reduksi dimensi (PCA) secara bebas data leakage (fit pada train, transform pada val/test),
serta dual-export dataset (standar bersih vs khusus DTL).
"""

from __future__ import annotations
import os
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    import pandas as pd


class PCAFromScratch:
    """
    Reduksi Dimensi Principal Component Analysis (PCA) from Scratch.
    Digunakan untuk mereduksi fitur berdimensi tinggi dan menyediakan proyeksi 2D
    bagi visualisasi bidang pemisah SVM (Anggota 4).
    """

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X: np.ndarray):
        """
        Menghitung vektor eigen (komponen utama) hanya dari data latih.
        """
        # TODO (Anggota 1): Implementasikan PCA fit (centering, covariance matrix, eigendecomposition)
        raise NotImplementedError("PCAFromScratch.fit belum diimplementasikan oleh Anggota 1.")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Memproyeksikan matriks fitur X ke ruang komponen utama.
        """
        # TODO (Anggota 1): Implementasikan proyeksi (X - self.mean_) @ self.components_.T
        raise NotImplementedError("PCAFromScratch.transform belum diimplementasikan oleh Anggota 1.")

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


class DataPipeline:
    """
    Pipeline Pemrosesan Data Bebas Leakage.
    Seluruh parameter statistik (mean, std, median, modus, kamus encoding, komponen PCA)
    wajib hanya dipelajari pada saat `fit(X_train, y_train)`.
    """

    def __init__(self, use_pca: bool = False, n_components: int = None):
        self.use_pca = use_pca
        self.n_components = n_components
        self.is_fitted = False

        # Parameter statistik yang disimpan dari X_train
        self.num_cols_ = []
        self.cat_cols_ = []
        self.impute_values_ = {}
        self.scaler_mean_ = {}
        self.scaler_std_ = {}
        self.encoding_maps_ = {}
        self.pca_ = PCAFromScratch(n_components=n_components) if use_pca else None

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Mempelajari parameter imputasi, encoding, dan scaling dari Data Latih.
        """
        # TODO (Anggota 1): Implementasikan perhitungan statistik dari X_train saja
        raise NotImplementedError("DataPipeline.fit belum diimplementasikan oleh Anggota 1.")

    def transform(self, X: pd.DataFrame, for_dtl: bool = False) -> pd.DataFrame:
        """
        Mentransformasi data menggunakan parameter yang telah dipelajari dari Data Latih.
        Jika `for_dtl=True`, nilai NaN dibiarkan (tidak diimputasi) dan fitur kategorikal
        dipertahankan sesuai kebutuhan DTL (Anggota 2).
        """
        # TODO (Anggota 1): Implementasikan transformasi bebas leakage
        raise NotImplementedError("DataPipeline.transform belum diimplementasikan oleh Anggota 1.")

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None, for_dtl: bool = False) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X, for_dtl=for_dtl)

    def export_processed_datasets(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: str = "data/processed",
    ):
        """
        Dual-Export Data ke `data/processed/`:
        1. Dataset Bersih Standar: `train_clean.csv`, `val_clean.csv`, `test_clean.csv`
           (untuk KNN, Logistic Regression, SVM, Naive Bayes).
        2. Dataset Khusus DTL: `train_dtl.csv`, `test_dtl.csv`
           (tanpa imputasi NaN, khusus untuk Anggota 2 - DTL).
        """
        os.makedirs(output_dir, exist_ok=True)
        # TODO (Anggota 1): Implementasikan ekspor kelima file CSV ke output_dir
        raise NotImplementedError("DataPipeline.export_processed_datasets belum diimplementasikan oleh Anggota 1.")
