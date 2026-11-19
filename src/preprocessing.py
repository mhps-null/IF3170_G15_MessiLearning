"""
src/preprocessing.py
Modul Pipeline Pemrosesan Data (Anggota 1).
Mengimplementasikan pembersihan, imputasi, encoding, standardisasi, penanganan imbalanced data,
dan reduksi dimensi (PCA) secara bebas data leakage (fit pada train, transform pada val/test),
serta dual-export dataset (standar bersih vs khusus DTL).
"""

from __future__ import annotations
import os
from typing import Optional, Union
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------------------------------------------------------
# Konstanta kolom
# ----------------------------------------------------------------------------
ID_COL = "id"
TARGET_COL = "is_canceled"

FEATURE_COLS = [
    "num_adults", "num_children", "weekend_nights", "weekday_nights",
    "meal_package", "needs_parking", "room_category", "days_before_arrival",
    "arrival_year", "arrival_month", "arrival_day", "booking_channel",
    "is_returning_guest", "prior_cancellations", "prior_completed_bookings",
    "nightly_rate_idr", "special_requests",
]

NUMERIC_IMPUTE_COLS = ["num_adults", "num_children", "special_requests", "days_before_arrival"]
CATEGORICAL_IMPUTE_COLS = ["meal_package", "room_category", "needs_parking"]
GROUP_IMPUTE_COL = "nightly_rate_idr"
GROUP_COL = "booking_channel"
MISSING_INDICATOR_COLS = NUMERIC_IMPUTE_COLS + CATEGORICAL_IMPUTE_COLS + [GROUP_IMPUTE_COL]

CLIP_COLS = ["weekend_nights", "weekday_nights", "days_before_arrival", "nightly_rate_idr", "num_children"]
UPPER_Q = 0.995

INTEGER_COLS = [
    "num_adults", "num_children", "weekend_nights", "weekday_nights", "days_before_arrival",
    "arrival_year", "arrival_month", "arrival_day", "prior_cancellations",
    "prior_completed_bookings", "nightly_rate_idr", "special_requests",
]
RAW_BINARY_COLS = ["needs_parking", "is_returning_guest"]
RAW_CATEGORY_COLS = ["meal_package", "room_category", "booking_channel"]
LOG_COLS = ["prior_cancellations", "prior_completed_bookings"]

NUMERIC_FEATURES = [
    "num_adults", "num_children", "weekend_nights", "weekday_nights", "days_before_arrival",
    "arrival_year", "arrival_month", "arrival_day", "prior_cancellations",
    "prior_completed_bookings", "nightly_rate_idr", "special_requests",
    "total_nights", "total_guests", "total_prior_bookings", "prior_cancellation_rate",
]
CATEGORICAL_FEATURES = ["meal_package", "room_category", "booking_channel", "arrival_day_of_week"]
BINARY_FEATURES = [
    "needs_parking", "is_returning_guest", "is_complimentary_stay",
    "has_completed_booking_history", "has_many_special_requests",
] + [f"{c}_missing" for c in MISSING_INDICATOR_COLS]

NEW_FEATURES = [
    "total_nights", "total_guests", "is_complimentary_stay", "has_completed_booking_history",
    "total_prior_bookings", "prior_cancellation_rate", "has_many_special_requests",
    "arrival_day_of_week",
]

class DataPipeline:
    """
    Pipeline Pemrosesan Data Bebas Leakage.
    Seluruh parameter statistik (mean, std, median, modus, kamus encoding, komponen PCA)
    wajib hanya dipelajari pada saat `fit(X_train, y_train)`.

    
    Jalur standar  : `transform(X)`               -> matriks numerik tanpa NaN (KNN, LogReg, SVM, NB).
    Jalur DTL      : `transform(X, for_dtl=True)` -> NaN dipertahankan, tanpa imputasi/scaling/encoding.
    Jalur PCA      : `transform(X, use_pca=True)` -> proyeksi PCA dari hasil jalur standar.
    """

    def __init__(self, use_pca: bool = False, n_components: Optional[Union[int, float]] = None, upper_quantile: float = UPPER_Q, random_state: int = 42):
        self.use_pca = use_pca
        self.n_components = 0.95 if (use_pca and n_components is None) else n_components
        self.upper_quantile = upper_quantile
        self.random_state = random_state
        self.is_fitted = False

        # Parameter statistik yang disimpan dari X_train
        self.numerical_medians_ = {}
        self.categorical_modes_ = {}
        self.nightly_rate_medians_ = None
        self.nightly_rate_global_median_ = None
        self.clip_upper_ = None
        self.scaling_pipeline_ = None
        self.encoding_pipeline_ = None
        self.encoded_feature_names_ = []
        self.pca_ = None
        self.pca_columns_ = []
        self.feature_names_out_ = []

    # ------------------------------------------------------------------
    # Validasi input
    # ------------------------------------------------------------------
    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("DataPipeline belum di-fit. Panggil fit() pada data latih terlebih dahulu.")

    @staticmethod
    def _select_features(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input harus berupa pandas DataFrame.")
        if len(X) == 0:
            raise ValueError("Input kosong (0 baris).")
        missing = [c for c in FEATURE_COLS if c not in X.columns]
        if missing:
            raise ValueError(f"Kolom fitur hilang pada input: {missing}")
        return X[FEATURE_COLS].copy()

    # ------------------------------------------------------------------
    # Langkah-langkah preprocessing (tanpa fit; memakai parameter hasil fit)
    # ------------------------------------------------------------------
    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in MISSING_INDICATOR_COLS:
            df[f"{col}_missing"] = df[col].isnull().astype(int)
        for col, median in self.numerical_medians_.items():
            df[col] = df[col].fillna(median)
        for col, mode in self.categorical_modes_.items():
            df[col] = df[col].fillna(mode)
        channel_medians = df[GROUP_COL].map(self.nightly_rate_medians_)
        df[GROUP_IMPUTE_COL] = (
            df[GROUP_IMPUTE_COL].fillna(channel_medians).fillna(self.nightly_rate_global_median_)
        )
        return df

    def _cap_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[CLIP_COLS] = df[CLIP_COLS].clip(upper=self.clip_upper_, axis=1)
        return df

    @staticmethod
    def _adjust_types(df: pd.DataFrame, for_dtl: bool = False) -> pd.DataFrame:
        df = df.copy()
        if for_dtl:
            # dtype nullable pandas agar NaN tetap bisa disimpan (np.int64 tidak bisa)
            for col in INTEGER_COLS:
                df[col] = np.trunc(df[col].astype("float64")).astype("Int64")
            df[RAW_BINARY_COLS] = df[RAW_BINARY_COLS].astype("Int8")
        else:
            df[INTEGER_COLS] = df[INTEGER_COLS].astype("int64")
            df[RAW_BINARY_COLS] = df[RAW_BINARY_COLS].astype("int8")
        df[RAW_CATEGORY_COLS] = df[RAW_CATEGORY_COLS].astype("category")
        return df

    @staticmethod
    def _log_transform(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[LOG_COLS] = np.log1p(df[LOG_COLS].astype("float64"))
        return df

    @staticmethod
    def _engineer(df: pd.DataFrame, for_dtl: bool = False) -> pd.DataFrame:
        df = df.copy()
        flag_dtype = "Int8" if for_dtl else "int8"

        df["total_nights"] = df["weekend_nights"] + df["weekday_nights"]
        df["total_guests"] = df["num_adults"] + df["num_children"]
        df["is_complimentary_stay"] = (df["nightly_rate_idr"] == 0).astype(flag_dtype)
        df["has_completed_booking_history"] = (df["prior_completed_bookings"] > 0).astype(flag_dtype)
        df["total_prior_bookings"] = df["prior_completed_bookings"] + df["prior_cancellations"]
        # Tanpa riwayat -> NaN (rate tidak terdefinisi)
        df["prior_cancellation_rate"] = (
            df["prior_cancellations"] / df["total_prior_bookings"].replace(0, np.nan)
        )
        df["has_many_special_requests"] = (df["special_requests"] >= 3).astype(flag_dtype)

        arrival_date = pd.to_datetime(
            df[["arrival_year", "arrival_month", "arrival_day"]]
            .astype("float64")
            .rename(columns={"arrival_year": "year", "arrival_month": "month", "arrival_day": "day"}),
            errors="coerce",
        )
        df["arrival_day_of_week"] = arrival_date.dt.dayofweek
        return df

    def _scale(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self.scaling_pipeline_.transform(df[NUMERIC_FEATURES]),
            columns=NUMERIC_FEATURES, index=df.index,
        )

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self.encoding_pipeline_.transform(df[CATEGORICAL_FEATURES]),
            columns=self.encoded_feature_names_, index=df.index,
        )

    def _transform_standard(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df = self._impute(df_raw)
        df = self._cap_outliers(df)
        df = self._adjust_types(df)
        df = self._engineer(df)
        df = self._log_transform(df)
        return pd.concat([self._scale(df), self._encode(df), df[BINARY_FEATURES]], axis=1)

    def _transform_dtl(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        # Tanpa imputasi, scaling, encoding. NaN dipertahankan; kategori tetap string.
        df = self._cap_outliers(df_raw)  # clip aman terhadap NaN
        df = self._adjust_types(df, for_dtl=True)
        df = self._engineer(df, for_dtl=True)
        df = self._log_transform(df)
        return df[FEATURE_COLS + NEW_FEATURES]

    # ------------------------------------------------------------------
    # Langkah fit (hanya dipanggil dari fit(), semuanya dari data latih)
    # ------------------------------------------------------------------
    def _fit_impute(self, df_raw: pd.DataFrame) -> None:
        self.numerical_medians_ = {c: df_raw[c].median() for c in NUMERIC_IMPUTE_COLS}
        self.categorical_modes_ = {c: df_raw[c].mode()[0] for c in CATEGORICAL_IMPUTE_COLS}
        self.nightly_rate_medians_ = df_raw.groupby(GROUP_COL)[GROUP_IMPUTE_COL].median()
        self.nightly_rate_global_median_ = df_raw[GROUP_IMPUTE_COL].median()

    def _fit_cap_outliers(self, df_imputed: pd.DataFrame) -> None:
        self.clip_upper_ = (
            df_imputed[CLIP_COLS].quantile(self.upper_quantile).apply(np.floor).astype("int64")
        )

    def _fit_scale(self, df: pd.DataFrame) -> None:
        self.scaling_pipeline_ = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]).fit(df[NUMERIC_FEATURES])

    def _fit_encode(self, df: pd.DataFrame) -> None:
        self.encoding_pipeline_ = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]).fit(df[CATEGORICAL_FEATURES])
        self.encoded_feature_names_ = list(
            self.encoding_pipeline_.named_steps["encoder"].get_feature_names_out(CATEGORICAL_FEATURES)
        )

    def _fit_pca(self, df_raw: pd.DataFrame) -> None:
        self.pca_ = PCA(n_components=self.n_components, random_state=self.random_state)
        self.pca_.fit(self._transform_standard(df_raw))
        self.pca_columns_ = [f"PC{i + 1}" for i in range(self.pca_.n_components_)]

    # ------------------------------------------------------------------
    # API publik
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Mempelajari parameter imputasi, encoding, dan scaling dari Data Latih.
        """

        df_raw = self._select_features(X)

        self._fit_impute(df_raw)
        df = self._impute(df_raw)

        self._fit_cap_outliers(df)
        df = self._cap_outliers(df)

        df = self._adjust_types(df)
        df = self._engineer(df)
        df = self._log_transform(df)

        self._fit_scale(df)
        self._fit_encode(df)
        self.feature_names_out_ = NUMERIC_FEATURES + self.encoded_feature_names_ + BINARY_FEATURES

        if self.use_pca:
            self._fit_pca(df_raw)

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame, for_dtl: bool = False, use_pca: Optional[bool] = None) -> pd.DataFrame:
        """
        Mentransformasi data menggunakan parameter yang telah dipelajari dari Data Latih.
        for_dtl=True : NaN dipertahankan, tanpa scaling/encoding (kolom kategorikal tetap string).
        use_pca      : None -> ikuti self.use_pca. Diabaikan pada jalur DTL.
        """

        self._check_fitted()
        df_raw = self._select_features(X)

        if for_dtl:
            return self._transform_dtl(df_raw)

        out = self._transform_standard(df_raw)
        if self.use_pca if use_pca is None else use_pca:
            if self.pca_ is None:
                raise RuntimeError("PCA belum di-fit. Buat pipeline dengan use_pca=True sebelum fit().")
            out = pd.DataFrame(self.pca_.transform(out), columns=self.pca_columns_, index=out.index)
        return out

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None, for_dtl: bool = False) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X, for_dtl=for_dtl)

    @staticmethod
    def _attach_meta(features: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
        out = features.copy()
        if ID_COL in raw.columns:
            out.insert(0, ID_COL, raw[ID_COL].to_numpy())
        if TARGET_COL in raw.columns:
            out[TARGET_COL] = raw[TARGET_COL].to_numpy()
        return out

    def export_processed_datasets(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: str = "data/processed",
    ) -> dict:
        """
        Dual-Export Data ke `data/processed/`:
        1. Dataset Bersih Standar: `train_clean.csv`, `val_clean.csv`, `test_clean.csv`
           (untuk KNN, Logistic Regression, SVM, Naive Bayes).
        2. Dataset Khusus DTL: `train_dtl.csv`, `test_dtl.csv`
           (tanpa imputasi NaN, khusus untuk Anggota 2 - DTL).
        """

        self._check_fitted()
        os.makedirs(output_dir, exist_ok=True)

        subsets = {"train": train_df, "val": val_df, "test": test_df}
        paths = {}
        for name, raw in subsets.items():
            clean = self._attach_meta(self.transform(raw, use_pca=False), raw)
            dtl = self._attach_meta(self.transform(raw, for_dtl=True), raw)
            for suffix, frame in (("clean", clean), ("dtl", dtl)):
                path = os.path.join(output_dir, f"{name}_{suffix}.csv")
                frame.to_csv(path, index=False)
                paths[f"{name}_{suffix}"] = path
        return paths
