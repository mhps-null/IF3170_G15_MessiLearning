"""
src/base.py
Kontrak Antarmuka Dasar (Abstract Base Class) untuk Seluruh Model Supervised Learning.
Dibuat untuk menjamin keseragaman fungsi fit, predict, predict_proba, save, dan load.
"""

from abc import ABC, abstractmethod
import pickle
import os


class BaseModel(ABC):
    def __init__(self, **kwargs):
        self.is_fitted = False
        self.classes_ = None

    @abstractmethod
    def fit(self, X, y):
        """
        Melatih model.
        Parameter:
            X: np.ndarray berukuran (n_samples, n_features)
            y: np.ndarray berukuran (n_samples,)
        Mengembalikan:
            self
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Melakukan prediksi label kelas diskrit (0 atau 1).
        Parameter:
            X: np.ndarray berukuran (n_samples, n_features)
        Mengembalikan:
            np.ndarray berukuran (n_samples,) dengan nilai 0 atau 1
        """
        pass

    @abstractmethod
    def predict_proba(self, X):
        """
        Menghasilkan probabilitas posterior untuk setiap kelas.
        Parameter:
            X: np.ndarray berukuran (n_samples, n_features)
        Mengembalikan:
            np.ndarray berukuran (n_samples, 2), di mana kolom 0 = P(y=0|X) dan kolom 1 = P(y=1|X)
        """
        pass

    def save(self, filepath: str):
        """
        Menyimpan instance model terlatih ke dalam format serialisasi biner.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"[Model Saved] Berhasil disimpan di: {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """
        Memuat instance model terlatih dari berkas serialisasi.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File model tidak ditemukan: {filepath}")
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        if not isinstance(model, cls):
            raise TypeError(f"Objek yang dimuat bukan instansiasi dari {cls.__name__}")
        print(f"[Model Loaded] Berhasil dimuat dari: {filepath}")
        return model

