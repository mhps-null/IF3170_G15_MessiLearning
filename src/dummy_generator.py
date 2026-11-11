"""
src/dummy_generator.py
Skrip pembantu untuk menghasilkan data dummy numerik dan campuran (kategorikal + missing value).
"""

import numpy as np


def generate_clean_synthetic_data(n_samples=500, n_features=6, random_state=42):
    """Dataset numerik bersih untuk testing KNN, LogReg, SVM, dan Naive Bayes."""
    rng = np.random.default_rng(random_state)
    X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
    # Logit linier sederhana dengan noise
    weights = rng.uniform(-1.5, 1.5, size=n_features)
    logits = np.dot(X, weights) + rng.normal(0, 0.5, size=n_samples)
    probs = 1 / (1 + np.exp(-logits))
    y = (probs >= 0.5).astype(int)
    return X, y


def generate_dtl_synthetic_data(n_samples=500, random_state=42):
    """Dataset campuran (numerik + kategorikal string/int + NaN) khusus pengujian DTL."""
    rng = np.random.default_rng(random_state)
    # 2 fitur numerik
    num_feat = rng.normal(50, 15, size=(n_samples, 2))
    # 2 fitur kategorikal (0, 1, 2)
    cat_feat = rng.choice(['A', 'B', 'C'], size=(n_samples, 2))

    # Gabungkan menjadi matriks object
    X = np.empty((n_samples, 4), dtype=object)
    X[:, 0:2] = num_feat
    X[:, 2:4] = cat_feat

    # Sisipkan missing values (NaN) secara acak sebesar 5%
    nan_mask = rng.random(size=X.shape) < 0.05
    X[nan_mask] = np.nan

    # Target sintetik
    y = rng.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    return X, y


if __name__ == "__main__":
    X_clean, y_clean = generate_clean_synthetic_data()
    print(f"[OK] Clean Synthetic Data Ready: X shape {X_clean.shape}, y distribution: {np.bincount(y_clean)}")
    X_dtl, y_dtl = generate_dtl_synthetic_data()
    print(f"[OK] DTL Synthetic Data Ready: X shape {X_dtl.shape}, Null count: {np.sum(X_dtl != X_dtl)}")

