"""
src/utils.py
Fungsi Evaluasi, Metrik, Validasi Silang, dan Cost-Sensitive Evaluation from Scratch.
"""

import numpy as np

RANDOM_SEED = 42


def set_global_seed(seed=RANDOM_SEED):
    """Menetapkan seed untuk menjamin reproducibility."""
    np.random.seed(seed)


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred):
    """
    Mengembalikan tuple: (TN, FP, FN, TP)
    Asumsi kelas biner: Positif = 1, Negatif = 0
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tn, fp, fn, tp


def precision_score(y_true, y_pred):
    _, fp, _, tp = confusion_matrix(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall_score(y_true, y_pred):
    _, _, fn, tp = confusion_matrix(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0


def calculate_custom_cost(y_true, y_pred, cost_fp=2.0, cost_fn=1.0):
    """
    Menghitung total penalti biaya hotel sesuai spesifikasi:
    False Positive bernilai 2x lebih mahal dari False Negative.
    Cost = (cost_fp * FP) + (cost_fn * FN)
    """
    _, fp, fn, _ = confusion_matrix(y_true, y_pred)
    return (cost_fp * fp) + (cost_fn * fn)


def k_fold_split(X, y, k=5, shuffle=True, seed=RANDOM_SEED):
    """
    Generator K-Fold Cross Validation manual murni menggunakan NumPy.
    """
    n_samples = len(X)
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1

    current = 0
    for fold_size in fold_sizes:
        val_idx = indices[current:current + fold_size]
        train_idx = np.setdiff1d(indices, val_idx)
        yield train_idx, val_idx
        current += fold_size

