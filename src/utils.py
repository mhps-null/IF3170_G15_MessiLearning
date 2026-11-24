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


def evaluate_cost_thresholds(y_true, probabilities, cost_fp=2.0, cost_fn=1.0):
    """Bandingkan ambang 0.5 dan ambang berbasis biaya pada probabilitas kelas 1."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    if (
        y_true.ndim != 1 or y_true.size == 0 or not np.isin(y_true, [0, 1]).all()
        or probabilities.shape != y_true.shape
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
    ):
        raise ValueError("Label biner dan probabilitas harus berupa vektor valid yang sejajar.")
    total_cost = cost_fp + cost_fn
    if not np.isfinite(total_cost) or min(cost_fp, cost_fn) < 0 or total_cost <= 0:
        raise ValueError("Bobot biaya harus nonnegatif dengan jumlah positif dan terbatas.")

    rows = []
    for rule, threshold in (("Standar", 0.5), ("Cost-sensitive", cost_fp / total_cost)):
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions)
        rows.append({
            "Aturan": rule,
            "Ambang": threshold,
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "Macro F1": 0.5 * (f1_score(y_true, predictions) +
                               f1_score(1 - y_true, 1 - predictions)),
            "Total biaya": calculate_custom_cost(y_true, predictions, cost_fp, cost_fn),
        })
    return rows


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

