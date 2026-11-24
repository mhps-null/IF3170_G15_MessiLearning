"""
src/svm_tuning.py
Pencarian Hyperparameter SVM dengan Stratified K-Fold Cross-Validation & Local Log-Refinement
(100% Pure NumPy — tanpa ketergantungan scikit-learn maupun scipy).

Referensi:
1. J. Bergstra and Y. Bengio, "Random search for hyper-parameter optimization,"
   Journal of Machine Learning Research, vol. 13, pp. 281-305, 2012.
2. C.-W. Hsu, C.-C. Chang, and C.-J. Lin, "A practical guide to support vector
   classification," Dept. Comput. Sci., National Taiwan Univ., Tech. Rep., 2003.
"""

import itertools
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .svm import SVM
from .utils import RANDOM_SEED, accuracy_score, f1_score


@dataclass
class SVMTuningResult:
    """Hasil pencarian hyperparameter SVM berbasis validasi silang berstrata."""

    results: list[dict[str, Any]]
    best_params_: dict[str, Any]
    best_score_: float
    best_estimator_: SVM | None = None


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 0.5 * (float(f1_score(y_true, y_pred)) + float(f1_score(1 - y_true, 1 - y_pred)))


def _stratified_subsample(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples is None or len(y) <= max_samples:
        return X, y
    chosen = []
    n_total = len(y)
    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        k = max(1, int(round(max_samples * len(cls_idx) / n_total)))
        chosen.append(rng.choice(cls_idx, size=min(k, len(cls_idx)), replace=False))
    sub_idx = np.sort(np.concatenate(chosen))
    return X[sub_idx], y[sub_idx]


def _stratified_kfold_indices(
    y: np.ndarray,
    n_splits: int,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray]]:
    class_folds = []
    for cls in np.unique(y):
        cls_idx = rng.permutation(np.flatnonzero(y == cls))
        class_folds.append(np.array_split(cls_idx, n_splits))
    folds = []
    n = len(y)
    for fold_id in range(n_splits):
        val_idx = np.sort(np.concatenate([cf[fold_id] for cf in class_folds]))
        mask = np.ones(n, dtype=bool)
        mask[val_idx] = False
        tr_idx = np.flatnonzero(mask)
        folds.append((tr_idx, val_idx))
    return folds


def _sample_param_value(spec: Any, rng: np.random.Generator) -> Any:
    if isinstance(spec, tuple) and len(spec) == 3 and isinstance(spec[0], str):
        mode, low, high = spec[0].lower(), float(spec[1]), float(spec[2])
        if mode == "log":
            return float(np.exp(rng.uniform(np.log(low), np.log(high))))
        if mode in ("uniform", "linear"):
            return float(rng.uniform(low, high))
        if mode == "int":
            return int(rng.integers(int(low), int(high) + 1))
        raise ValueError(f"Mode distribusi parameter tidak dikenal: {spec[0]!r}")
    if isinstance(spec, (list, tuple, np.ndarray)):
        if len(spec) == 1:
            return spec[0]
        return spec[int(rng.integers(0, len(spec)))]
    return spec


def _refine_param_value(
    best_val: Any,
    spec: Any,
    rng: np.random.Generator,
    factor: float = 2.0,
) -> Any:
    if isinstance(spec, tuple) and len(spec) == 3 and isinstance(spec[0], str):
        mode, low, high = spec[0].lower(), float(spec[1]), float(spec[2])
        if mode == "log":
            lo = max(low, float(best_val) / factor)
            hi = min(high, float(best_val) * factor)
            return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        if mode in ("uniform", "linear"):
            span = 0.25 * (high - low)
            lo = max(low, float(best_val) - span)
            hi = min(high, float(best_val) + span)
            return float(rng.uniform(lo, hi))
        if mode == "int":
            lo = max(int(low), int(best_val) - 1)
            hi = min(int(high), int(best_val) + 1)
            return int(rng.integers(lo, hi + 1))
    return _sample_param_value(spec, rng)


def tune_svm(
    X: np.ndarray,
    y: np.ndarray,
    param_space: dict[str, Any],
    search: str = "random",
    n_iter: int = 20,
    n_refine: int = 8,
    cv: int = 5,
    scoring: str = "macro_f1",
    tune_samples: int | None = 5000,
    tune_max_iter: int = 30,
    refit: bool = False,
    random_state: int = RANDOM_SEED,
    verbose: bool = False,
) -> SVMTuningResult:
    """
    Pencarian hyperparameter SVM dua tahap (Bergstra & Bengio, 2012; Hsu et al., 2003):
      1. Eksplorasi global (`n_iter` konfigurasi acak pada skala log/uniform atau grid penuh).
      2. Refinement lokal (`n_refine` konfigurasi di sekitar kandidat terbaik dalam rentang [0.5x, 2x]).
    Setiap kandidat dievaluasi menggunakan `cv`-fold stratified cross-validation murni NumPy.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=int).ravel()
    if cv < 2:
        raise ValueError("cv minimal bernilai 2.")

    score_fn = _macro_f1 if scoring == "macro_f1" else accuracy_score
    rng = np.random.default_rng(random_state)

    X_sub, y_sub = _stratified_subsample(X_arr, y_arr, tune_samples, rng)
    folds = _stratified_kfold_indices(y_sub, cv, rng)

    def _eval_candidate(cand: dict[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        fold_scores = []
        for tr_idx, va_idx in folds:
            est = SVM(
                **cand,
                max_iter=tune_max_iter,
                probability=False,
                random_state=random_state,
            ).fit(X_sub[tr_idx], y_sub[tr_idx])
            preds = est.predict(X_sub[va_idx])
            fold_scores.append(float(score_fn(y_sub[va_idx], preds)))
        elapsed = time.perf_counter() - t0
        arr = np.asarray(fold_scores, dtype=np.float64)
        rec = {
            "params": dict(cand),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "fold_scores": fold_scores,
            "seconds": float(elapsed),
        }
        if verbose:
            print(f"[tune_svm] {cand} -> {rec['mean']:.4f} +- {rec['std']:.4f} ({elapsed:.1f}s)")
        return rec

    records: list[dict[str, Any]] = []
    if search == "grid":
        keys = list(param_space.keys())
        values = [list(param_space[k]) if isinstance(param_space[k], (list, tuple, np.ndarray)) else [param_space[k]] for k in keys]
        for combo in itertools.product(*values):
            records.append(_eval_candidate(dict(zip(keys, combo))))
    elif search == "random":
        for _ in range(max(int(n_iter), 1)):
            cand = {k: _sample_param_value(spec, rng) for k, spec in param_space.items()}
            records.append(_eval_candidate(cand))
        records.sort(key=lambda r: (-r["mean"], r["std"]))
        best_init = records[0]["params"]
        for _ in range(max(int(n_refine), 0)):
            cand = {k: _refine_param_value(best_init[k], spec, rng, factor=2.0) for k, spec in param_space.items()}
            records.append(_eval_candidate(cand))
    else:
        raise ValueError("search harus 'random' atau 'grid'.")

    records.sort(key=lambda r: (-r["mean"], r["std"]))
    best_params = dict(records[0]["params"])
    best_score = float(records[0]["mean"])

    best_est = None
    if refit:
        best_est = SVM(**best_params, random_state=random_state).fit(X_arr, y_arr)

    return SVMTuningResult(
        results=records,
        best_params_=best_params,
        best_score_=best_score,
        best_estimator_=best_est,
    )
