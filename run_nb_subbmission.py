import os

import numpy as np
import pandas as pd

from src.naive_bayes import NaiveBayes

from src.utils import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    calculate_custom_cost,
    set_global_seed
)


set_global_seed()

# ==============================================================
# LOAD DATA
# ==============================================================

tr = pd.read_csv("data/processed/train_dtl.csv")
va = pd.read_csv("data/processed/val_dtl.csv")
te = pd.read_csv("data/processed/test_dtl.csv")

y_tr = tr["is_canceled"].to_numpy(dtype=int)
y_va = va["is_canceled"].to_numpy(dtype=int)

# ==============================================================
# TRAIN MODEL
# ==============================================================

nb = NaiveBayes(
    optimization_stage=2,

    encoding_smoothing=15.0,

    variance_candidates=(
        0.03, 0.1, 0.3, 1.0, 3.0
    ),

    regularization_candidates=(
        0.001, 0.01, 0.1
    ),

    smoothing_candidates=(
        1.0, 3.0, 5.0, 10.0,
        15.0, 30.0, 60.0, 120.0
    ),

    learning_rate=0.03,

    weight_iterations=200,

    n_folds=5,

    random_state=42
)

nb.fit(tr, y_tr)

nb.summary()

# ==============================================================
# VALIDATION
# ==============================================================

pred_va = nb.predict(va)

macro_f1 = (
    f1_score(y_va, pred_va)
    + f1_score(1 - y_va, 1 - pred_va)
) / 2

tn, fp, fn, tp = confusion_matrix(
    y_va,
    pred_va
)

print("\nVALIDATION RESULTS")

print(f"Accuracy  : {accuracy_score(y_va, pred_va):.6f}")
print(f"Precision : {precision_score(y_va, pred_va):.6f}")
print(f"Recall    : {recall_score(y_va, pred_va):.6f}")
print(f"F1 Binary : {f1_score(y_va, pred_va):.6f}")
print(f"F1 Macro  : {macro_f1:.6f}")

print(f"\nTN={tn} FP={fp} FN={fn} TP={tp}")

print(
    "Custom Cost:",
    calculate_custom_cost(y_va, pred_va)
)

# ==============================================================
# REFIT FOR KAGGLE SUBMISSION
# ==============================================================

# Only labeled training and validation data are combined.
# Kaggle test data is NEVER included in training.

full_train = pd.concat(
    [tr, va],
    ignore_index=True
)

full_y = full_train["is_canceled"].to_numpy(
    dtype=int
)

nb.fit(full_train, full_y)

# ==============================================================
# SAVE MODEL
# ==============================================================

nb.save(
    "output/models/naive_bayes_model_2.pkl"
)

# ==============================================================
# GENERATE SUBMISSION
# ==============================================================

sample = pd.read_csv(
    "data/raw/sample_submission.csv"
)

submission = pd.DataFrame({
    "id": te["id"],
    "is_canceled": nb.predict(te).astype(int)
})

assert list(submission.columns) == list(sample.columns)
assert len(submission) == len(sample)
assert set(submission["id"]) == set(sample["id"])

submission = (
    submission
    .set_index("id")
    .loc[sample["id"]]
    .reset_index()
)

os.makedirs(
    "output/submissions",
    exist_ok=True
)

submission.to_csv(
    "output/submissions/submission_nb.csv",
    index=False
)

print("\nSUBMISSION GENERATED")

print(submission.head())

print(
    "Positive prediction ratio:",
    submission["is_canceled"].mean()
)
