import hashlib
from pathlib import Path
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

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "output"
MODEL_DIR = OUTPUT_DIR / "models"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
set_global_seed(42)

NB_CONFIG = {
    "optimization_stage": 2,
    "encoding_smoothing": 15.0,
    "variance_candidates": (0.03, 0.1, 0.3, 1.0, 3.0),
    "regularization_candidates": (0.001, 0.01, 0.1),
    "smoothing_candidates": (1.0, 3.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0),
    "learning_rate": 0.03,
    "weight_iterations": 200,
    "n_folds": 5,
    "random_state": 42
}

# LOAD DATA
tr = pd.read_csv(DATA_DIR / "train_dtl.csv")
va = pd.read_csv(DATA_DIR / "val_dtl.csv")
te = pd.read_csv(DATA_DIR / "test_dtl.csv")
y_tr = tr["is_canceled"].to_numpy(dtype=int)
y_va = va["is_canceled"].to_numpy(dtype=int)
print("Training samples  :", len(tr))
print("Validation samples:", len(va))
print("Test samples      :", len(te))

# VALIDATION MODEL TRAINING
nb = NaiveBayes(**NB_CONFIG)
nb.fit(tr, y_tr)
nb.summary()

# VALIDATION EVALUATION
pred_va = nb.predict(va)
macro_f1 = (f1_score(y_va, pred_va) + f1_score(1 - y_va, 1 - pred_va)) / 2
tn, fp, fn, tp = confusion_matrix(y_va, pred_va)
print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)
print(f"Accuracy  : {accuracy_score(y_va, pred_va):.6f}")
print(f"Precision : {precision_score(y_va, pred_va):.6f}")
print(f"Recall    : {recall_score(y_va, pred_va):.6f}")
print(f"F1 Binary : {f1_score(y_va, pred_va):.6f}")
print(f"F1 Macro  : {macro_f1:.6f}")
print(f"\nTN={tn} FP={fp} FN={fn} TP={tp}")
print("Custom cost:", calculate_custom_cost(y_va, pred_va))
nb.save(str(MODEL_DIR / "naive_bayes.pkl"))

# FINAL TRAINING
# Combine only labeled data.
# Preserve original row order.
full_train = pd.concat([tr, va], ignore_index=True)
full_y = full_train["is_canceled"].to_numpy(dtype=int)
nb_final = NaiveBayes(**NB_CONFIG)
nb_final.fit(full_train, full_y)
print("\n" + "=" * 60)
print("FINAL MODEL")
print("=" * 60)
nb_final.summary()

# PREDICT TEST DATA
pred_test = nb_final.predict(te).astype(int)
sample = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
submission = pd.DataFrame({
    "id": te["id"],
    "is_canceled": pred_test
})
assert submission["id"].is_unique
assert sample["id"].is_unique
assert len(submission) == len(sample)
assert set(submission["id"]) == set(sample["id"])
assert np.isin(submission["is_canceled"], [0, 1]).all()
submission = submission.set_index("id").loc[sample["id"]].reset_index()
assert list(submission.columns) == list(sample.columns)

# VERIFY HISTORICAL SUBMISSION REPRODUCIBILITY
reference_path = SUBMISSION_DIR / "submission_nb_3.csv"
reference = pd.read_csv(reference_path)
assert reference["id"].is_unique
assert set(reference["id"]) == set(submission["id"])
reference = reference.set_index("id").loc[submission["id"]].reset_index()
reference_pred = reference["is_canceled"].to_numpy(dtype=int)
current_pred = submission["is_canceled"].to_numpy(dtype=int)
matches = np.sum(reference_pred == current_pred)
differences = np.sum(reference_pred != current_pred)
match_rate = matches / len(reference_pred)
print("\n" + "=" * 60)
print("REPRODUCIBILITY VERIFICATION")
print("=" * 60)
print("Matching predictions :", matches)
print("Different predictions:", differences)
print(f"Match rate           : {match_rate:.6%}")
assert differences == 0, (
    "Historical submission was not reproduced exactly. "
    "Check the source code, data, and model configuration."
)

# SAVE VERIFIED FINAL MODEL
model_path = MODEL_DIR / "naive_bayes_E2_final.pkl"
nb_final.save(str(model_path))

# VERIFY MODEL SAVE AND LOAD
nb_loaded = NaiveBayes.load(str(model_path))
np.testing.assert_array_equal(nb_final.predict(te), nb_loaded.predict(te))
np.testing.assert_allclose(nb_final.predict_proba(te), nb_loaded.predict_proba(te))
print("Model save/load verification: PASSED")

# SAVE VERIFIED FINAL SUBMISSION
submission_path = SUBMISSION_DIR / "submission_nb.csv"
submission.to_csv(submission_path, index=False)
sha256 = hashlib.sha256(submission_path.read_bytes()).hexdigest()
print("\n" + "=" * 60)
print("FINAL SUBMISSION")
print("=" * 60)
print("Model path     :", model_path)
print("Submission path:", submission_path)
print("SHA-256        :", sha256)
print("Positive predictions:", submission["is_canceled"].sum())
print("Positive ratio:", submission["is_canceled"].mean())
print(submission.head())
print("\nFinal model successfully reproduced.")