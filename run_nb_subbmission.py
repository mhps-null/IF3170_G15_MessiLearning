import os
import numpy as np
import pandas as pd

from src.naive_bayes import NaiveBayes
from src.naive_bayes_preprocessing import DiscreteNBPreprocessor
from src.utils import (accuracy_score, confusion_matrix, f1_score,
                       precision_score, recall_score, calculate_custom_cost, set_global_seed)

set_global_seed()

# 1. Data
tr = pd.read_csv("data/processed/train_dtl.csv")
va = pd.read_csv("data/processed/val_dtl.csv")
te = pd.read_csv("data/processed/test_dtl.csv")
y_tr = tr["is_canceled"].to_numpy(dtype=int)
y_va = va["is_canceled"].to_numpy(dtype=int)

# 2. Preprocessing: fit HANYA pada train
pre = DiscreteNBPreprocessor(n_bins=5)
X_tr = pre.fit_transform(tr)
X_va = pre.transform(va)
X_te = pre.transform(te)

# 3. Model
nb = NaiveBayes(alpha=1.0)
nb.fit(X_tr, y_tr, n_categories=pre.n_categories_)

# 4. Evaluasi pada validation (ambang standar 0.5)
def macro_f1(y_true, y_pred):
    return (f1_score(y_true, y_pred) + f1_score(1 - y_true, 1 - y_pred)) / 2

pred_va = nb.predict(X_va)
tn, fp, fn, tp = confusion_matrix(y_va, pred_va)
print(f"Val  acc={accuracy_score(y_va, pred_va):.4f}  prec={precision_score(y_va, pred_va):.4f}  "
      f"rec={recall_score(y_va, pred_va):.4f}  macroF1={macro_f1(y_va, pred_va):.4f}")
print(f"Val  TN={tn} FP={fp} FN={fn} TP={tp}  cost(2FP+FN)={calculate_custom_cost(y_va, pred_va):.0f}")

# 5. Prediksi test -> submission (format = sample_submission.csv: id,is_canceled)
sample = pd.read_csv("data/raw/sample_submission.csv")
sub = pd.DataFrame({"id": te["id"], "is_canceled": nb.predict(X_te).astype(int)})
assert list(sub.columns) == list(sample.columns)
assert len(sub) == len(sample) and set(sub["id"]) == set(sample["id"])
sub = sub.set_index("id").loc[sample["id"]].reset_index()   # samakan urutan baris dengan sample_submission
os.makedirs("output/submissions", exist_ok=True)
sub.to_csv("output/submissions/submission_nb.csv", index=False)
print("Submission:", sub.shape, "| proporsi prediksi 1 =", round(sub["is_canceled"].mean(), 4))
print(sub.head(3).to_string(index=False))