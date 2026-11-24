import json
import numpy as np
import pandas as pd

from pathlib import Path
from src.naive_bayes import NaiveBayes


ROOT = Path(__file__).resolve().parent

# ==============================================================
# LOAD BEST MODEL CONFIGURATION
# ==============================================================

with open(
    ROOT / "output/experiments/nb_best_config.json",
    "r",
    encoding="utf-8"
) as file:

    experiment = json.load(file)

config = experiment["model_config"]

config["verbose"] = True

print(
    "Selected stage:",
    experiment["selected_stage"]
)

print(
    "Mean CV accuracy:",
    experiment["mean_cv_accuracy"]
)

# ==============================================================
# LOAD DATA
# ==============================================================

tr = pd.read_csv(
    ROOT / "data/processed/train_dtl.csv"
)

va = pd.read_csv(
    ROOT / "data/processed/val_dtl.csv"
)

te = pd.read_csv(
    ROOT / "data/processed/test_dtl.csv"
)

# ==============================================================
# COMBINE ALL LABELED DATA
# ==============================================================

full_train = pd.concat(
    [tr, va],
    ignore_index=True
)

y_train = full_train[
    "is_canceled"
].to_numpy(dtype=int)

X_train = full_train.drop(
    columns=["is_canceled"]
)

# ==============================================================
# REFIT SELECTED MODEL
# ==============================================================

nb = NaiveBayes(
    **config
)

nb.fit(
    X_train,
    y_train
)

nb.summary()

# ==============================================================
# SAVE FINAL MODEL
# ==============================================================

stage = nb.optimization_stage

nb.save(
    str(
        ROOT
        / "output/models"
        / f"naive_bayes_E{stage}_final.pkl"
    )
)

# ==============================================================
# GENERATE PREDICTIONS
# ==============================================================

predictions = nb.predict(te)

sample = pd.read_csv(
    ROOT / "data/raw/sample_submission.csv"
)

submission = pd.DataFrame({
    "id": te["id"],
    "is_canceled": predictions.astype(int)
})

assert submission["id"].is_unique
assert sample["id"].is_unique
assert len(submission) == len(sample)

assert set(submission["id"]) == set(sample["id"])

submission = (
    submission
    .set_index("id")
    .loc[sample["id"]]
    .reset_index()
)

assert list(submission.columns) == list(sample.columns)

assert np.isin(
    submission["is_canceled"],
    [0, 1]
).all()

# ==============================================================
# SAVE SUBMISSION
# ==============================================================

output_dir = ROOT / "output/submissions"

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

output_path = (
    output_dir
    / f"submission_nb_E{stage}_final.csv"
)

submission.to_csv(
    output_path,
    index=False
)

print(
    "\nSubmission saved:",
    output_path
)

print(submission.head())

print(
    "Positive prediction ratio:",
    submission["is_canceled"].mean()
)
