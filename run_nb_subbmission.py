import numpy as np
import pandas as pd

from src.naive_bayes import NaiveBayes

# Load original processed datasets
tr = pd.read_csv("data/processed/train_dtl.csv")
va = pd.read_csv("data/processed/val_dtl.csv")
te = pd.read_csv("data/processed/test_dtl.csv")

full_train = pd.concat(
    [tr, va],
    ignore_index=True
)

y = full_train["is_canceled"].to_numpy(dtype=int)

nb = NaiveBayes(
    optimization_stage=3,
    encoding_smoothing=15.0,
    variance_candidates=(0.03, 0.1, 0.3, 1.0, 3.0),
    regularization_candidates=(0.001, 0.01, 0.1),
    smoothing_candidates=(
        1.0, 3.0, 5.0, 10.0,
        15.0, 30.0, 60.0, 120.0
    ),
    learning_rate=0.03,
    weight_iterations=200,
    n_folds=5,
    random_state=42
)

nb.fit(full_train, y)

# Generate predictions
submission = pd.DataFrame({
    "id": te["id"],
    "is_canceled": nb.predict(te).astype(int)
})

submission.to_csv(
    "output/submissions/submission_nb.csv",
    index=False
)

nb.save(
    "output/models/naive_bayes.pkl"
)
