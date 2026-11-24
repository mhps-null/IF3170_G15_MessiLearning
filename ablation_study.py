import time
import numpy as np
import pandas as pd

from src.naive_bayes import NaiveBayes

from src.utils import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ==========================================================
# LOAD DATA
# ==========================================================

tr = pd.read_csv(
    "data/processed/train_dtl.csv"
)

va = pd.read_csv(
    "data/processed/val_dtl.csv"
)

y_tr = tr["is_canceled"].to_numpy(dtype=int)

y_va = va["is_canceled"].to_numpy(dtype=int)

# ==========================================================
# EXPERIMENT CONFIGURATIONS
# ==========================================================

experiments = {
    3: "Optimized Weighted Gaussian NB",
    7: "Robust Weighted Gaussian NB",
    8: "Direct Likelihood Ratio NB"
}

results = []
models = {}

for stage, name in experiments.items():

    print("\n" + "=" * 70)

    print(f"E{stage}: {name}")

    print("=" * 70)

    nb = NaiveBayes(
        optimization_stage=stage,

        encoding_smoothing=15.0,

        smoothing_candidates=(
            1.0, 3.0, 5.0, 10.0,
            15.0, 30.0, 60.0, 120.0
        ),

        variance_candidates=(
            0.03, 0.1, 0.3, 1.0, 3.0
        ),

        regularization_candidates=(
            0.001, 0.01, 0.1
        ),

        calibration_folds=3,

        min_feature_coverage=0.20,

        threshold_tolerance=0.0005,

        learning_rate=0.03,

        weight_iterations=200,

        n_folds=3,

        random_state=42,

        verbose=True
    )

    start = time.perf_counter()

    nb.fit(tr, y_tr)

    training_time = (
        time.perf_counter() - start
    )

    pred = nb.predict(va)

    proba = nb.predict_proba(va)

    # Inference sanity checks.

    assert np.isfinite(proba).all()

    assert np.allclose(
        proba.sum(axis=1),
        1.0
    )

    assert np.all(
        np.isin(pred, [0, 1])
    )

    accuracy = accuracy_score(
        y_va,
        pred
    )

    f1_binary = f1_score(
        y_va,
        pred
    )

    f1_macro = (
        f1_binary
        + f1_score(
            1 - y_va,
            1 - pred
        )
    ) / 2

    tn, fp, fn, tp = confusion_matrix(
        y_va,
        pred
    )

    results.append({

        "Stage": f"E{stage}",

        "Model": name,

        "Accuracy": accuracy,

        "F1 Binary": f1_binary,

        "F1 Macro": f1_macro,

        "Precision": precision_score(
            y_va,
            pred
        ),

        "Recall": recall_score(
            y_va,
            pred
        ),

        "Brier Score": np.mean(
            (proba[:, 1] - y_va) ** 2
        ),

        "Active Features": int(
            np.sum(
                nb.feature_weights_ > 1e-8
            )
        ),

        "Threshold": nb.threshold_,

        "Predicted Positive Ratio": pred.mean(),

        "Training Time": training_time,

        "TN": tn,

        "FP": fp,

        "FN": fn,

        "TP": tp
    })

    models[stage] = nb

    nb.summary()

    print(f"\nAccuracy : {accuracy:.6f}")
    print(f"F1 Binary: {f1_binary:.6f}")
    print(f"F1 Macro : {f1_macro:.6f}")

# ==========================================================
# DISPLAY RESULTS
# ==========================================================

results_df = pd.DataFrame(results)

results_df["Accuracy Gain"] = (
    results_df["Accuracy"]
    - results_df["Accuracy"].iloc[0]
)

print("\n" + "=" * 70)

print("FINAL EXPERIMENT RESULTS")

print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)
