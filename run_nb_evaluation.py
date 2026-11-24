import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.naive_bayes import NaiveBayes


# ==============================================================
# CONFIGURATION
# ==============================================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data" / "processed"

OUTPUT_DIR = ROOT / "output" / "experiments"

TARGET = "is_canceled"

MODEL_NAMES = {
    0: "Original Gaussian Naive Bayes",
    1: "Weighted Gaussian Naive Bayes",
    2: "Advanced Feature Engineering",
    3: "Adaptive Target Encoding",
    4: "Cross-fitted Likelihood",
    5: "Hierarchical Target Encoding",
    6: "Gaussian Mixture Naive Bayes",
    7: "Robust Weighted Gaussian NB",
    8: "Direct Likelihood Ratio NB"
}


# ==============================================================
# COMMAND LINE ARGUMENTS
# ==============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Naive Bayes model evaluation"
    )

    parser.add_argument(
        "--stages",
        nargs="+",
        type=int,
        default=[3, 7, 8],
        choices=range(9)
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=3
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42]
    )

    parser.add_argument(
        "--inner-folds",
        type=int,
        default=3
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=200
    )

    parser.add_argument(
        "--baseline",
        type=int,
        default=3
    )

    return parser.parse_args()


# ==============================================================
# LOAD LABELED DATA
# ==============================================================

def load_dataset():

    train = pd.read_csv(
        DATA_DIR / "train_dtl.csv"
    )

    validation = pd.read_csv(
        DATA_DIR / "val_dtl.csv"
    )

    # Combine labeled data only.
    # Kaggle test data is NEVER included.

    data = pd.concat(
        [train, validation],
        ignore_index=True
    )

    y = data[TARGET].to_numpy(dtype=int)

    X = data.drop(
        columns=[TARGET]
    )

    if not np.isin(y, [0, 1]).all():
        raise ValueError(
            "Target must contain binary labels."
        )

    return X, y


# ==============================================================
# STRATIFIED CROSS-VALIDATION FROM SCRATCH
# ==============================================================

def stratified_folds(y, n_folds, seed):

    if n_folds < 2:
        raise ValueError(
            "n_folds must be at least 2."
        )

    rng = np.random.default_rng(seed)

    partitions = []

    for c in (0, 1):

        indices = np.flatnonzero(
            y == c
        )

        if len(indices) < n_folds:
            raise ValueError(
                "Insufficient samples in one class."
            )

        indices = rng.permutation(
            indices
        )

        partitions.append(
            np.array_split(
                indices,
                n_folds
            )
        )

    for fold in range(n_folds):

        validation_indices = np.concatenate([
            partition[fold]
            for partition in partitions
        ])

        mask = np.ones(
            len(y),
            dtype=bool
        )

        mask[validation_indices] = False

        training_indices = np.flatnonzero(
            mask
        )

        yield (
            training_indices,
            validation_indices
        )


# ==============================================================
# MODEL CONFIGURATION
# ==============================================================

def build_model_config(stage, args):

    return {
        "optimization_stage": stage,

        "encoding_smoothing": 15.0,

        "smoothing_candidates": (
            1.0, 3.0, 5.0, 10.0,
            15.0, 30.0, 60.0, 120.0
        ),

        "variance_candidates": (
            0.03, 0.1, 0.3, 1.0, 3.0
        ),

        "regularization_candidates": (
            0.001, 0.01, 0.1
        ),

        "calibration_folds": args.inner_folds,

        "n_folds": args.inner_folds,

        "learning_rate": 0.03,

        "weight_iterations": args.iterations,

        "min_feature_coverage": 0.20,

        "threshold_tolerance": 0.0005,

        "robust_configs": None,

        "random_state": 42,

        "verbose": False
    }


# ==============================================================
# CLASSIFICATION METRICS FROM SCRATCH
# ==============================================================

def calculate_metrics(y, predictions, probabilities):

    tn = int(np.sum(
        (y == 0) & (predictions == 0)
    ))

    fp = int(np.sum(
        (y == 0) & (predictions == 1)
    ))

    fn = int(np.sum(
        (y == 1) & (predictions == 0)
    ))

    tp = int(np.sum(
        (y == 1) & (predictions == 1)
    ))

    accuracy = np.mean(
        predictions == y
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1_positive = (
        2 * tp / (2 * tp + fp + fn)
        if 2 * tp + fp + fn > 0
        else 0.0
    )

    f1_negative = (
        2 * tn / (2 * tn + fp + fn)
        if 2 * tn + fp + fn > 0
        else 0.0
    )

    f1_macro = (
        f1_positive + f1_negative
    ) / 2

    brier_score = np.mean(
        (probabilities - y) ** 2
    )

    return {
        "accuracy": float(accuracy),

        "precision": float(precision),

        "recall": float(recall),

        "f1_binary": float(f1_positive),

        "f1_macro": float(f1_macro),

        "brier_score": float(brier_score),

        "tn": tn,

        "fp": fp,

        "fn": fn,

        "tp": tp,

        "custom_cost": 2 * fp + fn,

        "positive_ratio": float(
            predictions.mean()
        )
    }


# ==============================================================
# REPEATED NESTED CROSS-VALIDATION
# ==============================================================

def evaluate_models(X, y, args):

    results = []

    total_runs = (
        len(args.seeds)
        * args.folds
        * len(args.stages)
    )

    current_run = 0

    for seed in args.seeds:

        folds = stratified_folds(
            y,
            args.folds,
            seed
        )

        for fold, (ti, vi) in enumerate(
            folds,
            start=1
        ):

            X_train = X.iloc[ti].reset_index(
                drop=True
            )

            X_val = X.iloc[vi].reset_index(
                drop=True
            )

            y_train = y[ti]

            y_val = y[vi]

            for stage in args.stages:

                current_run += 1

                print(
                    "\n" + "=" * 70,
                    flush=True
                )

                print(
                    f"Run {current_run}/{total_runs} | "
                    f"Seed={seed} | "
                    f"Fold={fold}/{args.folds} | "
                    f"E{stage}",
                    flush=True
                )

                print(
                    MODEL_NAMES[stage],
                    flush=True
                )

                config = build_model_config(
                    stage,
                    args
                )

                model = NaiveBayes(
                    **config
                )

                # ----------------------------------------------
                # INNER MODEL TRAINING AND OPTIMIZATION
                # ----------------------------------------------

                start = time.perf_counter()

                model.fit(
                    X_train,
                    y_train
                )

                training_time = (
                    time.perf_counter()
                    - start
                )

                # ----------------------------------------------
                # OUTER HELD-OUT EVALUATION
                # ----------------------------------------------

                predictions = model.predict(
                    X_val
                )

                probabilities = model.predict_proba(
                    X_val
                )

                if not np.isfinite(probabilities).all():
                    raise FloatingPointError(
                        "Invalid probabilities"
                    )

                if not np.allclose(
                    probabilities.sum(axis=1),
                    1.0
                ):
                    raise ValueError(
                        "Probabilities do not sum to one."
                    )

                metrics = calculate_metrics(
                    y_val,
                    predictions,
                    probabilities[:, 1]
                )

                results.append({
                    "seed": seed,

                    "fold": fold,

                    "stage": stage,

                    "model": MODEL_NAMES[stage],

                    "n_train": len(ti),

                    "n_validation": len(vi),

                    **metrics,

                    "active_features": int(
                        np.sum(
                            model.feature_weights_ > 1e-8
                        )
                    ),

                    "threshold": float(
                        model.threshold_
                    ),

                    "training_time": training_time,

                    "selected_config": str(
                        getattr(
                            model,
                            "robust_config_",
                            None
                        )
                    )
                })

                # Save after each run so completed
                # results survive interruptions.

                pd.DataFrame(results).to_csv(
                    OUTPUT_DIR / "nb_cv_results.csv",
                    index=False
                )

                print(
                    f"Accuracy : "
                    f"{metrics['accuracy']:.6f}"
                )

                print(
                    f"F1 Binary: "
                    f"{metrics['f1_binary']:.6f}"
                )

                print(
                    f"F1 Macro : "
                    f"{metrics['f1_macro']:.6f}"
                )

                print(
                    f"Brier    : "
                    f"{metrics['brier_score']:.6f}"
                )

                print(
                    f"Time     : "
                    f"{training_time:.2f} seconds"
                )

    return pd.DataFrame(results)


# ==============================================================
# AGGREGATE RESULTS
# ==============================================================

def summarize_results(results):

    summary = (
        results
        .groupby(
            ["stage", "model"]
        )
        .agg(
            evaluations=("accuracy", "count"),

            accuracy_mean=("accuracy", "mean"),

            accuracy_std=("accuracy", "std"),

            accuracy_min=("accuracy", "min"),

            accuracy_max=("accuracy", "max"),

            f1_binary_mean=("f1_binary", "mean"),

            f1_macro_mean=("f1_macro", "mean"),

            brier_mean=("brier_score", "mean"),

            active_features_mean=(
                "active_features",
                "mean"
            ),

            training_time_mean=(
                "training_time",
                "mean"
            )
        )
        .reset_index()
        .sort_values(
            "accuracy_mean",
            ascending=False
        )
    )

    summary.to_csv(
        OUTPUT_DIR / "nb_cv_summary.csv",
        index=False
    )

    return summary


# ==============================================================
# PAIRED COMPARISON AGAINST BASELINE
# ==============================================================

def compare_against_baseline(
    results,
    baseline
):

    paired = results.pivot(
        index=["seed", "fold"],
        columns="stage",
        values="accuracy"
    )

    if baseline not in paired.columns:

        print(
            "Baseline was not evaluated. "
            "Skipping paired comparison."
        )

        return pd.DataFrame()

    comparisons = []

    for stage in paired.columns:

        if stage == baseline:
            continue

        differences = (
            paired[stage]
            - paired[baseline]
        ).dropna()

        comparisons.append({
            "stage": int(stage),

            "baseline": baseline,

            "mean_gain": float(
                differences.mean()
            ),

            "std_gain": float(
                differences.std()
            ),

            "worst_gain": float(
                differences.min()
            ),

            "best_gain": float(
                differences.max()
            ),

            "wins": int(
                (differences > 0).sum()
            ),

            "ties": int(
                (differences == 0).sum()
            ),

            "losses": int(
                (differences < 0).sum()
            ),

            "total_folds": len(differences),

            "win_rate": float(
                (differences > 0).mean()
            )
        })

    comparison = pd.DataFrame(comparisons)

    comparison.to_csv(
        OUTPUT_DIR / "nb_cv_paired.csv",
        index=False
    )

    return comparison


# ==============================================================
# MODEL SELECTION
# ==============================================================

def select_best_model(
    summary,
    comparison,
    args
):

    # Rank based on mean outer validation accuracy.

    best = summary.iloc[0]

    best_stage = int(
        best["stage"]
    )

    # Conservative model selection:
    # retain baseline when the improvement is too small
    # or inconsistent across identical outer folds.
    #
    # This is a practical heuristic, not a statistical test.

    minimum_gain = 0.001
    minimum_win_rate = 0.60

    baseline_rows = summary[
        summary["stage"] == args.baseline
    ]

    if (
        not baseline_rows.empty
        and best_stage != args.baseline
        and not comparison.empty
    ):

        row = comparison[
            comparison["stage"] == best_stage
        ].iloc[0]

        if (
            row["mean_gain"] < minimum_gain
            or row["win_rate"] < minimum_win_rate
        ):

            print(
                "\nCandidate improvement is "
                "too small or inconsistent."
            )

            print(
                "Retaining baseline model."
            )

            best = baseline_rows.iloc[0]

            best_stage = args.baseline

    config = build_model_config(
        best_stage,
        args
    )

    output = {
        "selected_stage": best_stage,

        "model_name": MODEL_NAMES[best_stage],

        "mean_cv_accuracy": float(
            best["accuracy_mean"]
        ),

        "std_cv_accuracy": float(
            best["accuracy_std"]
        ),

        "mean_f1_macro": float(
            best["f1_macro_mean"]
        ),

        "outer_folds": args.folds,

        "seeds": args.seeds,

        "model_config": config
    }

    with open(
        OUTPUT_DIR / "nb_best_config.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4
        )

    return output


# ==============================================================
# MAIN PROGRAM
# ==============================================================

def main():

    args = parse_arguments()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    X, y = load_dataset()

    print("=" * 70)

    print("NAIVE BAYES MODEL EVALUATION")

    print("=" * 70)

    print("Total samples :", len(X))

    print("Features      :", X.shape[1])

    print("Stages        :", args.stages)

    print("Outer folds   :", args.folds)

    print("Inner folds   :", args.inner_folds)

    print("Seeds         :", args.seeds)

    print("Total runs    :",
          len(args.seeds)
          * args.folds
          * len(args.stages))

    start = time.perf_counter()

    results = evaluate_models(
        X,
        y,
        args
    )

    summary = summarize_results(
        results
    )

    comparison = compare_against_baseline(
        results,
        args.baseline
    )

    best_model = select_best_model(
        summary,
        comparison,
        args
    )

    print("\n" + "=" * 70)

    print("CROSS-VALIDATION SUMMARY")

    print("=" * 70)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print("\nPAIRED COMPARISON")

    print(
        comparison.to_string(
            index=False
        )
    )

    print("\nSELECTED MODEL")

    print(
        json.dumps(
            best_model,
            indent=4
        )
    )

    print(
        "\nTotal evaluation time:",
        time.perf_counter() - start,
        "seconds"
    )


if __name__ == "__main__":

    main()
