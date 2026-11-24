"""
src/naive_bayes.py

Gaussian Naive Bayes from Scratch
IF3170 - G15 MessiLearning

Features:
    - Integrated preprocessing
    - Missing value handling
    - Smoothed Target Encoding
    - Out-of-Fold Target Encoding
    - Feature interaction
    - Standardization
    - Gaussian Naive Bayes
    - Variance smoothing
    - Automatic feature selection
    - Decision threshold optimization
    - Model serialization
    - Bayesian Network visualization
    - Cost-sensitive classification

Conceptual references:
    https://scikit-learn.org/stable/modules/naive_bayes.html
    https://scikit-learn.org/stable/modules/preprocessing.html
    https://scikit-learn.org/stable/modules/feature_selection.html

No sklearn estimator is used in the from-scratch implementation.
"""

import os
import pickle

import numpy as np
import pandas as pd

from .base import BaseModel


class _GaussianNaiveBayesBase(BaseModel):
    # ==============================================================
    # FEATURE CONFIGURATION
    # ==============================================================

    CATEGORICAL_FEATURES = (
        "booking_channel",
        "special_requests",
        "needs_parking",
        "room_category",
        "meal_package",
        "arrival_year",
        "arrival_month",
        "arrival_day",
        "arrival_day_of_week",
        "num_adults",
        "num_children",
        "total_nights",
        "weekend_nights",
        "weekday_nights",
        "is_returning_guest",
    )

    NUMERICAL_BINS = {
        "lead": (
            "days_before_arrival",
            [
                0, 1, 3, 7, 14, 21, 30,
                45, 60, 90, 120, 150,
                180, 240, 300
            ]
        ),
        "rate": (
            "nightly_rate_idr",
            [
                0, 500000, 1000000, 1200000,
                1400000, 1600000, 1800000,
                2000000, 2200000, 2500000,
                3000000
            ]
        ),
        "history": (
            "prior_completed_bookings",
            [0, 0.0001]
        ),
    }

    FEATURE_INTERACTIONS = (
        ("arrival_year", "arrival_month"),

        ("arrival_year", "arrival_month",
         "booking_channel"),

        ("arrival_year", "arrival_month",
         "arrival_day"),

        ("lead", "booking_channel"),

        ("lead", "special_requests"),

        ("rate", "booking_channel"),

        ("rate", "room_category"),

        ("rate", "arrival_year",
         "arrival_month"),

        ("booking_channel", "special_requests"),

        ("booking_channel", "meal_package"),

        ("booking_channel", "total_nights"),

        ("lead", "arrival_year",
         "arrival_month"),

        ("lead", "rate"),

        ("room_category", "num_adults"),

        ("arrival_year", "arrival_month",
         "lead", "booking_channel"),

        ("arrival_year", "arrival_month",
         "arrival_day", "booking_channel"),

        ("lead", "rate", "booking_channel"),

        ("lead", "booking_channel",
         "special_requests"),

        ("arrival_year", "arrival_month",
         "arrival_day", "lead"),

        ("rate", "booking_channel",
         "special_requests"),
    )

    MISSING_VALUE = "__NA__"

    # ==============================================================
    # INITIALIZATION
    # ==============================================================

    def __init__(
        self,
        encoding_smoothing=15.0,
        variance_candidates=(0.03, 0.1, 0.3, 1.0, 3.0),
        max_features=14,
        min_improvement=0.0003,
        validation_size=0.2,
        n_folds=5,
        random_state=42,
        **kwargs
    ):
        super().__init__(**kwargs)

        if encoding_smoothing <= 0:
            raise ValueError(
                "encoding_smoothing must be positive."
            )

        if not 0 < validation_size < 1:
            raise ValueError(
                "validation_size must be between 0 and 1."
            )

        if n_folds < 2:
            raise ValueError(
                "n_folds must be at least 2."
            )

        if max_features < 1:
            raise ValueError(
                "max_features must be positive."
            )

        if (
            len(variance_candidates) == 0
            or not np.all(np.isfinite(variance_candidates))
            or np.any(np.asarray(variance_candidates) <= 0)
        ):
            raise ValueError(
                "variance_candidates must contain "
                "positive finite values."
            )

        self.encoding_smoothing = float(encoding_smoothing)
        self.variance_candidates = tuple(variance_candidates)
        self.max_features = int(max_features)
        self.min_improvement = float(min_improvement)
        self.validation_size = float(validation_size)
        self.n_folds = int(n_folds)
        self.random_state = int(random_state)

        self.classes_ = np.array([0, 1])

        # Preprocessing parameters
        self.category_maps_ = {}
        self.encoding_prior_ = None
        self.candidate_groups_ = []
        self.selected_groups_ = []
        self.feature_names_ = []

        # Standardization parameters
        self.scaler_mean_ = None
        self.scaler_std_ = None

        # Gaussian parameters
        self.theta_ = None
        self.var_ = None
        self.class_counts_ = None
        self.class_priors_ = None
        self.log_class_priors_ = None

        # Optimization parameters
        self.var_smoothing_ = None
        self.log_lr_threshold_ = None
        self.threshold_ = None
        self.validation_accuracy_ = None
        self.optimization_history_ = []

        # Training representation for sklearn comparison
        self.X_fit_ = None

        self.n_features_in_ = None
        self.is_fitted = False

    # ==============================================================
    # INPUT VALIDATION
    # ==============================================================

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError(
                "Model has not been fitted."
            )

    @staticmethod
    def _validate_dataframe(X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "X must be a pandas DataFrame containing "
                "the original categorical and numerical features."
            )

        if len(X) == 0:
            raise ValueError("X must not be empty.")

        if X.columns.duplicated().any():
            raise ValueError(
                "X contains duplicated column names."
            )

    def _validate_target(self, X, y):

        y = np.asarray(y)

        if y.ndim != 1 or len(y) != len(X):
            raise ValueError(
                "y must be a one-dimensional array "
                "with the same length as X."
            )

        if not np.all(np.isin(y, [0, 1])):
            raise ValueError(
                "Target must contain binary labels {0, 1}."
            )

        y = y.astype(np.int64)

        class_counts = np.bincount(y, minlength=2)

        if np.any(class_counts < self.n_folds + 2):
            raise ValueError(
                "Insufficient samples in one or more classes."
            )

        return y

    # ==============================================================
    # FEATURE ENGINEERING
    # ==============================================================

    def _build_feature_keys(self, X):
        """
        Convert original features into categorical keys.

        ID and target columns are never used.
        No statistics are learned from validation or test data.
        """

        self._validate_dataframe(X)

        X = X.copy()

        # Support raw dataset as well as processed DTL dataset.
        if "total_nights" not in X.columns:
            X["total_nights"] = (
                X["weekend_nights"] + X["weekday_nights"]
            )

        if "arrival_day_of_week" not in X.columns:

            dates = pd.to_datetime(
                {
                    "year": X["arrival_year"],
                    "month": X["arrival_month"],
                    "day": X["arrival_day"],
                },
                errors="coerce"
            )

            X["arrival_day_of_week"] = dates.dt.dayofweek

        keys = {}

        # Categorical and discrete numerical features
        for name in self.CATEGORICAL_FEATURES:

            if name not in X.columns:
                raise ValueError(
                    f"Required feature missing: {name}"
                )

            keys[name] = (
                X[name]
                .astype("string")
                .fillna(self.MISSING_VALUE)
                .to_numpy(dtype=str)
            )

        # Continuous numerical features
        for alias, config in self.NUMERICAL_BINS.items():

            name, edges = config

            if name not in X.columns:
                raise ValueError(
                    f"Required feature missing: {name}"
                )

            values = pd.to_numeric(
                X[name],
                errors="coerce"
            ).to_numpy(dtype=np.float64)

            valid = np.isfinite(values)

            codes = np.full(
                len(X),
                self.MISSING_VALUE,
                dtype=object
            )

            codes[valid] = np.searchsorted(
                np.asarray(edges),
                values[valid],
                side="right"
            ).astype(str)

            keys[alias] = codes.astype(str)

        return keys

    @staticmethod
    def _subset_keys(keys, indices):

        return {
            name: values[indices]
            for name, values in keys.items()
        }

    def _candidate_groups(self, keys):

        individual = [
            (name,) for name in keys
        ]

        return individual + list(
            self.FEATURE_INTERACTIONS
        )

    @staticmethod
    def _group_key(keys, group):
        """
        Example:

        group = ("lead", "booking_channel")

        Each row becomes a tuple:
            ("5", "Online Travel Agent")
        """

        return pd.Series(
            list(
                zip(
                    *(keys[name] for name in group)
                )
            ),
            dtype=object
        )

    # ==============================================================
    # STRATIFIED SPLITTING FROM SCRATCH
    # ==============================================================

    def _train_validation_split(self, y):

        rng = np.random.default_rng(
            self.random_state
        )

        train_indices = []
        val_indices = []

        for c in self.classes_:

            indices = np.flatnonzero(y == c)
            indices = rng.permutation(indices)

            n_val = max(
                1,
                int(
                    round(
                        self.validation_size * len(indices)
                    )
                )
            )

            if len(indices) - n_val < self.n_folds:
                raise ValueError(
                    "Insufficient training samples "
                    "for out-of-fold encoding."
                )

            val_indices.extend(indices[:n_val])
            train_indices.extend(indices[n_val:])

        return (
            np.asarray(train_indices, dtype=np.int64),
            np.asarray(val_indices, dtype=np.int64)
        )

    def _stratified_folds(self, y):

        rng = np.random.default_rng(
            self.random_state
        )

        class_partitions = []

        for c in self.classes_:

            indices = np.flatnonzero(y == c)

            indices = rng.permutation(indices)

            partitions = np.array_split(
                indices,
                self.n_folds
            )

            class_partitions.append(partitions)

        for fold in range(self.n_folds):

            holdout = np.concatenate([
                partitions[fold]
                for partitions in class_partitions
            ])

            mask = np.ones(
                len(y),
                dtype=bool
            )

            mask[holdout] = False

            train = np.flatnonzero(mask)

            yield train, holdout

    # ==============================================================
    # SMOOTHED TARGET ENCODING
    # ==============================================================

    def _learn_category_map(self, keys, y):
        """
        Smoothed target encoding:

            encoded(v) =
                (sum(y | v) + alpha * prior)
                / (count(v) + alpha)

        Rare categories are shrunk toward the global
        cancellation probability.
        """

        prior = float(np.mean(y))

        frame = pd.DataFrame({
            "key": keys.to_numpy(),
            "target": y
        })

        stats = (
            frame
            .groupby("key")["target"]
            .agg(["sum", "count"])
        )

        mapping = (
            stats["sum"]
            + self.encoding_smoothing * prior
        ) / (
            stats["count"]
            + self.encoding_smoothing
        )

        return mapping, prior

    @staticmethod
    def _apply_category_map(keys, mapping, prior):

        return (
            keys
            .map(mapping)
            .fillna(prior)
            .to_numpy(dtype=np.float64)
        )

    def _fit_encoder(self, keys, y, groups):
        """
        Produce out-of-fold training representation.

        Every training row is encoded using a mapping
        that does not contain its own target value.

        A separate full-training mapping is saved for
        future validation and test predictions.
        """

        n_samples = len(y)
        n_features = len(groups)

        encoded = np.empty(
            (n_samples, n_features),
            dtype=np.float64
        )

        mappings = {}

        folds = list(
            self._stratified_folds(y)
        )

        for j, group in enumerate(groups):

            group_keys = self._group_key(
                keys,
                group
            )

            # Out-of-fold encoding
            for train_idx, holdout_idx in folds:

                mapping, prior = self._learn_category_map(
                    group_keys.iloc[train_idx],
                    y[train_idx]
                )

                encoded[holdout_idx, j] = (
                    self._apply_category_map(
                        group_keys.iloc[holdout_idx],
                        mapping,
                        prior
                    )
                )

            # Full-training mapping for inference
            full_mapping, full_prior = (
                self._learn_category_map(
                    group_keys,
                    y
                )
            )

            mappings[group] = full_mapping

        return encoded, mappings, float(np.mean(y))

    def _transform_encoder(
        self,
        keys,
        groups,
        mappings,
        prior
    ):

        n_samples = len(
            next(iter(keys.values()))
        )

        encoded = np.empty(
            (n_samples, len(groups)),
            dtype=np.float64
        )

        for j, group in enumerate(groups):

            group_keys = self._group_key(
                keys,
                group
            )

            encoded[:, j] = (
                self._apply_category_map(
                    group_keys,
                    mappings[group],
                    prior
                )
            )

        return encoded

    # ==============================================================
    # STANDARDIZATION
    # ==============================================================

    @staticmethod
    def _fit_scaler(X):

        mean = np.mean(X, axis=0)

        std = np.std(X, axis=0)

        std = np.maximum(std, 1e-8)

        return mean, std

    @staticmethod
    def _transform_scaler(X, mean, std):

        return (X - mean) / std

    # ==============================================================
    # GAUSSIAN NAIVE BAYES FROM SCRATCH
    # ==============================================================

    @staticmethod
    def _fit_gaussian(X, y, var_smoothing):
        """
        Estimate class-conditional Gaussian parameters.

        Mean:
            mu[c, j] = mean(X[y == c, j])

        Variance:
            var[c, j] = variance(X[y == c, j]) + epsilon

        Class prior:
            prior[c] = count(y == c) / len(y)
        """

        means = []
        variances = []
        counts = []

        for c in (0, 1):

            X_c = X[y == c]

            means.append(
                np.mean(X_c, axis=0)
            )

            variances.append(
                np.var(X_c, axis=0)
                + var_smoothing
            )

            counts.append(len(X_c))

        means = np.asarray(
            means,
            dtype=np.float64
        )

        variances = np.asarray(
            variances,
            dtype=np.float64
        )

        counts = np.asarray(
            counts,
            dtype=np.int64
        )

        priors = counts / len(y)

        return means, variances, counts, priors

    @staticmethod
    def _gaussian_log_pdf(X, mean, variance):
        """
        Gaussian log probability density function:

        log p(x|c) =
            -0.5 * (
                log(2*pi*variance)
                + (x-mean)^2 / variance
            )
        """

        return -0.5 * (
            np.log(2.0 * np.pi * variance)
            + ((X - mean) ** 2) / variance
        )

    @classmethod
    def _feature_log_likelihood_ratio(
        cls,
        X,
        means,
        variances
    ):
        """
        Per-feature log-likelihood ratio:

        log p(x_j | y=1) - log p(x_j | y=0)

        The common log(2*pi) term cancels.
        """

        log_prob_0 = -0.5 * (
            np.log(variances[0])
            + (X - means[0]) ** 2 / variances[0]
        )

        log_prob_1 = -0.5 * (
            np.log(variances[1])
            + (X - means[1]) ** 2 / variances[1]
        )

        return log_prob_1 - log_prob_0

    # ==============================================================
    # DECISION THRESHOLD OPTIMIZATION
    # ==============================================================

    @staticmethod
    def _find_best_threshold(scores, y):

        candidates = np.linspace(
            -5.0,
            5.0,
            101
        )

        accuracies = np.array([
            np.mean(
                (scores >= threshold) == y
            )
            for threshold in candidates
        ])

        best_idx = int(
            np.argmax(accuracies)
        )

        return (
            float(accuracies[best_idx]),
            float(candidates[best_idx])
        )

    # ==============================================================
    # AUTOMATIC FEATURE SELECTION
    # ==============================================================

    def _forward_feature_selection(
        self,
        likelihood_ratios,
        y
    ):
        """
        Greedy Sequential Forward Selection.

        At each iteration, add the feature that
        produces the highest internal validation accuracy.

        Stop when improvement is insignificant.
        """

        n_features = likelihood_ratios.shape[1]

        selected = []
        current_accuracy = 0.0

        max_features = min(
            self.max_features,
            n_features
        )

        for _ in range(max_features):

            best_candidate = None

            for j in range(n_features):

                if j in selected:
                    continue

                candidate = selected + [j]

                scores = (
                    likelihood_ratios[:, candidate]
                    .sum(axis=1)
                )

                accuracy, _ = (
                    self._find_best_threshold(
                        scores,
                        y
                    )
                )

                result = (accuracy, j)

                if (
                    best_candidate is None
                    or result > best_candidate
                ):
                    best_candidate = result

            if best_candidate is None:
                break

            accuracy, feature_idx = best_candidate

            if (
                accuracy
                <= current_accuracy + self.min_improvement
            ):
                break

            selected.append(feature_idx)

            current_accuracy = accuracy

        if not selected:
            raise RuntimeError(
                "Feature selection failed."
            )

        scores = (
            likelihood_ratios[:, selected]
            .sum(axis=1)
        )

        accuracy, threshold = (
            self._find_best_threshold(
                scores,
                y
            )
        )

        return selected, accuracy, threshold

    # ==============================================================
    # HYPERPARAMETER OPTIMIZATION
    # ==============================================================

    def _optimize(self, keys, y, groups):
        """
        Select:
            - Feature subset
            - Variance smoothing
            - Classification threshold

        Only the internal training-validation split
        is used during optimization.
        """

        train_idx, val_idx = (
            self._train_validation_split(y)
        )

        keys_train = self._subset_keys(
            keys,
            train_idx
        )

        keys_val = self._subset_keys(
            keys,
            val_idx
        )

        y_train = y[train_idx]
        y_val = y[val_idx]

        # Fit encoder using internal training only.
        X_train, mappings, prior = self._fit_encoder(
            keys_train,
            y_train,
            groups
        )

        X_val = self._transform_encoder(
            keys_val,
            groups,
            mappings,
            prior
        )

        # Fit scaler using internal training only.
        mean, std = self._fit_scaler(
            X_train
        )

        X_train = self._transform_scaler(
            X_train,
            mean,
            std
        )

        X_val = self._transform_scaler(
            X_val,
            mean,
            std
        )

        best = None

        self.optimization_history_ = []

        # Grid search over variance smoothing values.
        for smoothing in self.variance_candidates:

            means, variances, _, _ = (
                self._fit_gaussian(
                    X_train,
                    y_train,
                    smoothing
                )
            )

            ratios = (
                self._feature_log_likelihood_ratio(
                    X_val,
                    means,
                    variances
                )
            )

            selected, accuracy, threshold = (
                self._forward_feature_selection(
                    ratios,
                    y_val
                )
            )

            result = {
                "accuracy": accuracy,
                "var_smoothing": float(smoothing),
                "threshold": threshold,
                "selected_indices": selected,
                "selected_groups": [
                    groups[j] for j in selected
                ],
            }

            self.optimization_history_.append(result)

            if (
                best is None
                or accuracy > best["accuracy"]
            ):
                best = result

        return best

    # ==============================================================
    # PUBLIC FIT API
    # ==============================================================

    def fit(self, X, y=None):
        """
        Train preprocessing and Gaussian NB in one call.

        Parameters
        ----------
        X : pandas.DataFrame
            Original or DTL-processed dataset.

        y : array-like, optional
            Binary target. If omitted, use the
            'is_canceled' column from X.
        """

        self.is_fitted = False

        self._validate_dataframe(X)

        if y is None:

            if "is_canceled" not in X.columns:
                raise ValueError(
                    "Provide y or an is_canceled column."
                )

            y = X["is_canceled"].to_numpy()

        y = self._validate_target(X, y)

        # 1. Feature engineering
        keys = self._build_feature_keys(X)

        groups = self._candidate_groups(keys)

        self.candidate_groups_ = groups

        # 2. Internal model optimization
        best = self._optimize(
            keys,
            y,
            groups
        )

        self.selected_groups_ = (
            best["selected_groups"]
        )

        self.var_smoothing_ = (
            best["var_smoothing"]
        )

        self.log_lr_threshold_ = (
            best["threshold"]
        )

        self.validation_accuracy_ = (
            best["accuracy"]
        )

        self.feature_names_ = [
            " × ".join(group)
            for group in self.selected_groups_
        ]

        # 3. Refit selected encoder on all supplied
        # training data using out-of-fold encoding.
        encoded, mappings, prior = (
            self._fit_encoder(
                keys,
                y,
                self.selected_groups_
            )
        )

        self.category_maps_ = mappings
        self.encoding_prior_ = prior

        # 4. Standardization
        (
            self.scaler_mean_,
            self.scaler_std_
        ) = self._fit_scaler(encoded)

        X_train = self._transform_scaler(
            encoded,
            self.scaler_mean_,
            self.scaler_std_
        )

        # 5. Fit final Gaussian Naive Bayes
        (
            self.theta_,
            self.var_,
            self.class_counts_,
            self.class_priors_
        ) = self._fit_gaussian(
            X_train,
            y,
            self.var_smoothing_
        )

        self.log_class_priors_ = np.log(
            self.class_priors_
        )

        # Convert likelihood-ratio threshold into
        # equivalent posterior-probability threshold.
        prior_log_odds = (
            self.log_class_priors_[1]
            - self.log_class_priors_[0]
        )

        posterior_log_odds = (
            self.log_lr_threshold_
            + prior_log_odds
        )

        self.threshold_ = float(
            1.0 / (
                1.0 + np.exp(-posterior_log_odds)
            )
        )

        # Keep exact OOF training features to enable
        # fair comparison with sklearn GaussianNB.
        self.X_fit_ = X_train

        self.n_features_in_ = (
            X_train.shape[1]
        )

        self.is_fitted = True

        return self

    # ==============================================================
    # PUBLIC PREPROCESSING API
    # ==============================================================

    def transform(self, X):
        """
        Apply the fitted preprocessing pipeline.

        No parameters are learned from X.
        """

        self._check_fitted()

        keys = self._build_feature_keys(X)

        encoded = self._transform_encoder(
            keys,
            self.selected_groups_,
            self.category_maps_,
            self.encoding_prior_
        )

        return self._transform_scaler(
            encoded,
            self.scaler_mean_,
            self.scaler_std_
        )

    # ==============================================================
    # INFERENCE
    # ==============================================================

    def _compute_log_likelihood(self, X):

        X_transformed = self.transform(X)

        n_samples = len(X_transformed)

        scores = np.empty(
            (n_samples, 2),
            dtype=np.float64
        )

        for c in self.classes_:

            log_density = self._gaussian_log_pdf(
                X_transformed,
                self.theta_[c],
                self.var_[c]
            )

            scores[:, c] = (
                self.log_class_priors_[c]
                + log_density.sum(axis=1)
            )

        return scores

    def predict_log_proba(self, X):

        scores = self._compute_log_likelihood(X)

        row_max = np.max(
            scores,
            axis=1,
            keepdims=True
        )

        shifted = scores - row_max

        log_sum_exp = np.log(
            np.sum(
                np.exp(shifted),
                axis=1,
                keepdims=True
            )
        )

        return shifted - log_sum_exp

    def predict_proba(self, X):
        """
        Return posterior probabilities:

            column 0 = P(y=0 | X)
            column 1 = P(y=1 | X)
        """

        return np.exp(
            self.predict_log_proba(X)
        )

    def predict(self, X, threshold=None):
        """
        By default, use optimized classification threshold.

        To use the standard Bayesian threshold:
            model.predict(X, threshold=0.5)
        """

        self._check_fitted()

        if threshold is None:

            transformed = self.transform(X)

            ratios = (
                self._feature_log_likelihood_ratio(
                    transformed,
                    self.theta_,
                    self.var_
                )
            )

            return (
                ratios.sum(axis=1)
                >= self.log_lr_threshold_
            ).astype(np.int64)

        if not 0 <= threshold <= 1:
            raise ValueError(
                "threshold must be between 0 and 1."
            )

        probabilities = self.predict_proba(X)

        return (
            probabilities[:, 1] >= threshold
        ).astype(np.int64)

    def score(self, X, y):

        y = np.asarray(y)

        predictions = self.predict(X)

        if y.shape != predictions.shape:
            raise ValueError(
                "Target and predictions must have "
                "the same shape."
            )

        return float(
            np.mean(predictions == y)
        )

    # ==============================================================
    # MODEL INFORMATION
    # ==============================================================

    def get_feature_info(self):

        self._check_fitted()

        return pd.DataFrame({
            "feature": self.feature_names_,
            "mean_class_0": self.theta_[0],
            "mean_class_1": self.theta_[1],
            "variance_class_0": self.var_[0],
            "variance_class_1": self.var_[1],
        })

    def get_optimization_history(self):

        self._check_fitted()

        return pd.DataFrame([
            {
                "var_smoothing": result["var_smoothing"],
                "validation_accuracy": result["accuracy"],
                "log_lr_threshold": result["threshold"],
                "n_features": len(
                    result["selected_groups"]
                ),
            }
            for result in self.optimization_history_
        ])

    def summary(self):

        self._check_fitted()

        print("=" * 60)
        print("GAUSSIAN NAIVE BAYES FROM SCRATCH")
        print("=" * 60)

        print(
            "Candidate features    :",
            len(self.candidate_groups_)
        )

        print(
            "Selected features     :",
            self.n_features_in_
        )

        print(
            "Variance smoothing    :",
            self.var_smoothing_
        )

        print(
            "Encoding smoothing    :",
            self.encoding_smoothing
        )

        print(
            "Decision threshold    :",
            round(self.threshold_, 6)
        )

        print(
            "Internal validation   :",
            f"{self.validation_accuracy_:.4%}"
        )

        print(
            "Class priors          :",
            self.class_priors_
        )

        print("\nSelected features:")

        for i, name in enumerate(
            self.feature_names_,
            start=1
        ):
            print(f"  {i}. {name}")

        print("=" * 60)

    # ==============================================================
    # MODEL SERIALIZATION
    # ==============================================================

    def save(self, filepath):

        self._check_fitted()

        os.makedirs(
            os.path.dirname(filepath) or ".",
            exist_ok=True
        )

        with open(filepath, "wb") as file:

            pickle.dump(
                self,
                file,
                protocol=pickle.HIGHEST_PROTOCOL
            )

        print(
            f"[Model Saved] {filepath}"
        )

    @classmethod
    def load(cls, filepath):

        with open(filepath, "rb") as file:

            model = pickle.load(file)

        if not isinstance(model, cls):
            raise TypeError(
                "Invalid NaiveBayes model."
            )

        model._check_fitted()

        return model

class _WeightedGaussianNaiveBayes(
    _GaussianNaiveBayesBase
):
    """
    Optimized Weighted Gaussian Naive Bayes.

    IF3170 - G15 MessiLearning

    Optimization stages:
        0 : Original Gaussian Naive Bayes
        1 : Weighted Gaussian Naive Bayes
        2 : Advanced Feature Engineering
        3 : Adaptive Target Encoding

    All stages are cumulative.

    References:
        Zaidi et al. (2013).
        Alleviating Naive Bayes Attribute Independence
        Assumption by Attribute Weighting.
        https://jmlr.org/papers/v14/zaidi13a.html

        Kingma and Ba (2015).
        Adam: A Method for Stochastic Optimization.
        https://arxiv.org/abs/1412.6980

        Target Encoder Internal Cross-Fitting:
        https://scikit-learn.org/stable/auto_examples/
        preprocessing/plot_target_encoder_cross_val.html

    All optimization algorithms are implemented from scratch.
    """

    # ==========================================================
    # INITIALIZATION
    # ==========================================================

    def __init__(
        self,
        optimization_stage=3,
        regularization_candidates=(0.001, 0.01, 0.1),
        smoothing_candidates=(
            1.0, 3.0, 5.0, 10.0,
            15.0, 30.0, 60.0, 120.0
        ),
        learning_rate=0.03,
        weight_iterations=200,
        **kwargs
    ):

        super().__init__(**kwargs)

        if optimization_stage not in (0, 1, 2, 3):
            raise ValueError(
                "optimization_stage must be 0, 1, 2, or 3."
            )

        for candidates in (
            regularization_candidates,
            smoothing_candidates
        ):
            values = np.asarray(candidates, dtype=float)

            if (
                values.ndim != 1
                or len(values) == 0
                or not np.all(np.isfinite(values))
                or np.any(values <= 0)
            ):
                raise ValueError(
                    "Candidates must contain positive "
                    "finite values."
                )

        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive and finite."
            )

        if weight_iterations < 1:
            raise ValueError(
                "weight_iterations must be positive."
            )

        self.optimization_stage = optimization_stage

        self.regularization_candidates = tuple(
            regularization_candidates
        )

        self.smoothing_candidates = tuple(
            smoothing_candidates
        )

        self.learning_rate = float(learning_rate)
        self.weight_iterations = int(weight_iterations)

        self.feature_weights_ = None
        self.intercept_ = None
        self.regularization_ = None
        self.feature_smoothing_ = {}

    # ==========================================================
    # STAGE 2: ADVANCED FEATURE ENGINEERING
    # ==========================================================

    def _build_feature_keys(self, X):

        keys = super()._build_feature_keys(X)

        if self.optimization_stage < 2:
            return keys

        # ------------------------------------------------------
        # Helper: categorical representation
        # ------------------------------------------------------

        def add_category(name, values):

            keys[name] = (
                pd.Series(values)
                .astype("string")
                .fillna(self.MISSING_VALUE)
                .to_numpy(dtype=str)
            )

        # ------------------------------------------------------
        # Helper: numerical discretization
        # ------------------------------------------------------

        def add_bins(name, values, edges):

            values = np.asarray(
                values,
                dtype=np.float64
            )

            result = np.full(
                len(values),
                self.MISSING_VALUE,
                dtype=object
            )

            valid = np.isfinite(values)

            result[valid] = np.searchsorted(
                np.asarray(edges),
                values[valid],
                side="right"
            ).astype(str)

            keys[name] = result.astype(str)

        # ------------------------------------------------------
        # Original numerical features
        # ------------------------------------------------------

        lead = X["days_before_arrival"]

        rate = X["nightly_rate_idr"]

        nights = (
            X["weekday_nights"]
            + X["weekend_nights"]
        )

        guests = (
            X["num_adults"]
            + X["num_children"]
        )

        # ------------------------------------------------------
        # Reconstruct arrival and booking dates
        # ------------------------------------------------------

        arrival_date = pd.to_datetime(
            {
                "year": X["arrival_year"],
                "month": X["arrival_month"],
                "day": X["arrival_day"]
            },
            errors="coerce"
        )

        booking_date = (
            arrival_date
            - pd.to_timedelta(
                lead,
                unit="D",
                errors="coerce"
            )
        )

        add_category(
            "arrival_date",
            arrival_date.dt.strftime("%Y-%m-%d")
        )

        add_category(
            "booking_date",
            booking_date.dt.strftime("%Y-%m-%d")
        )

        add_category(
            "booking_month",
            booking_date.dt.strftime("%Y-%m")
        )

        add_category(
            "booking_week",
            booking_date.dt.strftime("%G-%V")
        )

        add_category(
            "booking_dow",
            booking_date.dt.dayofweek
        )

        # ------------------------------------------------------
        # Exact numerical values as categorical keys
        # ------------------------------------------------------

        add_category("lead_exact", lead)

        add_category("rate_exact", rate)

        add_category("guests", guests)

        add_category(
            "prior_cancel",
            X["prior_cancellations"]
        )

        # ------------------------------------------------------
        # Fine-grained numerical binning
        # ------------------------------------------------------

        add_bins(
            "lead_fine",
            lead,
            np.arange(0, 365, 5)
        )

        add_bins(
            "rate_fine",
            rate,
            np.arange(0, 4000001, 50000)
        )

        # ------------------------------------------------------
        # Estimated booking value
        # ------------------------------------------------------

        add_bins(
            "value",
            rate * nights,
            [
                0, 1e6, 2e6, 3e6, 4e6,
                5e6, 7e6, 10e6, 15e6, 20e6
            ]
        )

        # ------------------------------------------------------
        # Price per guest per night
        # ------------------------------------------------------

        add_bins(
            "rate_guest",
            rate / guests.replace(0, np.nan),
            [
                0, 400000, 600000, 800000,
                1000000, 1200000,
                1500000, 2000000
            ]
        )

        # ------------------------------------------------------
        # Weekend stay proportion
        # ------------------------------------------------------

        add_bins(
            "weekend_ratio",
            X["weekend_nights"]
            / nights.replace(0, np.nan),
            [0, 0.01, 0.25, 0.5, 0.75, 1]
        )

        return keys

    # ==========================================================
    # STAGE 2: ADVANCED FEATURE INTERACTION
    # ==========================================================

    def _candidate_groups(self, keys):

        groups = super()._candidate_groups(keys)

        if self.optimization_stage < 2:
            return groups

        derived_features = (
            "arrival_date",
            "booking_date",
            "booking_month",
            "booking_week",
            "lead_exact",
            "lead_fine",
            "rate_exact",
            "rate_fine",
            "value",
            "rate_guest",
            "guests",
            "weekend_ratio",
            "prior_cancel"
        )

        contextual_features = (
            "booking_channel",
            "special_requests",
            "room_category",
            "lead",
            "rate"
        )

        # Automatically generate pairwise interactions.
        for first in derived_features:

            for second in contextual_features:

                groups.append(
                    (first, second)
                )

        # Domain-specific higher-order interactions.
        groups.extend([

            ("arrival_date", "lead_exact"),

            ("arrival_date", "lead_fine"),

            ("booking_date", "arrival_date"),

            (
                "booking_date",
                "booking_channel",
                "special_requests"
            ),

            (
                "booking_month",
                "lead",
                "booking_channel"
            ),

            (
                "arrival_date",
                "booking_channel",
                "special_requests"
            ),

            (
                "lead_fine",
                "booking_channel",
                "special_requests"
            ),

            (
                "rate_fine",
                "booking_channel",
                "special_requests"
            ),

            ("arrival_date", "rate_fine"),

            (
                "booking_week",
                "lead",
                "booking_channel"
            ),

            ("lead_exact", "rate_exact"),

            (
                "rate_exact",
                "room_category",
                "booking_channel"
            ),

            (
                "arrival_date",
                "room_category",
                "booking_channel"
            ),

            ("booking_date", "rate_fine"),

            (
                "rate_fine",
                "arrival_year",
                "arrival_month"
            ),

            (
                "arrival_date",
                "lead",
                "booking_channel"
            )
        ])

        # Remove duplicate groups while preserving order.
        return list(dict.fromkeys(groups))

    # ==========================================================
    # STAGE 3: ADAPTIVE TARGET ENCODING
    # ==========================================================

    def _fit_encoder(self, keys, y, groups):
        """
        Learn target encoding parameters using stratified
        out-of-fold encoding.

        Stage 1 and 2:
            Use the original fixed smoothing parameter.

        Stage 3:
            Select an individual smoothing parameter
            for every feature group using OOF Brier score.
        """

        # Preserve the original baseline exactly.
        if self.optimization_stage == 0:

            result = super()._fit_encoder(
                keys,
                y,
                groups
            )

            self.feature_smoothing_ = {
                group: self.encoding_smoothing
                for group in groups
            }

            return result

        n_samples = len(y)
        n_features = len(groups)

        encoded = np.empty(
            (n_samples, n_features),
            dtype=np.float64
        )

        mappings = {}
        selected_smoothing = {}

        folds = list(
            self._stratified_folds(y)
        )

        global_prior = float(np.mean(y))

        if self.optimization_stage >= 3:

            candidates = np.asarray(
                self.smoothing_candidates,
                dtype=np.float64
            )

        else:

            candidates = np.array(
                [self.encoding_smoothing],
                dtype=np.float64
            )

        for j, group in enumerate(groups):

            group_keys = self._group_key(
                keys,
                group
            )

            # Integer identifiers are used only for
            # efficient grouping, not as model features.
            codes, unique_categories = pd.factorize(
                group_keys,
                sort=False
            )

            n_categories = len(unique_categories)

            # Every column corresponds to one alpha.
            candidate_encoding = np.empty(
                (
                    n_samples,
                    len(candidates)
                ),
                dtype=np.float64
            )

            # --------------------------------------------------
            # Out-of-fold encoding for every alpha
            # --------------------------------------------------

            for train_idx, holdout_idx in folds:

                train_codes = codes[train_idx]

                counts = np.bincount(
                    train_codes,
                    minlength=n_categories
                )

                sums = np.bincount(
                    train_codes,
                    weights=y[train_idx],
                    minlength=n_categories
                )

                fold_prior = float(
                    np.mean(y[train_idx])
                )

                holdout_codes = codes[holdout_idx]

                candidate_encoding[holdout_idx] = (
                    sums[holdout_codes, None]
                    + candidates * fold_prior
                ) / (
                    counts[holdout_codes, None]
                    + candidates
                )

            # --------------------------------------------------
            # Select alpha with minimum OOF Brier score
            # --------------------------------------------------

            brier_scores = np.mean(
                (
                    candidate_encoding
                    - y[:, None]
                ) ** 2,
                axis=0
            )

            best_idx = int(
                np.argmin(brier_scores)
            )

            best_alpha = float(
                candidates[best_idx]
            )

            encoded[:, j] = (
                candidate_encoding[:, best_idx]
            )

            selected_smoothing[group] = best_alpha

            # --------------------------------------------------
            # Refit full-training mapping for inference
            # --------------------------------------------------

            full_counts = np.bincount(
                codes,
                minlength=n_categories
            )

            full_sums = np.bincount(
                codes,
                weights=y,
                minlength=n_categories
            )

            full_encoding = (
                full_sums
                + best_alpha * global_prior
            ) / (
                full_counts
                + best_alpha
            )

            mappings[group] = pd.Series(
                full_encoding,
                index=unique_categories
            )

        self.feature_smoothing_ = selected_smoothing

        return (
            encoded,
            mappings,
            global_prior
        )

    # ==========================================================
    # NUMERICALLY STABLE SIGMOID
    # ==========================================================

    @staticmethod
    def _sigmoid(z):

        return np.exp(
            -np.logaddexp(0.0, -z)
        )

    # ==========================================================
    # STAGE 1: REGULARIZED FEATURE WEIGHTING
    # ==========================================================

    def _learn_feature_weights(
        self,
        likelihood_ratios,
        y,
        class_priors,
        regularization
    ):
        """
        Optimize feature weights using projected Adam
        gradient descent with L2 regularization.

        Model:
            z = intercept + sum(w_j * r_j)

        Objective:
            Binary cross-entropy
            + lambda / 2 * sum(w_j^2)

        Constraint:
            w_j >= 0
        """

        R = np.asarray(
            likelihood_ratios,
            dtype=np.float64
        )

        n_samples, n_features = R.shape

        # Scale optimization coordinates to improve
        # numerical conditioning.
        scale = np.maximum(
            np.sqrt(
                np.mean(R ** 2, axis=0)
            ),
            0.1
        )

        Z = R / scale

        # Initialize weights at zero and the intercept
        # at empirical class prior log-odds.
        prior_log_odds = np.log(
            class_priors[1] / class_priors[0]
        )

        parameters = np.concatenate([
            np.zeros(n_features),
            [prior_log_odds]
        ])

        first_moment = np.zeros_like(parameters)

        second_moment = np.zeros_like(parameters)

        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8

        # ------------------------------------------------------
        # Adam optimization loop
        # ------------------------------------------------------

        for step in range(
            1,
            self.weight_iterations + 1
        ):

            weights_scaled = parameters[:-1]

            intercept = parameters[-1]

            logits = (
                Z @ weights_scaled
                + intercept
            )

            probabilities = self._sigmoid(
                logits
            )

            error = probabilities - y

            # Gradient of BCE + L2.
            # Divide by scale^2 because regularization
            # applies to original feature weights.
            gradient_weights = (
                Z.T @ error / n_samples
                + regularization
                * weights_scaled
                / (scale ** 2)
            )

            gradient_intercept = np.mean(
                error
            )

            gradient = np.concatenate([
                gradient_weights,
                [gradient_intercept]
            ])

            # Adam first moment.
            first_moment = (
                beta1 * first_moment
                + (1.0 - beta1) * gradient
            )

            # Adam second moment.
            second_moment = (
                beta2 * second_moment
                + (1.0 - beta2) * gradient ** 2
            )

            # Bias correction.
            corrected_first = (
                first_moment
                / (1.0 - beta1 ** step)
            )

            corrected_second = (
                second_moment
                / (1.0 - beta2 ** step)
            )

            # Update.
            parameters -= (
                self.learning_rate
                * corrected_first
                / (
                    np.sqrt(corrected_second)
                    + epsilon
                )
            )

            # Project feature weights into nonnegative
            # space. The intercept is unconstrained.
            parameters[:-1] = np.maximum(
                parameters[:-1],
                0.0
            )

        # Convert back to original likelihood ratio space.
        weights = parameters[:-1] / scale

        intercept = float(parameters[-1])

        return weights, intercept

    # ==========================================================
    # WEIGHTED MODEL HYPERPARAMETER OPTIMIZATION
    # ==========================================================

    def _optimize_weighted(self, keys, y, groups):

        train_idx, val_idx = (
            self._train_validation_split(y)
        )

        train_keys = self._subset_keys(
            keys,
            train_idx
        )

        val_keys = self._subset_keys(
            keys,
            val_idx
        )

        y_train = y[train_idx]
        y_val = y[val_idx]

        # Fit all encoding parameters using the
        # internal training subset only.
        X_train, mappings, prior = (
            self._fit_encoder(
                train_keys,
                y_train,
                groups
            )
        )

        X_val = self._transform_encoder(
            val_keys,
            groups,
            mappings,
            prior
        )

        mean, std = self._fit_scaler(X_train)

        X_train = self._transform_scaler(
            X_train,
            mean,
            std
        )

        X_val = self._transform_scaler(
            X_val,
            mean,
            std
        )

        best = None

        self.optimization_history_ = []

        # ------------------------------------------------------
        # Grid search variance smoothing and regularization
        # ------------------------------------------------------

        for smoothing in self.variance_candidates:

            means, variances, _, priors = (
                self._fit_gaussian(
                    X_train,
                    y_train,
                    smoothing
                )
            )

            train_ratios = (
                self._feature_log_likelihood_ratio(
                    X_train,
                    means,
                    variances
                )
            )

            val_ratios = (
                self._feature_log_likelihood_ratio(
                    X_val,
                    means,
                    variances
                )
            )

            for regularization in (
                self.regularization_candidates
            ):

                weights, intercept = (
                    self._learn_feature_weights(
                        train_ratios,
                        y_train,
                        priors,
                        regularization
                    )
                )

                scores = (
                    val_ratios @ weights
                    + intercept
                )

                accuracy, threshold = (
                    self._find_best_threshold(
                        scores,
                        y_val
                    )
                )

                result = {
                    "accuracy": float(accuracy),
                    "var_smoothing": float(smoothing),
                    "regularization": float(
                        regularization
                    ),
                    "threshold": float(threshold),
                    "active_features": int(
                        np.sum(weights > 1e-8)
                    )
                }

                self.optimization_history_.append(
                    result
                )

                if (
                    best is None
                    or accuracy > best["accuracy"]
                ):

                    best = result.copy()

        return best

    # ==========================================================
    # PUBLIC FIT API
    # ==========================================================

    def fit(self, X, y=None):

        # ------------------------------------------------------
        # Stage 0: reproduce the original Gaussian NB
        # ------------------------------------------------------

        if self.optimization_stage == 0:

            super().fit(X, y)

            self.feature_weights_ = np.ones(
                self.n_features_in_
            )

            self.intercept_ = float(
                self.log_class_priors_[1]
                - self.log_class_priors_[0]
            )

            self.regularization_ = None

            return self

        # ------------------------------------------------------
        # Stages 1, 2, and 3
        # ------------------------------------------------------

        self.is_fitted = False

        self._validate_dataframe(X)

        if y is None:

            if "is_canceled" not in X.columns:

                raise ValueError(
                    "Provide y or an is_canceled column."
                )

            y = X["is_canceled"].to_numpy()

        y = self._validate_target(X, y)

        # 1. Integrated feature engineering.
        keys = self._build_feature_keys(X)

        groups = self._candidate_groups(keys)

        self.candidate_groups_ = groups

        # 2. Optimize using internal validation.
        best = self._optimize_weighted(
            keys,
            y,
            groups
        )

        self.var_smoothing_ = best["var_smoothing"]

        self.regularization_ = best["regularization"]

        self.log_lr_threshold_ = best["threshold"]

        self.validation_accuracy_ = best["accuracy"]

        # Unlike stage 0, weighted stages use continuous
        # feature weighting instead of hard forward selection.
        self.selected_groups_ = groups

        self.feature_names_ = [
            " × ".join(group)
            for group in groups
        ]

        # 3. Refit preprocessing on all supplied
        # training samples.
        (
            encoded,
            self.category_maps_,
            self.encoding_prior_
        ) = self._fit_encoder(
            keys,
            y,
            groups
        )

        # 4. Standardization.
        (
            self.scaler_mean_,
            self.scaler_std_
        ) = self._fit_scaler(encoded)

        X_train = self._transform_scaler(
            encoded,
            self.scaler_mean_,
            self.scaler_std_
        )

        # 5. Train Gaussian Naive Bayes parameters.
        (
            self.theta_,
            self.var_,
            self.class_counts_,
            self.class_priors_
        ) = self._fit_gaussian(
            X_train,
            y,
            self.var_smoothing_
        )

        self.log_class_priors_ = np.log(
            self.class_priors_
        )

        # 6. Generate Gaussian likelihood ratios.
        ratios = self._feature_log_likelihood_ratio(
            X_train,
            self.theta_,
            self.var_
        )

        # 7. Learn final feature weights.
        (
            self.feature_weights_,
            self.intercept_
        ) = self._learn_feature_weights(
            ratios,
            y,
            self.class_priors_,
            self.regularization_
        )

        # 8. Convert optimized log-odds threshold
        # to probability threshold.
        self.threshold_ = float(
            self._sigmoid(
                self.log_lr_threshold_
            )
        )

        # Preserve compatibility with the previous API.
        self.X_fit_ = X_train

        self.n_features_in_ = X_train.shape[1]

        self.is_fitted = True

        return self

    # ==========================================================
    # WEIGHTED INFERENCE
    # ==========================================================

    def decision_function(self, X):

        self._check_fitted()

        X_transformed = self.transform(X)

        ratios = self._feature_log_likelihood_ratio(
            X_transformed,
            self.theta_,
            self.var_
        )

        return (
            ratios @ self.feature_weights_
            + self.intercept_
        )

    def _compute_log_likelihood(self, X):

        if self.optimization_stage == 0:

            return super()._compute_log_likelihood(X)

        scores = self.decision_function(X)

        # Equivalent binary class scores up to
        # a common additive constant.
        # The inherited predict_log_proba() will
        # normalize these using log-sum-exp.
        return np.column_stack([
            np.zeros(len(scores)),
            scores
        ])

    def predict(self, X, threshold=None):

        self._check_fitted()

        if self.optimization_stage == 0:

            return super().predict(
                X,
                threshold=threshold
            )

        if threshold is None:

            scores = self.decision_function(X)

            return (
                scores >= self.log_lr_threshold_
            ).astype(np.int64)

        if not 0 <= threshold <= 1:

            raise ValueError(
                "threshold must be between 0 and 1."
            )

        probabilities = self.predict_proba(X)

        return (
            probabilities[:, 1] >= threshold
        ).astype(np.int64)

    # ==========================================================
    # FEATURE INFORMATION
    # ==========================================================

    def get_feature_info(self):

        self._check_fitted()

        information = super().get_feature_info()

        information["weight"] = (
            self.feature_weights_
        )

        information["encoding_smoothing"] = [
            self.feature_smoothing_[group]
            for group in self.selected_groups_
        ]

        information["is_active"] = (
            information["weight"] > 1e-8
        )

        return information

    def get_optimization_history(self):

        self._check_fitted()

        if self.optimization_stage == 0:

            return super().get_optimization_history()

        return pd.DataFrame(
            self.optimization_history_
        )

    # ==========================================================
    # MODEL SUMMARY
    # ==========================================================

    def summary(self):

        self._check_fitted()

        if self.optimization_stage == 0:

            return super().summary()

        active_features = int(
            np.sum(
                self.feature_weights_ > 1e-8
            )
        )

        print("=" * 65)

        print(
            "OPTIMIZED WEIGHTED GAUSSIAN NAIVE BAYES"
        )

        print("=" * 65)

        print(
            "Optimization stage      :",
            self.optimization_stage
        )

        print(
            "Candidate features      :",
            len(self.candidate_groups_)
        )

        print(
            "Active features         :",
            active_features
        )

        print(
            "Variance smoothing      :",
            self.var_smoothing_
        )

        print(
            "L2 regularization       :",
            self.regularization_
        )

        print(
            "Learning rate           :",
            self.learning_rate
        )

        print(
            "Weight iterations       :",
            self.weight_iterations
        )

        print(
            "Decision threshold      :",
            round(self.threshold_, 6)
        )

        print(
            "Internal validation     :",
            f"{self.validation_accuracy_:.4%}"
        )

        print(
            "Class priors            :",
            self.class_priors_
        )

        print("\nTop 15 features by weight:")

        information = (
            self.get_feature_info()
            .sort_values(
                "weight",
                ascending=False
            )
        )

        print(
            information[
                [
                    "feature",
                    "weight",
                    "encoding_smoothing"
                ]
            ]
            .head(15)
            .to_string(index=False)
        )

        print("=" * 65)



class _ExtendedNaiveBayes(_WeightedGaussianNaiveBayes):
    """
    Extended Optimized Weighted Gaussian Naive Bayes.

    Optimization stages:
        E0: Original Gaussian Naive Bayes
        E1: Weighted Gaussian Naive Bayes
        E2: Advanced Feature Engineering
        E3: Adaptive Target Encoding
        E4: Cross-fitted Likelihood Training
        E5: Hierarchical Target Encoding
        E6: Gaussian Mixture Naive Bayes

    References:

    Target Encoder Internal Cross-Fitting:
        https://scikit-learn.org/stable/auto_examples/
        preprocessing/plot_target_encoder_cross_val.html

    Micci-Barreca (2001):
        A Preprocessing Scheme for High-Cardinality
        Categorical Attributes in Classification
        and Prediction Problems.
        https://doi.org/10.1145/507533.507538

    Dempster, Laird, Rubin (1977):
        Maximum Likelihood from Incomplete Data
        via the EM Algorithm.
        https://doi.org/10.1111/j.2517-6161.1977.tb01600.x

    All machine learning algorithms are implemented
    from scratch using NumPy and Pandas.
    """

    # ==========================================================
    # INITIALIZATION
    # ==========================================================

    def __init__(
        self,
        optimization_stage=6,
        mixture_components=(1, 2, 3),
        mixture_variance_candidates=(0.03, 0.1),
        em_iterations=30,
        em_tolerance=1e-5,
        mixture_batch_size=16,
        verbose=True,
        **kwargs
    ):

        if optimization_stage not in range(7):
            raise ValueError(
                "optimization_stage must be between 0 and 6."
            )

        # Parent implementation supports E0-E3.
        super().__init__(
            optimization_stage=min(optimization_stage, 3),
            **kwargs
        )

        self.optimization_stage = optimization_stage

        self.mixture_components = tuple(
            sorted(set(mixture_components))
        )

        if (
            not self.mixture_components
            or 1 not in self.mixture_components
            or any(
                not isinstance(k, (int, np.integer)) or k < 1
                for k in self.mixture_components
            )
        ):
            raise ValueError(
                "mixture_components must contain positive "
                "integers, including 1."
            )

        self.mixture_variance_candidates = tuple(
            mixture_variance_candidates
        )

        if (
            not self.mixture_variance_candidates
            or not np.all(
                np.isfinite(self.mixture_variance_candidates)
            )
            or np.any(
                np.asarray(self.mixture_variance_candidates) <= 0
            )
        ):
            raise ValueError(
                "Mixture variance candidates must be positive."
            )

        if em_iterations < 1 or mixture_batch_size < 1:
            raise ValueError(
                "EM iterations and batch size must be positive."
            )

        if not np.isfinite(em_tolerance) or em_tolerance <= 0:
            raise ValueError(
                "EM tolerance must be positive and finite."
            )

        self.em_iterations = int(em_iterations)
        self.em_tolerance = float(em_tolerance)
        self.mixture_batch_size = int(mixture_batch_size)
        self.verbose = bool(verbose)

        # Final density parameters.
        self.n_components_ = 1

        self.mixture_weights_ = None
        self.mixture_means_ = None
        self.mixture_variances_ = None

        # Cross-fitted likelihood representation.
        self.oof_likelihood_ratios_ = None

    # ==========================================================
    # LOGGING
    # ==========================================================

    def _log(self, message):

        if self.verbose:
            print(message, flush=True)

    # ==========================================================
    # E5: HIERARCHICAL FEATURE GROUPS
    # ==========================================================

    @staticmethod
    def _hierarchical_levels(group):
        """
        Define a deterministic hierarchy from contextual,
        coarse-grained features to fine-grained features.

        Example:

            booking_channel
                -> booking_channel, special_requests
                -> booking_channel, special_requests,
                   arrival_date

        Hierarchy ordering does not use target labels.
        """

        priority = {
            "booking_channel": 0,
            "special_requests": 1,
            "room_category": 2,
            "meal_package": 3,

            "arrival_year": 4,
            "arrival_month": 5,

            "booking_month": 6,
            "booking_week": 7,

            "lead": 8,
            "rate": 9,

            "num_adults": 10,
            "num_children": 11,
            "guests": 12,

            "lead_fine": 20,
            "rate_fine": 21,

            "arrival_day": 30,
            "arrival_date": 31,
            "booking_date": 32,

            "lead_exact": 33,
            "rate_exact": 34
        }

        ordered = tuple(
            sorted(
                group,
                key=lambda feature: priority.get(feature, 15)
            )
        )

        return [
            ordered[:level]
            for level in range(1, len(ordered) + 1)
        ]

    # ==========================================================
    # E5: HIERARCHICAL ADAPTIVE TARGET ENCODING
    # ==========================================================

    def _fit_encoder(self, keys, y, groups):
        """
        E0-E4:
            Use the previous target encoding implementation.

        E5-E6:
            Hierarchical target encoding with adaptive
            smoothing selected using OOF Brier score.
        """

        if self.optimization_stage < 5:

            return super()._fit_encoder(
                keys,
                y,
                groups
            )

        n_samples = len(y)

        candidates = np.asarray(
            self.smoothing_candidates,
            dtype=np.float64
        )

        encoded = np.empty(
            (n_samples, len(groups)),
            dtype=np.float64
        )

        mappings = {}
        selected_smoothing = {}

        folds = list(self._stratified_folds(y))

        global_prior = float(np.mean(y))

        # Cache factorized keys shared by hierarchies.
        factorization_cache = {}

        def factorize_level(level):

            if level not in factorization_cache:

                group_keys = self._group_key(keys, level)

                codes, unique = pd.factorize(
                    group_keys,
                    sort=False
                )

                if np.any(codes < 0):
                    raise ValueError(
                        "Unexpected missing category code."
                    )

                first_indices = np.unique(
                    codes,
                    return_index=True
                )[1]

                factorization_cache[level] = (
                    codes,
                    unique,
                    first_indices
                )

            return factorization_cache[level]

        for j, group in enumerate(groups):

            levels = self._hierarchical_levels(group)

            level_data = [
                factorize_level(level)
                for level in levels
            ]

            candidate_oof = np.empty(
                (n_samples, len(candidates)),
                dtype=np.float64
            )

            # --------------------------------------------------
            # OUT-OF-FOLD HIERARCHICAL ENCODING
            # --------------------------------------------------

            for train_idx, holdout_idx in folds:

                fold_prior = float(
                    np.mean(y[train_idx])
                )

                # Level 0: global training prior.
                parent_encoding = np.full(
                    (len(holdout_idx), len(candidates)),
                    fold_prior,
                    dtype=np.float64
                )

                for codes, unique, _ in level_data:

                    n_categories = len(unique)

                    counts = np.bincount(
                        codes[train_idx],
                        minlength=n_categories
                    )

                    positives = np.bincount(
                        codes[train_idx],
                        weights=y[train_idx],
                        minlength=n_categories
                    )

                    holdout_codes = codes[holdout_idx]

                    # Hierarchical shrinkage:
                    # child is smoothed toward its parent.
                    parent_encoding = (
                        positives[holdout_codes, None]
                        + candidates * parent_encoding
                    ) / (
                        counts[holdout_codes, None]
                        + candidates
                    )

                candidate_oof[holdout_idx] = (
                    parent_encoding
                )

            # --------------------------------------------------
            # ADAPTIVE SMOOTHING PER FEATURE GROUP
            # --------------------------------------------------

            brier_scores = np.mean(
                (
                    candidate_oof - y[:, None]
                ) ** 2,
                axis=0
            )

            best_idx = int(
                np.argmin(brier_scores)
            )

            best_alpha = float(
                candidates[best_idx]
            )

            selected_smoothing[group] = best_alpha

            encoded[:, j] = (
                candidate_oof[:, best_idx]
            )

            # --------------------------------------------------
            # FULL-TRAINING HIERARCHICAL MAPS
            # --------------------------------------------------

            group_maps = []

            parent_encoding = np.full(
                n_samples,
                global_prior,
                dtype=np.float64
            )

            for level, data in zip(
                levels,
                level_data
            ):

                codes, unique, first_indices = data

                n_categories = len(unique)

                counts = np.bincount(
                    codes,
                    minlength=n_categories
                )

                positives = np.bincount(
                    codes,
                    weights=y,
                    minlength=n_categories
                )

                parent_encoding = (
                    positives[codes]
                    + best_alpha * parent_encoding
                ) / (
                    counts[codes]
                    + best_alpha
                )

                # All rows belonging to a category
                # have the same encoded value.
                mapping = dict(
                    zip(
                        unique,
                        parent_encoding[first_indices]
                    )
                )

                group_maps.append(
                    (level, mapping)
                )

            mappings[group] = group_maps

        self.feature_smoothing_ = selected_smoothing

        return encoded, mappings, global_prior

    # ==========================================================
    # E5: HIERARCHICAL ENCODING INFERENCE
    # ==========================================================

    def _transform_encoder(
        self,
        keys,
        groups,
        mappings,
        prior
    ):

        if self.optimization_stage < 5:

            return super()._transform_encoder(
                keys,
                groups,
                mappings,
                prior
            )

        n_samples = len(
            next(iter(keys.values()))
        )

        encoded = np.empty(
            (n_samples, len(groups)),
            dtype=np.float64
        )

        key_cache = {}

        for j, group in enumerate(groups):

            # Initial fallback: global prior.
            values = np.full(
                n_samples,
                prior,
                dtype=np.float64
            )

            for level, mapping in mappings[group]:

                if level not in key_cache:

                    key_cache[level] = (
                        self._group_key(keys, level)
                    )

                mapped = (
                    key_cache[level]
                    .map(mapping)
                    .to_numpy(dtype=np.float64)
                )

                # If a child category is unknown,
                # preserve its parent's probability.
                values = np.where(
                    np.isfinite(mapped),
                    mapped,
                    values
                )

            encoded[:, j] = values

        return encoded

    # ==========================================================
    # E6: GAUSSIAN MIXTURE LOG DENSITY
    # ==========================================================

    @staticmethod
    def _mixture_component_log_pdf(
        X,
        weights,
        means,
        variances
    ):
        """
        X shape:
            (n_samples, n_features)

        Parameter shape:
            (n_features, n_components)

        Output:
            (n_samples, n_features, n_components)
        """

        difference = (
            X[:, :, None] - means[None, :, :]
        )

        return (
            np.log(weights)[None, :, :]
            - 0.5 * (
                np.log(
                    2.0 * np.pi * variances
                )[None, :, :]
                + difference ** 2
                / variances[None, :, :]
            )
        )

    @classmethod
    def _mixture_log_pdf(
        cls,
        X,
        weights,
        means,
        variances
    ):

        component_log_probs = (
            cls._mixture_component_log_pdf(
                X,
                weights,
                means,
                variances
            )
        )

        # Numerically stable log-sum-exp.
        return np.logaddexp.reduce(
            component_log_probs,
            axis=2
        )

    # ==========================================================
    # E6: EXPECTATION-MAXIMIZATION FROM SCRATCH
    # ==========================================================

    def _fit_mixture_batch(
        self,
        X,
        n_components,
        variance_floor
    ):
        """
        Fit independent univariate Gaussian mixtures.

        Every feature has its own mixture parameters.
        Features are not combined into a multivariate GMM.
        """

        n_samples, n_features = X.shape

        K = n_components

        # ------------------------------------------------------
        # INITIALIZATION
        # ------------------------------------------------------

        quantiles = (
            np.arange(K) + 0.5
        ) / K

        # Deterministic initialization using quantiles.
        means = np.quantile(
            X,
            quantiles,
            axis=0
        ).T.copy()

        initial_variance = np.maximum(
            np.var(X, axis=0) / K,
            variance_floor
        )

        variances = np.repeat(
            initial_variance[:, None],
            K,
            axis=1
        )

        weights = np.full(
            (n_features, K),
            1.0 / K,
            dtype=np.float64
        )

        previous_log_likelihood = None

        # ------------------------------------------------------
        # EXPECTATION-MAXIMIZATION LOOP
        # ------------------------------------------------------

        for iteration in range(self.em_iterations):

            # ==================================================
            # E-STEP
            # ==================================================

            log_joint = (
                self._mixture_component_log_pdf(
                    X,
                    weights,
                    means,
                    variances
                )
            )

            log_normalizer = np.logaddexp.reduce(
                log_joint,
                axis=2,
                keepdims=True
            )

            responsibilities = np.exp(
                log_joint - log_normalizer
            )

            current_log_likelihood = float(
                np.mean(log_normalizer)
            )

            # Check convergence.
            if previous_log_likelihood is not None:

                improvement = abs(
                    current_log_likelihood
                    - previous_log_likelihood
                )

                if improvement < self.em_tolerance:
                    break

            previous_log_likelihood = (
                current_log_likelihood
            )

            # ==================================================
            # M-STEP
            # ==================================================

            component_mass = responsibilities.sum(
                axis=0
            )

            safe_mass = np.maximum(
                component_mass,
                1e-12
            )

            new_weights = (
                component_mass / n_samples
            )

            new_means = (
                responsibilities
                * X[:, :, None]
            ).sum(axis=0) / safe_mass

            difference = (
                X[:, :, None]
                - new_means[None, :, :]
            )

            new_variances = (
                responsibilities
                * difference ** 2
            ).sum(axis=0) / safe_mass

            # Handle nearly empty mixture components.
            active = component_mass > 1e-6

            means = np.where(
                active,
                new_means,
                means
            )

            variances = np.where(
                active,
                new_variances,
                variances
            )

            # Prevent singular Gaussian distributions.
            variances = np.maximum(
                variances,
                variance_floor
            )

            weights = np.maximum(
                new_weights,
                1e-8
            )

            weights /= weights.sum(
                axis=1,
                keepdims=True
            )

        return weights, means, variances

    # ==========================================================
    # E6: CLASS-CONDITIONAL DENSITY FITTING
    # ==========================================================

    def _fit_density(
        self,
        X,
        y,
        n_components,
        smoothing
    ):
        """
        Return arrays with shape:
            (2, n_features, n_components)

        Axis 0: class
        Axis 1: feature
        Axis 2: Gaussian component
        """

        n_features = X.shape[1]

        # K=1 preserves the previous Gaussian model.
        if n_components == 1:

            means, variances, _, _ = (
                self._fit_gaussian(
                    X,
                    y,
                    smoothing
                )
            )

            weights = np.ones(
                (2, n_features, 1),
                dtype=np.float64
            )

            return (
                weights,
                means[:, :, None],
                variances[:, :, None]
            )

        # K>1 uses independent Gaussian mixtures.
        shape = (
            2,
            n_features,
            n_components
        )

        mixture_weights = np.empty(shape)
        mixture_means = np.empty(shape)
        mixture_variances = np.empty(shape)

        for c in (0, 1):

            X_c = X[y == c]

            if len(X_c) == 0:
                raise ValueError(
                    "Both classes must be represented."
                )

            # Process features in batches to limit memory.
            for start in range(
                0,
                n_features,
                self.mixture_batch_size
            ):

                end = min(
                    start + self.mixture_batch_size,
                    n_features
                )

                weights, means, variances = (
                    self._fit_mixture_batch(
                        X_c[:, start:end],
                        n_components,
                        smoothing
                    )
                )

                mixture_weights[c, start:end] = weights

                mixture_means[c, start:end] = means

                mixture_variances[c, start:end] = variances

        return (
            mixture_weights,
            mixture_means,
            mixture_variances
        )

    # ==========================================================
    # E6: FEATURE LOG-LIKELIHOOD RATIOS
    # ==========================================================

    def _density_likelihood_ratios(
        self,
        X,
        density
    ):

        weights, means, variances = density

        n_components = weights.shape[2]

        # Preserve the original Gaussian ratio for K=1.
        if n_components == 1:

            return self._feature_log_likelihood_ratio(
                X,
                means[:, :, 0],
                variances[:, :, 0]
            )

        n_samples, n_features = X.shape

        ratios = np.empty(
            (n_samples, n_features),
            dtype=np.float64
        )

        for start in range(
            0,
            n_features,
            self.mixture_batch_size
        ):

            end = min(
                start + self.mixture_batch_size,
                n_features
            )

            X_batch = X[:, start:end]

            log_prob_0 = self._mixture_log_pdf(
                X_batch,
                weights[0, start:end],
                means[0, start:end],
                variances[0, start:end]
            )

            log_prob_1 = self._mixture_log_pdf(
                X_batch,
                weights[1, start:end],
                means[1, start:end],
                variances[1, start:end]
            )

            ratios[:, start:end] = (
                log_prob_1 - log_prob_0
            )

        return ratios

    # ==========================================================
    # E4: PREPROCESSING PIPELINE
    # ==========================================================

    def _fit_preprocessing_pipeline(
        self,
        keys,
        y,
        groups
    ):

        counts = np.bincount(
            y,
            minlength=2
        )

        if np.min(counts) < self.n_folds:
            raise ValueError(
                "Insufficient class samples for nested "
                "out-of-fold target encoding."
            )

        encoded, mappings, prior = (
            self._fit_encoder(
                keys,
                y,
                groups
            )
        )

        mean, std = self._fit_scaler(
            encoded
        )

        X_train = self._transform_scaler(
            encoded,
            mean,
            std
        )

        pipeline = {
            "mappings": mappings,
            "prior": prior,
            "mean": mean,
            "std": std,
            "feature_smoothing": (
                self.feature_smoothing_.copy()
            )
        }

        return X_train, pipeline

    def _apply_preprocessing_pipeline(
        self,
        keys,
        groups,
        pipeline
    ):

        encoded = self._transform_encoder(
            keys,
            groups,
            pipeline["mappings"],
            pipeline["prior"]
        )

        return self._transform_scaler(
            encoded,
            pipeline["mean"],
            pipeline["std"]
        )

    # ==========================================================
    # E4: CROSS-FITTED LIKELIHOOD TRAINING
    # ==========================================================

    def _crossfit_likelihood_ratios(
        self,
        keys,
        y,
        groups,
        density_configs
    ):
        """
        Generate out-of-fold likelihood ratios.

        For every outer fold:
            1. Fit an OOF target encoder using only
               outer-fold training data.
            2. Fit scaler on outer-fold training data.
            3. Fit Gaussian/GMM density.
            4. Generate ratios for held-out rows.

        No held-out label is used by that fold's
        encoder, scaler, or density estimator.
        """

        n_samples = len(y)

        n_features = len(groups)

        results = {
            config: np.empty(
                (n_samples, n_features),
                dtype=np.float64
            )
            for config in density_configs
        }

        folds = list(
            self._stratified_folds(y)
        )

        for fold_number, indices in enumerate(
            folds,
            start=1
        ):

            train_idx, holdout_idx = indices

            self._log(
                f"  Cross-fitting fold "
                f"{fold_number}/{len(folds)}"
            )

            train_keys = self._subset_keys(
                keys,
                train_idx
            )

            holdout_keys = self._subset_keys(
                keys,
                holdout_idx
            )

            y_train = y[train_idx]

            # Entire preprocessing is independently
            # fitted inside every outer fold.
            X_train, pipeline = (
                self._fit_preprocessing_pipeline(
                    train_keys,
                    y_train,
                    groups
                )
            )

            X_holdout = (
                self._apply_preprocessing_pipeline(
                    holdout_keys,
                    groups,
                    pipeline
                )
            )

            # Reuse fold preprocessing across density
            # hyperparameter configurations.
            for config in density_configs:

                n_components, smoothing = config

                density = self._fit_density(
                    X_train,
                    y_train,
                    n_components,
                    smoothing
                )

                ratios = self._density_likelihood_ratios(
                    X_holdout,
                    density
                )

                results[config][holdout_idx] = ratios

        for config, ratios in results.items():

            if not np.all(np.isfinite(ratios)):
                raise FloatingPointError(
                    f"Non-finite likelihood ratios: {config}"
                )

        return results

    # ==========================================================
    # DENSITY HYPERPARAMETER CONFIGURATIONS
    # ==========================================================

    def _density_configurations(self):

        # E4 and E5 use single Gaussian distributions.
        configs = [
            (1, float(smoothing))
            for smoothing in self.variance_candidates
        ]

        if self.optimization_stage >= 6:

            # Additional mixture configurations.
            # Limit the grid to avoid excessive EM training.
            configs.extend([
                (int(k), float(smoothing))
                for k in self.mixture_components
                if k > 1
                for smoothing
                in self.mixture_variance_candidates
            ])

        return configs

    # ==========================================================
    # INTERNAL VALIDATION AND HYPERPARAMETER TUNING
    # ==========================================================

    def _optimize_crossfitted(
        self,
        keys,
        y,
        groups
    ):

        train_idx, val_idx = (
            self._train_validation_split(y)
        )

        train_keys = self._subset_keys(
            keys,
            train_idx
        )

        val_keys = self._subset_keys(
            keys,
            val_idx
        )

        y_train = y[train_idx]

        y_val = y[val_idx]

        configs = self._density_configurations()

        self._log(
            "\nGenerating internal OOF likelihood ratios..."
        )

        oof_ratios = self._crossfit_likelihood_ratios(
            train_keys,
            y_train,
            groups,
            configs
        )

        # Fit inference preprocessing on the entire
        # internal training subset.
        X_train, pipeline = (
            self._fit_preprocessing_pipeline(
                train_keys,
                y_train,
                groups
            )
        )

        X_val = self._apply_preprocessing_pipeline(
            val_keys,
            groups,
            pipeline
        )

        counts = np.bincount(
            y_train,
            minlength=2
        )

        priors = counts / counts.sum()

        best = None

        self.optimization_history_ = []

        # ------------------------------------------------------
        # HYPERPARAMETER GRID SEARCH
        # ------------------------------------------------------

        for config in configs:

            n_components, smoothing = config

            self._log(
                f"\nTesting K={n_components}, "
                f"smoothing={smoothing}"
            )

            density = self._fit_density(
                X_train,
                y_train,
                n_components,
                smoothing
            )

            validation_ratios = (
                self._density_likelihood_ratios(
                    X_val,
                    density
                )
            )

            training_ratios = oof_ratios[config]

            for regularization in (
                self.regularization_candidates
            ):

                # Learn weights using exclusively
                # cross-fitted training ratios.
                weights, intercept = (
                    self._learn_feature_weights(
                        training_ratios,
                        y_train,
                        priors,
                        regularization
                    )
                )

                scores = (
                    validation_ratios @ weights
                    + intercept
                )

                accuracy, threshold = (
                    self._find_best_threshold(
                        scores,
                        y_val
                    )
                )

                result = {
                    "accuracy": float(accuracy),
                    "n_components": int(n_components),
                    "var_smoothing": float(smoothing),
                    "regularization": float(
                        regularization
                    ),
                    "threshold": float(threshold),
                    "active_features": int(
                        np.sum(weights > 1e-8)
                    )
                }

                self.optimization_history_.append(
                    result
                )

                self._log(
                    f"  L2={regularization}: "
                    f"accuracy={accuracy:.4%}"
                )

                # Candidates are evaluated in order of
                # increasing mixture complexity.
                # Ties preserve the earlier candidate.
                if (
                    best is None
                    or accuracy > best["accuracy"]
                ):

                    best = result.copy()

        return best

    # ==========================================================
    # FINAL MODEL TRAINING
    # ==========================================================

    def fit(self, X, y=None):

        # Preserve previous implementations.
        if self.optimization_stage <= 3:

            self.n_components_ = 1

            self.mixture_weights_ = None
            self.mixture_means_ = None
            self.mixture_variances_ = None

            self.oof_likelihood_ratios_ = None

            return super().fit(X, y)

        self.is_fitted = False

        self._validate_dataframe(X)

        if y is None:

            if "is_canceled" not in X.columns:

                raise ValueError(
                    "Provide y or an is_canceled column."
                )

            y = X["is_canceled"].to_numpy()

        y = self._validate_target(X, y)

        self._log(
            f"\nTraining optimization stage E"
            f"{self.optimization_stage}"
        )

        # ------------------------------------------------------
        # 1. FEATURE ENGINEERING
        # ------------------------------------------------------

        keys = self._build_feature_keys(X)

        groups = self._candidate_groups(keys)

        self.candidate_groups_ = groups

        self.selected_groups_ = groups

        self.feature_names_ = [
            " × ".join(group)
            for group in groups
        ]

        # ------------------------------------------------------
        # 2. INTERNAL HYPERPARAMETER OPTIMIZATION
        # ------------------------------------------------------

        best = self._optimize_crossfitted(
            keys,
            y,
            groups
        )

        self.n_components_ = best["n_components"]

        self.var_smoothing_ = best["var_smoothing"]

        self.regularization_ = best["regularization"]

        self.log_lr_threshold_ = best["threshold"]

        self.validation_accuracy_ = best["accuracy"]

        self._log(
            "\nBest internal configuration:"
        )

        self._log(str(best))

        # ------------------------------------------------------
        # 3. FINAL CROSS-FITTED LIKELIHOOD RATIOS
        # ------------------------------------------------------

        selected_config = (
            self.n_components_,
            self.var_smoothing_
        )

        self._log(
            "\nGenerating final OOF likelihood ratios..."
        )

        oof_results = self._crossfit_likelihood_ratios(
            keys,
            y,
            groups,
            [selected_config]
        )

        self.oof_likelihood_ratios_ = (
            oof_results[selected_config]
        )

        # ------------------------------------------------------
        # 4. FINAL PREPROCESSING PIPELINE
        # ------------------------------------------------------

        X_train, pipeline = (
            self._fit_preprocessing_pipeline(
                keys,
                y,
                groups
            )
        )

        self.category_maps_ = pipeline["mappings"]

        self.encoding_prior_ = pipeline["prior"]

        self.scaler_mean_ = pipeline["mean"]

        self.scaler_std_ = pipeline["std"]

        self.feature_smoothing_ = (
            pipeline["feature_smoothing"]
        )

        # ------------------------------------------------------
        # 5. FINAL GAUSSIAN OR GMM DENSITY
        # ------------------------------------------------------

        (
            self.mixture_weights_,
            self.mixture_means_,
            self.mixture_variances_
        ) = self._fit_density(
            X_train,
            y,
            self.n_components_,
            self.var_smoothing_
        )

        self.class_counts_ = np.bincount(
            y,
            minlength=2
        )

        self.class_priors_ = (
            self.class_counts_
            / self.class_counts_.sum()
        )

        self.log_class_priors_ = np.log(
            self.class_priors_
        )

        # Aggregate mixture moments for compatibility
        # with inherited diagnostic functions.
        self.theta_ = np.sum(
            self.mixture_weights_
            * self.mixture_means_,
            axis=2
        )

        self.var_ = np.sum(
            self.mixture_weights_
            * (
                self.mixture_variances_
                + (
                    self.mixture_means_
                    - self.theta_[:, :, None]
                ) ** 2
            ),
            axis=2
        )

        # ------------------------------------------------------
        # 6. FINAL REGULARIZED FEATURE WEIGHTING
        # ------------------------------------------------------

        # IMPORTANT:
        # Weights are learned from OOF likelihood ratios,
        # not from the final density's in-sample ratios.
        (
            self.feature_weights_,
            self.intercept_
        ) = self._learn_feature_weights(
            self.oof_likelihood_ratios_,
            y,
            self.class_priors_,
            self.regularization_
        )

        # ------------------------------------------------------
        # 7. FINAL CLASSIFICATION PARAMETERS
        # ------------------------------------------------------

        self.threshold_ = float(
            self._sigmoid(
                self.log_lr_threshold_
            )
        )

        self.X_fit_ = X_train

        self.n_features_in_ = X_train.shape[1]

        self.is_fitted = True

        return self

    # ==========================================================
    # FINAL WEIGHTED INFERENCE
    # ==========================================================

    def decision_function(self, X):

        self._check_fitted()

        if self.optimization_stage <= 3:

            return super().decision_function(X)

        X_transformed = self.transform(X)

        density = (
            self.mixture_weights_,
            self.mixture_means_,
            self.mixture_variances_
        )

        ratios = self._density_likelihood_ratios(
            X_transformed,
            density
        )

        scores = (
            ratios @ self.feature_weights_
            + self.intercept_
        )

        return scores

    # ==========================================================
    # MIXTURE PARAMETER INSPECTION
    # ==========================================================

    def get_mixture_info(self):

        self._check_fitted()

        if self.optimization_stage < 4:

            raise RuntimeError(
                "Mixture inspection requires E4 or later."
            )

        records = []

        for c in (0, 1):

            for j, feature in enumerate(
                self.feature_names_
            ):

                for k in range(self.n_components_):

                    records.append({
                        "class": c,
                        "feature": feature,
                        "component": k + 1,
                        "mixture_weight": (
                            self.mixture_weights_[c, j, k]
                        ),
                        "mean": (
                            self.mixture_means_[c, j, k]
                        ),
                        "variance": (
                            self.mixture_variances_[c, j, k]
                        ),
                        "feature_weight": (
                            self.feature_weights_[j]
                        )
                    })

        return pd.DataFrame(records)

    # ==========================================================
    # MODEL SUMMARY
    # ==========================================================

    def summary(self):

        self._check_fitted()

        super().summary()

        if self.optimization_stage >= 4:

            print(
                "Cross-fitted likelihood : Enabled"
            )

            print(
                "Hierarchical encoding   :",
                self.optimization_stage >= 5
            )

            print(
                "Gaussian components     :",
                self.n_components_
            )

            print(
                "Density estimator       :",
                (
                    "Gaussian Mixture"
                    if self.n_components_ > 1
                    else "Single Gaussian"
                )
            )

            print(
                "EM iterations (maximum) :",
                self.em_iterations
            )

            print("=" * 65)

class NaiveBayes(_ExtendedNaiveBayes):
    """
    Robust Weighted Naive Bayes from Scratch.

    IF3170 - G15 MessiLearning

    Optimization stages:
        E0: Original Gaussian Naive Bayes
        E1: Weighted Gaussian Naive Bayes
        E2: Advanced Feature Engineering
        E3: Adaptive Target Encoding
        E4: Cross-fitted Likelihood Training
        E5: Hierarchical Target Encoding
        E6: Gaussian Mixture Naive Bayes
        E7: Robust Weighted Gaussian Naive Bayes
        E8: Direct Likelihood Ratio Naive Bayes

    E7 and E8 are alternative developments of E3.
    They do not inherit Gaussian mixture density estimation.

    References:
        Zou and Hastie (2005).
        Regularization and Variable Selection
        via the Elastic Net.
        https://doi.org/10.1111/j.1467-9868.2005.00503.x

        Micci-Barreca (2001).
        A Preprocessing Scheme for High-Cardinality
        Categorical Attributes in Classification
        and Prediction Problems.
        https://doi.org/10.1145/507533.507538

        Zaidi et al. (2013).
        Alleviating Naive Bayes Attribute Independence
        Assumption by Attribute Weighting.
        https://jmlr.org/papers/v14/zaidi13a.html

        Kingma and Ba (2015).
        Adam: A Method for Stochastic Optimization.
        https://arxiv.org/abs/1412.6980

    No sklearn estimator is used.
    """

    # ==========================================================
    # INITIALIZATION
    # ==========================================================

    def __init__(
        self,
        optimization_stage=8,
        robust_configs=None,
        calibration_folds=3,
        min_feature_coverage=0.20,
        threshold_tolerance=0.0005,
        **kwargs
    ):

        if optimization_stage not in range(9):
            raise ValueError(
                "optimization_stage must be 0..8"
            )

        super().__init__(
            optimization_stage=min(
                optimization_stage, 6
            ),
            **kwargs
        )

        self.optimization_stage = optimization_stage

        self.calibration_folds = int(
            calibration_folds
        )

        self.min_feature_coverage = float(
            min_feature_coverage
        )

        self.threshold_tolerance = float(
            threshold_tolerance
        )

        # Configuration format:
        #
        # (
        #     minimum_category_support,
        #     encoding_method,
        #     variance_smoothing,
        #     lambda_L1,
        #     lambda_L2
        # )

        if optimization_stage == 7:

            defaults = (
                (3, "oof", 0.1, 0.0, 0.01),
                (3, "oof", 0.1, 0.001, 0.01),
                (10, "oof", 0.1, 0.001, 0.01),
                (10, "loo", 0.1, 0.001, 0.01),
                (10, "oof", 0.3, 0.005, 0.05),
            )

        else:

            # E8 does not use Gaussian variance.
            defaults = (
                (3, "oof", 0.0, 0.0, 0.01),
                (3, "oof", 0.0, 0.001, 0.01),
                (10, "oof", 0.0, 0.001, 0.01),
                (20, "oof", 0.0, 0.005, 0.05),
            )

        self.robust_configs = tuple(
            dict.fromkeys(
                defaults
                if robust_configs is None
                else map(tuple, robust_configs)
            )
        )

        if (
            self.calibration_folds < 2
            or not 0 <= self.min_feature_coverage <= 1
            or not 0 <= self.threshold_tolerance <= 1
            or not self.robust_configs
        ):
            raise ValueError(
                "Invalid robust configuration"
            )

        if optimization_stage >= 7:

            for config in self.robust_configs:

                support, mode, variance, l1, l2 = config

                if (
                    not isinstance(
                        support, (int, np.integer)
                    )
                    or support < 1
                    or mode not in ("oof", "loo")
                    or not np.all(
                        np.isfinite([variance, l1, l2])
                    )
                    or min(l1, l2) < 0
                    or (
                        optimization_stage == 7
                        and variance <= 0
                    )
                    or (
                        optimization_stage == 8
                        and (
                            variance != 0
                            or mode != "oof"
                        )
                    )
                ):
                    raise ValueError(
                        "Invalid candidate"
                    )

        self.robust_pipeline_ = None
        self.robust_config_ = None

    # ==========================================================
    # NUMERICALLY STABLE LOGIT
    # ==========================================================

    @staticmethod
    def _logit(p):

        p = np.clip(
            p,
            1e-6,
            1 - 1e-6
        )

        return (
            np.log(p)
            - np.log1p(-p)
        )

    # ==========================================================
    # STRATIFIED CROSS-VALIDATION FROM SCRATCH
    # ==========================================================

    def _robust_folds(self, y, n, seed):

        rng = np.random.default_rng(seed)

        parts = []

        for c in (0, 1):

            indices = rng.permutation(
                np.flatnonzero(y == c)
            )

            if len(indices) < n:

                raise ValueError(
                    "Too few samples for cross-fitting"
                )

            parts.append(
                np.array_split(indices, n)
            )

        for k in range(n):

            val = np.concatenate([
                p[k] for p in parts
            ])

            mask = np.ones(
                len(y),
                dtype=bool
            )

            mask[val] = False

            yield np.flatnonzero(mask), val

    # ==========================================================
    # SUPPORT-AWARE ADAPTIVE TARGET ENCODING
    # ==========================================================

    def _robust_encode(
        self,
        keys,
        y,
        groups,
        support,
        mode
    ):
        """
        Fit adaptive target encoders.

        E7:
            Flat target encoding with minimum support.

        E8:
            Hierarchical target encoding with
            minimum support and parent fallback.

        Supported training encodings:
            - Out-of-fold encoding
            - Leave-one-out encoding (E7 only)
        """

        n = len(y)

        alpha = np.asarray(
            self.smoothing_candidates,
            dtype=float
        )

        prior = float(y.mean())

        encoded = np.empty(
            (n, len(groups))
        )

        maps = {}
        alphas = {}

        active = np.zeros(
            len(groups),
            dtype=bool
        )

        cache = {}

        folds = list(
            self._robust_folds(
                y,
                self.n_folds,
                self.random_state
            )
        )

        # ------------------------------------------------------
        # Cache categorical factorization and statistics
        # ------------------------------------------------------

        def factor(level):

            if level not in cache:

                codes, unique = pd.factorize(
                    self._group_key(keys, level),
                    sort=False
                )

                counts = np.bincount(
                    codes,
                    minlength=len(unique)
                )

                positives = np.bincount(
                    codes,
                    weights=y,
                    minlength=len(unique)
                )

                first = np.unique(
                    codes,
                    return_index=True
                )[1]

                cache[level] = (
                    codes,
                    unique,
                    counts,
                    positives,
                    first
                )

            return cache[level]

        # ------------------------------------------------------
        # Process every candidate feature
        # ------------------------------------------------------

        for j, group in enumerate(groups):

            if self.optimization_stage == 8:

                levels = self._hierarchical_levels(
                    group
                )

            else:

                levels = [group]

            data = [
                factor(level)
                for level in levels
            ]

            # ==================================================
            # FREQUENCY-AWARE FEATURE SELECTION
            # ==================================================

            (
                leaf_codes,
                _,
                leaf_counts,
                _,
                _
            ) = data[-1]

            coverage = np.mean(
                leaf_counts[leaf_codes] >= support
            )

            active[j] = (
                coverage >= self.min_feature_coverage
            )

            candidates = np.empty(
                (n, len(alpha))
            )

            # ==================================================
            # LEAVE-ONE-OUT TARGET ENCODING
            # ==================================================

            if mode == "loo":

                # Exclude the current sample's target
                # from the global prior.

                p = np.broadcast_to(
                    (
                        (y.sum() - y) / (n - 1)
                    )[:, None],
                    (n, len(alpha))
                ).copy()

                for (
                    codes,
                    _,
                    counts,
                    positives,
                    _
                ) in data:

                    # Exclude the current sample's
                    # category count and target.

                    cnt = counts[codes] - 1

                    pos = positives[codes] - y

                    q = (
                        pos[:, None] + alpha * p
                    ) / (
                        cnt[:, None] + alpha
                    )

                    p = np.where(
                        cnt[:, None] >= support,
                        q,
                        p
                    )

                candidates[:] = p

            # ==================================================
            # OUT-OF-FOLD TARGET ENCODING
            # ==================================================

            else:

                for ti, vi in folds:

                    p = np.full(
                        (len(vi), len(alpha)),
                        y[ti].mean()
                    )

                    for (
                        codes,
                        unique,
                        _,
                        _,
                        _
                    ) in data:

                        cnt = np.bincount(
                            codes[ti],
                            minlength=len(unique)
                        )[codes[vi]]

                        pos = np.bincount(
                            codes[ti],
                            weights=y[ti],
                            minlength=len(unique)
                        )[codes[vi]]

                        q = (
                            pos[:, None] + alpha * p
                        ) / (
                            cnt[:, None] + alpha
                        )

                        # Use parent encoding or global
                        # prior when support is insufficient.

                        p = np.where(
                            cnt[:, None] >= support,
                            q,
                            p
                        )

                    candidates[vi] = p

            # ==================================================
            # ADAPTIVE SMOOTHING PER FEATURE
            # ==================================================

            brier_scores = np.mean(
                (
                    candidates - y[:, None]
                ) ** 2,
                axis=0
            )

            k = int(
                np.argmin(brier_scores)
            )

            a = float(alpha[k])

            alphas[group] = a

            encoded[:, j] = candidates[:, k]

            # ==================================================
            # FULL-TRAINING CATEGORY MAPS
            # ==================================================

            p = np.full(
                n,
                prior
            )

            group_maps = []

            for level, stats in zip(levels, data):

                (
                    codes,
                    unique,
                    counts,
                    positives,
                    first
                ) = stats

                cnt = counts[codes]

                q = (
                    positives[codes] + a * p
                ) / (
                    cnt + a
                )

                p = np.where(
                    cnt >= support,
                    q,
                    p
                )

                # Categories below minimum support
                # are excluded from the final map.

                mapping = {
                    key: float(p[i])
                    for key, i, c in zip(
                        unique,
                        first,
                        counts
                    )
                    if c >= support
                }

                group_maps.append(
                    (level, mapping)
                )

            maps[group] = group_maps

        pipeline = {
            "maps": maps,
            "alphas": alphas,
            "prior": prior,
            "active": active
        }

        return encoded, pipeline

    # ==========================================================
    # SUPPORT-AWARE ENCODING INFERENCE
    # ==========================================================

    def _robust_apply(
        self,
        keys,
        groups,
        pipe
    ):

        n = len(
            next(iter(keys.values()))
        )

        encoded = np.full(
            (n, len(groups)),
            pipe["prior"]
        )

        seen = np.zeros_like(
            encoded,
            dtype=bool
        )

        cache = {}

        for j, group in enumerate(groups):

            for level, mapping in pipe["maps"][group]:

                if level not in cache:

                    cache[level] = self._group_key(
                        keys,
                        level
                    )

                values = (
                    cache[level]
                    .map(mapping)
                    .to_numpy(dtype=float)
                )

                valid = np.isfinite(values)

                encoded[valid, j] = values[valid]

                seen[:, j] |= valid

        return encoded, seen

    # ==========================================================
    # E7/E8: FIT PROBABILITY ESTIMATION BASIS
    # ==========================================================

    def _fit_robust_basis(
        self,
        keys,
        y,
        groups,
        config
    ):

        support, mode, variance, _, _ = config

        encoded, pipe = self._robust_encode(
            keys,
            y,
            groups,
            support,
            mode
        )

        # ------------------------------------------------------
        # E7: Gaussian Naive Bayes
        # ------------------------------------------------------

        if self.optimization_stage == 7:

            mean, std = self._fit_scaler(
                encoded
            )

            X = self._transform_scaler(
                encoded,
                mean,
                std
            )

            theta, var, _, _ = self._fit_gaussian(
                X,
                y,
                variance
            )

            pipe.update(
                mean=mean,
                std=std,
                theta=theta,
                var=var
            )

        # ------------------------------------------------------
        # E8: Direct Likelihood Ratio
        # ------------------------------------------------------

        else:

            X = (
                self._logit(encoded)
                - self._logit(pipe["prior"])
            )

        return X, pipe

    # ==========================================================
    # E7/E8: COMPUTE FEATURE LIKELIHOOD RATIOS
    # ==========================================================

    def _robust_ratios(
        self,
        keys,
        groups,
        pipe
    ):

        encoded, seen = self._robust_apply(
            keys,
            groups,
            pipe
        )

        if self.optimization_stage == 7:

            X = self._transform_scaler(
                encoded,
                pipe["mean"],
                pipe["std"]
            )

            R = self._feature_log_likelihood_ratio(
                X,
                pipe["theta"],
                pipe["var"]
            )

        else:

            # Bayes-derived direct categorical evidence:
            #
            # logit(P(Y=1 | category))
            # - logit(P(Y=1))

            R = (
                self._logit(encoded)
                - self._logit(pipe["prior"])
            )

        # Unsupported or unreliable features contribute
        # zero evidence to the final classification.

        return np.where(
            seen & pipe["active"][None, :],
            R,
            0.0
        )

    # ==========================================================
    # ELASTIC NET FEATURE WEIGHTING
    # ==========================================================

    def _elastic_weights(
        self,
        R,
        y,
        l1,
        l2
    ):
        """
        Projected Adam gradient descent.

        Objective:
            BCE(y, prediction)
            + L1 * sum(abs(weights))
            + L2 / 2 * sum(weights ** 2)

        Constraint:
            weights >= 0
        """

        # RMS scaling improves numerical conditioning.

        scale = np.maximum(
            np.sqrt(
                np.mean(R ** 2, axis=0)
            ),
            0.1
        )

        Z = R / scale

        u = np.zeros(
            Z.shape[1]
        )

        b = float(
            self._logit(y.mean())
        )

        # Adam optimizer states.

        m = np.zeros(
            len(u) + 1
        )

        v = np.zeros_like(m)

        for t in range(
            1,
            self.weight_iterations + 1
        ):

            score = (
                np.einsum(
                    "ij,j->i",
                    Z,
                    u
                )
                + b
            )

            error = (
                self._sigmoid(score)
                - y
            )

            # Regularization is applied to the
            # original, unscaled feature weights.
            #
            # Nonnegative constraint allows the L1
            # penalty to be represented as a linear term.

            grad = np.r_[
                np.einsum(
                    "ij,i->j",
                    Z,
                    error
                ) / len(y)
                + l1 / scale
                + l2 * u / scale**2,

                error.mean()
            ]

            # Adam first and second moments.

            m = (
                0.9 * m
                + 0.1 * grad
            )

            v = (
                0.999 * v
                + 0.001 * grad**2
            )

            # Adam bias correction and update.

            step = (
                self.learning_rate
                * (m / (1 - 0.9**t))
                / (
                    np.sqrt(
                        v / (1 - 0.999**t)
                    )
                    + 1e-8
                )
            )

            # Nonnegative projection.

            u = np.maximum(
                u - step[:-1],
                0.0
            )

            # Intercept is not regularized.

            b -= step[-1]

        return (
            u / scale,
            float(b)
        )

    # ==========================================================
    # CROSS-FITTED LIKELIHOOD ESTIMATION
    # ==========================================================

    def _crossfit_robust(
        self,
        keys,
        y,
        groups,
        config
    ):

        R = np.empty(
            (len(y), len(groups))
        )

        for ti, vi in self._robust_folds(
            y,
            self.n_folds,
            self.random_state + 17
        ):

            _, pipe = self._fit_robust_basis(
                self._subset_keys(keys, ti),
                y[ti],
                groups,
                config
            )

            R[vi] = self._robust_ratios(
                self._subset_keys(keys, vi),
                groups,
                pipe
            )

        if not np.isfinite(R).all():

            raise FloatingPointError(
                "Non-finite OOF likelihood ratio"
            )

        return R

    # ==========================================================
    # ROBUST OOF THRESHOLD CALIBRATION
    # ==========================================================

    def _calibrate_threshold(
        self,
        scores,
        y
    ):

        # Limit threshold search around log-odds zero.
        # Avoid fitting extreme classification thresholds
        # to small validation fluctuations.

        grid = np.linspace(
            -1.5,
            1.5,
            61
        )

        accuracy = np.array([
            np.mean(
                (scores >= t) == y
            )
            for t in grid
        ])

        # If several thresholds perform almost equally,
        # prefer the threshold closest to zero
        # (equivalent to probability 0.5).

        eligible = np.flatnonzero(
            accuracy
            >= accuracy.max() - self.threshold_tolerance
        )

        k = eligible[
            np.argmin(
                np.abs(grid[eligible])
            )
        ]

        return (
            float(accuracy[k]),
            float(grid[k])
        )

    # ==========================================================
    # NESTED OOF CONFIGURATION SELECTION
    # ==========================================================

    def _select_robust_config(
        self,
        keys,
        y,
        groups
    ):
        """
        Generate OOF predictions from the complete
        weighted classification pipeline.

        Every calibration fold independently learns:
            - Adaptive target encoding
            - Feature support selection
            - Gaussian density or direct evidence
            - Cross-fitted feature likelihoods
            - Elastic Net weights

        The held-out fold is used only for prediction.
        """

        # Reuse likelihood ratios for configurations
        # sharing identical preprocessing and density.

        by_basis = {}

        for config in self.robust_configs:

            by_basis.setdefault(
                config[:3],
                []
            ).append(config)

        scores = {
            cfg: np.empty(len(y))
            for cfg in self.robust_configs
        }

        folds = self._robust_folds(
            y,
            self.calibration_folds,
            self.random_state + 101
        )

        for fold, (ti, vi) in enumerate(
            folds,
            1
        ):

            self._log(
                f"E{self.optimization_stage}: "
                f"calibration fold "
                f"{fold}/{self.calibration_folds}"
            )

            kt = self._subset_keys(
                keys,
                ti
            )

            kv = self._subset_keys(
                keys,
                vi
            )

            yt = y[ti]

            for configs in by_basis.values():

                cfg = configs[0]

                # Generate training ratios without
                # using each sample's own label.

                R = self._crossfit_robust(
                    kt,
                    yt,
                    groups,
                    cfg
                )

                # Fit the fold inference pipeline.

                _, pipe = self._fit_robust_basis(
                    kt,
                    yt,
                    groups,
                    cfg
                )

                # Held-out fold evidence.

                Rv = self._robust_ratios(
                    kv,
                    groups,
                    pipe
                )

                # Compare Elastic Net configurations.

                for cfg in configs:

                    w, b = self._elastic_weights(
                        R,
                        yt,
                        cfg[3],
                        cfg[4]
                    )

                    scores[cfg][vi] = (
                        Rv @ w + b
                    )

        self.optimization_history_ = []

        best = None

        for cfg in self.robust_configs:

            acc, threshold = (
                self._calibrate_threshold(
                    scores[cfg],
                    y
                )
            )

            row = {
                "config": cfg,
                "accuracy": acc,
                "threshold": threshold
            }

            self.optimization_history_.append(
                row
            )

            if (
                best is None
                or acc > best["accuracy"]
            ):

                best = row

        self.calibration_scores_ = (
            scores[best["config"]]
        )

        return best

    # ==========================================================
    # PUBLIC TRAINING API
    # ==========================================================

    def fit(self, X, y=None):

        # Preserve E0-E6.

        if self.optimization_stage <= 6:

            return super().fit(
                X,
                y
            )

        self.is_fitted = False

        self._validate_dataframe(X)

        y = self._validate_target(
            X,
            X["is_canceled"].to_numpy()
            if y is None
            else y
        )

        # ------------------------------------------------------
        # 1. Feature engineering
        # ------------------------------------------------------

        keys = self._build_feature_keys(X)

        groups = self._candidate_groups(keys)

        # ------------------------------------------------------
        # 2. Nested configuration selection
        # ------------------------------------------------------

        best = self._select_robust_config(
            keys,
            y,
            groups
        )

        cfg = best["config"]

        self.robust_config_ = cfg

        self.validation_accuracy_ = (
            best["accuracy"]
        )

        self.log_lr_threshold_ = (
            best["threshold"]
        )

        self.threshold_ = float(
            self._sigmoid(
                self.log_lr_threshold_
            )
        )

        # ------------------------------------------------------
        # 3. Final cross-fitted likelihood ratios
        # ------------------------------------------------------

        R = self._crossfit_robust(
            keys,
            y,
            groups,
            cfg
        )

        # ------------------------------------------------------
        # 4. Final inference pipeline
        # ------------------------------------------------------

        self.X_fit_, pipe = (
            self._fit_robust_basis(
                keys,
                y,
                groups,
                cfg
            )
        )

        # ------------------------------------------------------
        # 5. Final Elastic Net feature weights
        # ------------------------------------------------------

        w, b = self._elastic_weights(
            R,
            y,
            cfg[3],
            cfg[4]
        )

        self.feature_weights_ = (
            w * pipe["active"]
        )

        self.intercept_ = b

        # ------------------------------------------------------
        # 6. Store final model parameters
        # ------------------------------------------------------

        self.robust_pipeline_ = pipe

        self.oof_likelihood_ratios_ = R

        self.candidate_groups_ = groups

        self.selected_groups_ = groups

        self.feature_names_ = [
            " × ".join(g)
            for g in groups
        ]

        self.n_features_in_ = len(groups)

        self.feature_smoothing_ = (
            pipe["alphas"]
        )

        self.category_maps_ = (
            pipe["maps"]
        )

        self.encoding_prior_ = (
            pipe["prior"]
        )

        self.class_counts_ = np.bincount(
            y,
            minlength=2
        )

        self.class_priors_ = (
            self.class_counts_ / len(y)
        )

        self.log_class_priors_ = np.log(
            self.class_priors_
        )

        self.regularization_ = cfg[4]

        self.l1_regularization_ = cfg[3]

        self.var_smoothing_ = (
            cfg[2]
            if self.optimization_stage == 7
            else None
        )

        self.n_components_ = (
            1
            if self.optimization_stage == 7
            else 0
        )

        self.theta_ = pipe.get("theta")

        self.var_ = pipe.get("var")

        self.scaler_mean_ = pipe.get("mean")

        self.scaler_std_ = pipe.get("std")

        self.is_fitted = True

        return self

    # ==========================================================
    # PUBLIC TRANSFORMATION API
    # ==========================================================

    def transform(self, X):

        if self.optimization_stage <= 6:

            return super().transform(X)

        self._check_fitted()

        pipe = self.robust_pipeline_

        keys = self._build_feature_keys(X)

        # E8 returns direct feature evidence,
        # not standardized Gaussian input.

        if self.optimization_stage == 8:

            return self._robust_ratios(
                keys,
                self.selected_groups_,
                pipe
            )

        # E7 returns standardized target encoding.

        encoded, _ = self._robust_apply(
            keys,
            self.selected_groups_,
            pipe
        )

        return self._transform_scaler(
            encoded,
            pipe["mean"],
            pipe["std"]
        )

    # ==========================================================
    # FINAL INFERENCE
    # ==========================================================

    def decision_function(self, X):

        if self.optimization_stage <= 6:

            return super().decision_function(X)

        self._check_fitted()

        R = self._robust_ratios(
            self._build_feature_keys(X),
            self.selected_groups_,
            self.robust_pipeline_
        )

        return (
            R @ self.feature_weights_
            + self.intercept_
        )

    # ==========================================================
    # FEATURE INFORMATION
    # ==========================================================

    def get_feature_info(self):

        if self.optimization_stage <= 6:

            return super().get_feature_info()

        self._check_fitted()

        info = pd.DataFrame({

            "feature": self.feature_names_,

            "weight": self.feature_weights_,

            "encoding_smoothing": [
                self.feature_smoothing_[g]
                for g in self.selected_groups_
            ],

            "sufficient_support": (
                self.robust_pipeline_["active"]
            ),

            "is_active": (
                self.feature_weights_ > 1e-8
            )
        })

        if self.optimization_stage == 7:

            for c in (0, 1):

                info[f"mean_class_{c}"] = (
                    self.theta_[c]
                )

                info[f"variance_class_{c}"] = (
                    self.var_[c]
                )

        return info

    # ==========================================================
    # MIXTURE INFORMATION
    # ==========================================================

    def get_mixture_info(self):

        if self.optimization_stage >= 7:

            raise RuntimeError(
                "E7/E8 do not use Gaussian mixtures."
            )

        return super().get_mixture_info()

    # ==========================================================
    # MODEL SUMMARY
    # ==========================================================

    def summary(self):

        if self.optimization_stage <= 6:

            return super().summary()

        self._check_fitted()

        print(
            f"E{self.optimization_stage}: "
            "Robust weighted "
            + (
                "Gaussian NB"
                if self.optimization_stage == 7
                else "Direct likelihood ratio NB"
            )
        )

        print(
            "Config (support, encoding, variance, L1, L2):",
            self.robust_config_
        )

        print(
            "OOF selection accuracy:",
            self.validation_accuracy_
        )

        print(
            "Probability threshold:",
            self.threshold_
        )

        print(
            self.get_feature_info()
            .sort_values(
                "weight",
                ascending=False
            )
            .head(15)
        )

# ==================================================================
# COST-SENSITIVE CLASSIFICATION
# ==================================================================

def predict_cost_sensitive(
    model,
    X,
    cost_fp=2.0,
    cost_fn=1.0
):
    """
    Bayesian cost-sensitive decision threshold:

        threshold = cost_fp / (cost_fp + cost_fn)

    With FP cost twice FN cost:
        threshold = 2 / 3

    Optimality assumes calibrated posterior probabilities.
    """

    if (
        not np.isfinite(cost_fp)
        or not np.isfinite(cost_fn)
        or cost_fp < 0
        or cost_fn < 0
        or cost_fp + cost_fn <= 0
    ):
        raise ValueError(
            "Costs must be finite, non-negative, "
            "and have a positive sum."
        )

    threshold = cost_fp / (
        cost_fp + cost_fn
    )

    probabilities = model.predict_proba(X)

    return (
        probabilities[:, 1] >= threshold
    ).astype(np.int64)


# ==================================================================
# BAYESIAN NETWORK GRAPH VISUALIZATION
# ==================================================================

def export_bayesian_network_graph(
    model,
    feature_names=None,
    target_name="is_canceled (Y)",
    output_path="output/plots/bayesian_network_graph.png"
):
    """
    Visualize the Gaussian Naive Bayes DAG.

    Each selected transformed feature is a child
    of the target class.

    This is a Naive Bayes graph, not TAN.
    """

    import matplotlib.pyplot as plt
    import networkx as nx
    import textwrap

    model._check_fitted()

    if feature_names is None:
        feature_names = model.feature_names_

    feature_names = list(feature_names)

    if len(feature_names) != model.n_features_in_:
        raise ValueError(
            "Invalid number of feature names."
        )

    graph = nx.DiGraph()

    target_node = "target"

    graph.add_node(target_node)

    labels = {
        target_node: target_name
    }

    feature_nodes = []

    for j, name in enumerate(feature_names):

        node = f"feature_{j}"

        graph.add_edge(
            target_node,
            node
        )

        feature_nodes.append(node)

        labels[node] = textwrap.fill(
            str(name),
            width=22
        )

    n_features = len(feature_nodes)

    positions = {
        target_node: (0, 0)
    }

    angles = np.linspace(
        np.pi / 2,
        np.pi / 2 + 2 * np.pi,
        n_features,
        endpoint=False
    )

    for node, angle in zip(
        feature_nodes,
        angles
    ):
        positions[node] = (
            3.0 * np.cos(angle),
            3.0 * np.sin(angle)
        )

    fig, ax = plt.subplots(
        figsize=(14, 11)
    )

    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=[target_node],
        node_color="#F59E0B",
        node_size=5500,
        ax=ax
    )

    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=feature_nodes,
        node_color="#93C5FD",
        node_size=5000,
        ax=ax
    )

    nx.draw_networkx_edges(
        graph,
        positions,
        arrows=True,
        arrowsize=22,
        width=1.8,
        node_size=5000,
        ax=ax
    )

    nx.draw_networkx_labels(
        graph,
        positions,
        labels=labels,
        font_size=9,
        ax=ax
    )

    ax.set_title(
        "Gaussian Naive Bayes - Bayesian Network",
        fontsize=17,
        fontweight="bold"
    )

    ax.margins(0.2)
    ax.axis("off")

    fig.tight_layout()

    os.makedirs(
        os.path.dirname(output_path) or ".",
        exist_ok=True
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"[Graph Saved] {output_path}"
    )

    return graph
