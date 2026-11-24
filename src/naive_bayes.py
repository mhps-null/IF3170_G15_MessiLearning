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

"""

import os
import pickle
import numpy as np
import pandas as pd
from .base import BaseModel

class _GaussianNaiveBayesBase(BaseModel):
    # FEATURE CONFIGURATION
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
        "lead": ("days_before_arrival", [0, 1, 3, 7, 14, 21, 30, 45, 60, 90, 120, 150, 180, 240, 300]),
        "rate": (
            "nightly_rate_idr",
            [0, 500000, 1000000, 1200000, 1400000, 1600000, 1800000, 2000000, 2200000, 2500000, 3000000]
        ),
        "history": ("prior_completed_bookings", [0, 0.0001]),
    }
    FEATURE_INTERACTIONS = (
        ("arrival_year", "arrival_month"),
        ("arrival_year", "arrival_month", "booking_channel"),
        ("arrival_year", "arrival_month", "arrival_day"),
        ("lead", "booking_channel"),
        ("lead", "special_requests"),
        ("rate", "booking_channel"),
        ("rate", "room_category"),
        ("rate", "arrival_year", "arrival_month"),
        ("booking_channel", "special_requests"),
        ("booking_channel", "meal_package"),
        ("booking_channel", "total_nights"),
        ("lead", "arrival_year", "arrival_month"),
        ("lead", "rate"),
        ("room_category", "num_adults"),
        ("arrival_year", "arrival_month", "lead", "booking_channel"),
        ("arrival_year", "arrival_month", "arrival_day", "booking_channel"),
        ("lead", "rate", "booking_channel"),
        ("lead", "booking_channel", "special_requests"),
        ("arrival_year", "arrival_month", "arrival_day", "lead"),
        ("rate", "booking_channel", "special_requests"),
    )
    MISSING_VALUE = "__NA__"

    # INITIALIZATION
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
            raise ValueError("encoding_smoothing must be positive.")
        if not 0 < validation_size < 1:
            raise ValueError("validation_size must be between 0 and 1.")
        if n_folds < 2:
            raise ValueError("n_folds must be at least 2.")
        if max_features < 1:
            raise ValueError("max_features must be positive.")
        if (
            len(variance_candidates) == 0
            or not np.all(np.isfinite(variance_candidates))
            or np.any(np.asarray(variance_candidates) <= 0)
        ):
            raise ValueError("variance_candidates must contain " "positive finite values.")
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

    # INPUT VALIDATION
    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted.")

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
            raise ValueError("X contains duplicated column names.")

    def _validate_target(self, X, y):
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be a one-dimensional array " "with the same length as X.")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("Target must contain binary labels {0, 1}.")
        y = y.astype(np.int64)
        class_counts = np.bincount(y, minlength=2)
        if np.any(class_counts < self.n_folds + 2):
            raise ValueError("Insufficient samples in one or more classes.")
        return y

    # FEATURE ENGINEERING
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
            X["total_nights"] = (X["weekend_nights"] + X["weekday_nights"])
        if "arrival_day_of_week" not in X.columns:
            dates = pd.to_datetime(
                {"year": X["arrival_year"], "month": X["arrival_month"], "day": X["arrival_day"],},
                errors="coerce"
            )
            X["arrival_day_of_week"] = dates.dt.dayofweek
        keys = {}
        # Categorical and discrete numerical features
        for name in self.CATEGORICAL_FEATURES:
            if name not in X.columns:
                raise ValueError(f"Required feature missing: {name}")
            keys[name] = (X[name].astype("string").fillna(self.MISSING_VALUE).to_numpy(dtype=str))
        # Continuous numerical features
        for alias, config in self.NUMERICAL_BINS.items():
            name, edges = config
            if name not in X.columns:
                raise ValueError(f"Required feature missing: {name}")
            values = pd.to_numeric(X[name], errors="coerce").to_numpy(dtype=np.float64)
            valid = np.isfinite(values)
            codes = np.full(len(X), self.MISSING_VALUE, dtype=object)
            codes[valid] = np.searchsorted(np.asarray(edges), values[valid], side="right").astype(str)
            keys[alias] = codes.astype(str)
        return keys

    @staticmethod
    def _subset_keys(keys, indices):
        return {name: values[indices] for name, values in keys.items()}

    def _candidate_groups(self, keys):
        individual = [(name,) for name in keys]
        return individual + list(self.FEATURE_INTERACTIONS)

    @staticmethod
    def _group_key(keys, group):
        """
        Example:

        group = ("lead", "booking_channel")

        Each row becomes a tuple:
            ("5", "Online Travel Agent")
        """
        return pd.Series(list(zip(*(keys[name] for name in group))), dtype=object)

    # STRATIFIED SPLITTING FROM SCRATCH
    def _train_validation_split(self, y):
        rng = np.random.default_rng(self.random_state)
        train_indices = []
        val_indices = []
        for c in self.classes_:
            indices = np.flatnonzero(y == c)
            indices = rng.permutation(indices)
            n_val = max(1, int(round(self.validation_size * len(indices))))
            if len(indices) - n_val < self.n_folds:
                raise ValueError("Insufficient training samples " "for out-of-fold encoding.")
            val_indices.extend(indices[:n_val])
            train_indices.extend(indices[n_val:])
        return (np.asarray(train_indices, dtype=np.int64), np.asarray(val_indices, dtype=np.int64))

    def _stratified_folds(self, y):
        rng = np.random.default_rng(self.random_state)
        class_partitions = []
        for c in self.classes_:
            indices = np.flatnonzero(y == c)
            indices = rng.permutation(indices)
            partitions = np.array_split(indices, self.n_folds)
            class_partitions.append(partitions)
        for fold in range(self.n_folds):
            holdout = np.concatenate([partitions[fold] for partitions in class_partitions])
            mask = np.ones(len(y), dtype=bool)
            mask[holdout] = False
            train = np.flatnonzero(mask)
            yield train, holdout

    # SMOOTHED TARGET ENCODING
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
        frame = pd.DataFrame({"key": keys.to_numpy(), "target": y})
        stats = (frame.groupby("key")["target"].agg(["sum", "count"]))
        mapping = (stats["sum"] + self.encoding_smoothing * prior) / (
            stats["count"]
            + self.encoding_smoothing
        )
        return mapping, prior

    @staticmethod
    def _apply_category_map(keys, mapping, prior):
        return (keys.map(mapping).fillna(prior).to_numpy(dtype=np.float64))

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
        encoded = np.empty((n_samples, n_features), dtype=np.float64)
        mappings = {}
        folds = list(self._stratified_folds(y))
        for j, group in enumerate(groups):
            group_keys = self._group_key(keys, group)
            # Out-of-fold encoding
            for train_idx, holdout_idx in folds:
                mapping, prior = self._learn_category_map(group_keys.iloc[train_idx], y[train_idx])
                encoded[holdout_idx, j] = (
                    self._apply_category_map(group_keys.iloc[holdout_idx], mapping, prior)
                )
            # Full-training mapping for inference
            full_mapping, full_prior = (self._learn_category_map(group_keys, y))
            mappings[group] = full_mapping
        return encoded, mappings, float(np.mean(y))

    def _transform_encoder(self, keys, groups, mappings, prior):
        n_samples = len(next(iter(keys.values())))
        encoded = np.empty((n_samples, len(groups)), dtype=np.float64)
        for j, group in enumerate(groups):
            group_keys = self._group_key(keys, group)
            encoded[:, j] = (self._apply_category_map(group_keys, mappings[group], prior))
        return encoded

    # STANDARDIZATION
    @staticmethod
    def _fit_scaler(X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std = np.maximum(std, 1e-8)
        return mean, std

    @staticmethod
    def _transform_scaler(X, mean, std):
        return (X - mean) / std

    # GAUSSIAN NAIVE BAYES FROM SCRATCH
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
            means.append(np.mean(X_c, axis=0))
            variances.append(np.var(X_c, axis=0) + var_smoothing)
            counts.append(len(X_c))
        means = np.asarray(means, dtype=np.float64)
        variances = np.asarray(variances, dtype=np.float64)
        counts = np.asarray(counts, dtype=np.int64)
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
        return -0.5 * (np.log(2.0 * np.pi * variance) + ((X - mean) ** 2) / variance)

    @classmethod
    def _feature_log_likelihood_ratio(cls, X, means, variances):
        """
        Per-feature log-likelihood ratio:

        log p(x_j | y=1) - log p(x_j | y=0)

        The common log(2*pi) term cancels.
        """
        log_prob_0 = -0.5 * (np.log(variances[0]) + (X - means[0]) ** 2 / variances[0])
        log_prob_1 = -0.5 * (np.log(variances[1]) + (X - means[1]) ** 2 / variances[1])
        return log_prob_1 - log_prob_0

    # DECISION THRESHOLD OPTIMIZATION
    @staticmethod
    def _find_best_threshold(scores, y):
        candidates = np.linspace(-5.0, 5.0, 101)
        accuracies = np.array([np.mean((scores >= threshold) == y) for threshold in candidates])
        best_idx = int(np.argmax(accuracies))
        return (float(accuracies[best_idx]), float(candidates[best_idx]))

    # AUTOMATIC FEATURE SELECTION
    def _forward_feature_selection(self, likelihood_ratios, y):
        """
        Greedy Sequential Forward Selection.

        At each iteration, add the feature that
        produces the highest internal validation accuracy.

        Stop when improvement is insignificant.
        """
        n_features = likelihood_ratios.shape[1]
        selected = []
        current_accuracy = 0.0
        max_features = min(self.max_features, n_features)
        for _ in range(max_features):
            best_candidate = None
            for j in range(n_features):
                if j in selected:
                    continue
                candidate = selected + [j]
                scores = (likelihood_ratios[:, candidate].sum(axis=1))
                accuracy, _ = (self._find_best_threshold(scores, y))
                result = (accuracy, j)
                if (best_candidate is None or result > best_candidate):
                    best_candidate = result
            if best_candidate is None:
                break
            accuracy, feature_idx = best_candidate
            if (accuracy <= current_accuracy + self.min_improvement):
                break
            selected.append(feature_idx)
            current_accuracy = accuracy
        if not selected:
            raise RuntimeError("Feature selection failed.")
        scores = (likelihood_ratios[:, selected].sum(axis=1))
        accuracy, threshold = (self._find_best_threshold(scores, y))
        return selected, accuracy, threshold

    # HYPERPARAMETER OPTIMIZATION
    def _optimize(self, keys, y, groups):
        """
        Select:
            - Feature subset
            - Variance smoothing
            - Classification threshold

        Only the internal training-validation split
        is used during optimization.
        """
        train_idx, val_idx = (self._train_validation_split(y))
        keys_train = self._subset_keys(keys, train_idx)
        keys_val = self._subset_keys(keys, val_idx)
        y_train = y[train_idx]
        y_val = y[val_idx]
        # Fit encoder using internal training only.
        X_train, mappings, prior = self._fit_encoder(keys_train, y_train, groups)
        X_val = self._transform_encoder(keys_val, groups, mappings, prior)
        # Fit scaler using internal training only.
        mean, std = self._fit_scaler(X_train)
        X_train = self._transform_scaler(X_train, mean, std)
        X_val = self._transform_scaler(X_val, mean, std)
        best = None
        self.optimization_history_ = []
        # Grid search over variance smoothing values.
        for smoothing in self.variance_candidates:
            means, variances, _, _ = (self._fit_gaussian(X_train, y_train, smoothing))
            ratios = (self._feature_log_likelihood_ratio(X_val, means, variances))
            selected, accuracy, threshold = (self._forward_feature_selection(ratios, y_val))
            result = {
                "accuracy": accuracy,
                "var_smoothing": float(smoothing),
                "threshold": threshold,
                "selected_indices": selected,
                "selected_groups": [groups[j] for j in selected],
            }
            self.optimization_history_.append(result)
            if (best is None or accuracy > best["accuracy"]):
                best = result
        return best

    # PUBLIC FIT API
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
                raise ValueError("Provide y or an is_canceled column.")
            y = X["is_canceled"].to_numpy()
        y = self._validate_target(X, y)
        # 1. Feature engineering
        keys = self._build_feature_keys(X)
        groups = self._candidate_groups(keys)
        self.candidate_groups_ = groups
        # 2. Internal model optimization
        best = self._optimize(keys, y, groups)
        self.selected_groups_ = (best["selected_groups"])
        self.var_smoothing_ = (best["var_smoothing"])
        self.log_lr_threshold_ = (best["threshold"])
        self.validation_accuracy_ = (best["accuracy"])
        self.feature_names_ = [" × ".join(group) for group in self.selected_groups_]
        # 3. Refit selected encoder on all supplied
        # training data using out-of-fold encoding.
        encoded, mappings, prior = (self._fit_encoder(keys, y, self.selected_groups_))
        self.category_maps_ = mappings
        self.encoding_prior_ = prior
        # 4. Standardization
        (self.scaler_mean_, self.scaler_std_) = self._fit_scaler(encoded)
        X_train = self._transform_scaler(encoded, self.scaler_mean_, self.scaler_std_)
        # 5. Fit final Gaussian Naive Bayes
        (self.theta_, self.var_, self.class_counts_, self.class_priors_) = self._fit_gaussian(
            X_train,
            y,
            self.var_smoothing_
        )
        self.log_class_priors_ = np.log(self.class_priors_)
        # Convert likelihood-ratio threshold into
        # equivalent posterior-probability threshold.
        prior_log_odds = (self.log_class_priors_[1] - self.log_class_priors_[0])
        posterior_log_odds = (self.log_lr_threshold_ + prior_log_odds)
        self.threshold_ = float(1.0 / (1.0 + np.exp(-posterior_log_odds)))
        # Keep exact OOF training features to enable
        # fair comparison with sklearn GaussianNB.
        self.X_fit_ = X_train
        self.n_features_in_ = (X_train.shape[1])
        self.is_fitted = True
        return self

    # PUBLIC PREPROCESSING API
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
        return self._transform_scaler(encoded, self.scaler_mean_, self.scaler_std_)

    # INFERENCE
    def _compute_log_likelihood(self, X):
        X_transformed = self.transform(X)
        n_samples = len(X_transformed)
        scores = np.empty((n_samples, 2), dtype=np.float64)
        for c in self.classes_:
            log_density = self._gaussian_log_pdf(X_transformed, self.theta_[c], self.var_[c])
            scores[:, c] = (self.log_class_priors_[c] + log_density.sum(axis=1))
        return scores

    def predict_log_proba(self, X):
        scores = self._compute_log_likelihood(X)
        row_max = np.max(scores, axis=1, keepdims=True)
        shifted = scores - row_max
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        return shifted - log_sum_exp

    def predict_proba(self, X):
        """
        Return posterior probabilities:

            column 0 = P(y=0 | X)
            column 1 = P(y=1 | X)
        """
        return np.exp(self.predict_log_proba(X))

    def predict(self, X, threshold=None):
        """
        By default, use optimized classification threshold.

        To use the standard Bayesian threshold:
            model.predict(X, threshold=0.5)
        """
        self._check_fitted()
        if threshold is None:
            transformed = self.transform(X)
            ratios = (self._feature_log_likelihood_ratio(transformed, self.theta_, self.var_))
            return (ratios.sum(axis=1) >= self.log_lr_threshold_).astype(np.int64)
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1.")
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] >= threshold).astype(np.int64)

    def score(self, X, y):
        y = np.asarray(y)
        predictions = self.predict(X)
        if y.shape != predictions.shape:
            raise ValueError("Target and predictions must have " "the same shape.")
        return float(np.mean(predictions == y))

    # MODEL INFORMATION
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
                "n_features": len(result["selected_groups"]),
            }
            for result in self.optimization_history_
        ])

    def summary(self):
        self._check_fitted()
        print("=" * 60)
        print("GAUSSIAN NAIVE BAYES FROM SCRATCH")
        print("=" * 60)
        print("Candidate features    :", len(self.candidate_groups_))
        print("Selected features     :", self.n_features_in_)
        print("Variance smoothing    :", self.var_smoothing_)
        print("Encoding smoothing    :", self.encoding_smoothing)
        print("Decision threshold    :", round(self.threshold_, 6))
        print("Internal validation   :", f"{self.validation_accuracy_:.4%}")
        print("Class priors          :", self.class_priors_)
        print("\nSelected features:")
        for i, name in enumerate(self.feature_names_, start=1):
            print(f"  {i}. {name}")
        print("=" * 60)

    # MODEL SERIALIZATION
    def save(self, filepath):
        self._check_fitted()
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "wb") as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Model Saved] {filepath}")

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as file:
            model = pickle.load(file)
        if not isinstance(model, cls):
            raise TypeError("Invalid NaiveBayes model.")
        model._check_fitted()
        return model

class NaiveBayes(_GaussianNaiveBayesBase):
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

    # INITIALIZATION
    def __init__(
        self,
        optimization_stage=3,
        regularization_candidates=(0.001, 0.01, 0.1),
        smoothing_candidates=(1.0, 3.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0),
        learning_rate=0.03,
        weight_iterations=200,
        **kwargs
    ):
        super().__init__(**kwargs)
        if optimization_stage not in (0, 1, 2, 3):
            raise ValueError("optimization_stage must be 0, 1, 2, or 3.")
        for candidates in (regularization_candidates, smoothing_candidates):
            values = np.asarray(candidates, dtype=float)
            if (
                values.ndim != 1
                or len(values) == 0
                or not np.all(np.isfinite(values))
                or np.any(values <= 0)
            ):
                raise ValueError("Candidates must contain positive " "finite values.")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite.")
        if weight_iterations < 1:
            raise ValueError("weight_iterations must be positive.")
        self.optimization_stage = optimization_stage
        self.regularization_candidates = tuple(regularization_candidates)
        self.smoothing_candidates = tuple(smoothing_candidates)
        self.learning_rate = float(learning_rate)
        self.weight_iterations = int(weight_iterations)
        self.feature_weights_ = None
        self.intercept_ = None
        self.regularization_ = None
        self.feature_smoothing_ = {}

    # STAGE 2: ADVANCED FEATURE ENGINEERING
    def _build_feature_keys(self, X):
        keys = super()._build_feature_keys(X)
        if self.optimization_stage < 2:
            return keys
        # ------------------------------------------------------
        # Helper: categorical representation
        # ------------------------------------------------------
        def add_category(name, values):
            keys[name] = (pd.Series(values).astype("string").fillna(self.MISSING_VALUE).to_numpy(dtype=str))
        # ------------------------------------------------------
        # Helper: numerical discretization
        # ------------------------------------------------------
        def add_bins(name, values, edges):
            values = np.asarray(values, dtype=np.float64)
            result = np.full(len(values), self.MISSING_VALUE, dtype=object)
            valid = np.isfinite(values)
            result[valid] = np.searchsorted(np.asarray(edges), values[valid], side="right").astype(str)
            keys[name] = result.astype(str)
        # ------------------------------------------------------
        # Original numerical features
        # ------------------------------------------------------
        lead = X["days_before_arrival"]
        rate = X["nightly_rate_idr"]
        nights = (X["weekday_nights"] + X["weekend_nights"])
        guests = (X["num_adults"] + X["num_children"])
        # ------------------------------------------------------
        # Reconstruct arrival and booking dates
        # ------------------------------------------------------
        arrival_date = pd.to_datetime(
            {"year": X["arrival_year"], "month": X["arrival_month"], "day": X["arrival_day"]},
            errors="coerce"
        )
        booking_date = (arrival_date - pd.to_timedelta(lead, unit="D", errors="coerce"))
        add_category("arrival_date", arrival_date.dt.strftime("%Y-%m-%d"))
        add_category("booking_date", booking_date.dt.strftime("%Y-%m-%d"))
        add_category("booking_month", booking_date.dt.strftime("%Y-%m"))
        add_category("booking_week", booking_date.dt.strftime("%G-%V"))
        add_category("booking_dow", booking_date.dt.dayofweek)
        # ------------------------------------------------------
        # Exact numerical values as categorical keys
        # ------------------------------------------------------
        add_category("lead_exact", lead)
        add_category("rate_exact", rate)
        add_category("guests", guests)
        add_category("prior_cancel", X["prior_cancellations"])
        # ------------------------------------------------------
        # Fine-grained numerical binning
        # ------------------------------------------------------
        add_bins("lead_fine", lead, np.arange(0, 365, 5))
        add_bins("rate_fine", rate, np.arange(0, 4000001, 50000))
        # ------------------------------------------------------
        # Estimated booking value
        # ------------------------------------------------------
        add_bins("value", rate * nights, [0, 1e6, 2e6, 3e6, 4e6, 5e6, 7e6, 10e6, 15e6, 20e6])
        # ------------------------------------------------------
        # Price per guest per night
        # ------------------------------------------------------
        add_bins(
            "rate_guest",
            rate / guests.replace(0, np.nan),
            [0, 400000, 600000, 800000, 1000000, 1200000, 1500000, 2000000]
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

    # STAGE 2: ADVANCED FEATURE INTERACTION
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
        contextual_features = ("booking_channel", "special_requests", "room_category", "lead", "rate")
        # Automatically generate pairwise interactions.
        for first in derived_features:
            for second in contextual_features:
                groups.append((first, second))
        # Domain-specific higher-order interactions.
        groups.extend([
            ("arrival_date", "lead_exact"),
            ("arrival_date", "lead_fine"),
            ("booking_date", "arrival_date"),
            ("booking_date", "booking_channel", "special_requests"),
            ("booking_month", "lead", "booking_channel"),
            ("arrival_date", "booking_channel", "special_requests"),
            ("lead_fine", "booking_channel", "special_requests"),
            ("rate_fine", "booking_channel", "special_requests"),
            ("arrival_date", "rate_fine"),
            ("booking_week", "lead", "booking_channel"),
            ("lead_exact", "rate_exact"),
            ("rate_exact", "room_category", "booking_channel"),
            ("arrival_date", "room_category", "booking_channel"),
            ("booking_date", "rate_fine"),
            ("rate_fine", "arrival_year", "arrival_month"),
            ("arrival_date", "lead", "booking_channel")
        ])
        # Remove duplicate groups while preserving order.
        return list(dict.fromkeys(groups))

    # STAGE 3: ADAPTIVE TARGET ENCODING
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
            result = super()._fit_encoder(keys, y, groups)
            self.feature_smoothing_ = {group: self.encoding_smoothing for group in groups}
            return result
        n_samples = len(y)
        n_features = len(groups)
        encoded = np.empty((n_samples, n_features), dtype=np.float64)
        mappings = {}
        selected_smoothing = {}
        folds = list(self._stratified_folds(y))
        global_prior = float(np.mean(y))
        if self.optimization_stage >= 3:
            candidates = np.asarray(self.smoothing_candidates, dtype=np.float64)
        else:
            candidates = np.array([self.encoding_smoothing], dtype=np.float64)
        for j, group in enumerate(groups):
            group_keys = self._group_key(keys, group)
            # Integer identifiers are used only for
            # efficient grouping, not as model features.
            codes, unique_categories = pd.factorize(group_keys, sort=False)
            n_categories = len(unique_categories)
            # Every column corresponds to one alpha.
            candidate_encoding = np.empty((n_samples, len(candidates)), dtype=np.float64)
            # --------------------------------------------------
            # Out-of-fold encoding for every alpha
            # --------------------------------------------------
            for train_idx, holdout_idx in folds:
                train_codes = codes[train_idx]
                counts = np.bincount(train_codes, minlength=n_categories)
                sums = np.bincount(train_codes, weights=y[train_idx], minlength=n_categories)
                fold_prior = float(np.mean(y[train_idx]))
                holdout_codes = codes[holdout_idx]
                candidate_encoding[holdout_idx] = (sums[holdout_codes, None] + candidates * fold_prior) / (
                    counts[holdout_codes, None]
                    + candidates
                )
            # --------------------------------------------------
            # Select alpha with minimum OOF Brier score
            # --------------------------------------------------
            brier_scores = np.mean((candidate_encoding - y[:, None]) ** 2, axis=0)
            best_idx = int(np.argmin(brier_scores))
            best_alpha = float(candidates[best_idx])
            encoded[:, j] = (candidate_encoding[:, best_idx])
            selected_smoothing[group] = best_alpha
            # --------------------------------------------------
            # Refit full-training mapping for inference
            # --------------------------------------------------
            full_counts = np.bincount(codes, minlength=n_categories)
            full_sums = np.bincount(codes, weights=y, minlength=n_categories)
            full_encoding = (full_sums + best_alpha * global_prior) / (full_counts + best_alpha)
            mappings[group] = pd.Series(full_encoding, index=unique_categories)
        self.feature_smoothing_ = selected_smoothing
        return (encoded, mappings, global_prior)

    # NUMERICALLY STABLE SIGMOID
    @staticmethod
    def _sigmoid(z):
        return np.exp(-np.logaddexp(0.0, -z))

    # STAGE 1: REGULARIZED FEATURE WEIGHTING
    def _learn_feature_weights(self, likelihood_ratios, y, class_priors, regularization):
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
        R = np.asarray(likelihood_ratios, dtype=np.float64)
        n_samples, n_features = R.shape
        # Scale optimization coordinates to improve
        # numerical conditioning.
        scale = np.maximum(np.sqrt(np.mean(R ** 2, axis=0)), 0.1)
        Z = R / scale
        # Initialize weights at zero and the intercept
        # at empirical class prior log-odds.
        prior_log_odds = np.log(class_priors[1] / class_priors[0])
        parameters = np.concatenate([np.zeros(n_features), [prior_log_odds]])
        first_moment = np.zeros_like(parameters)
        second_moment = np.zeros_like(parameters)
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        # ------------------------------------------------------
        # Adam optimization loop
        # ------------------------------------------------------
        for step in range(1, self.weight_iterations + 1):
            weights_scaled = parameters[:-1]
            intercept = parameters[-1]
            logits = (Z @ weights_scaled + intercept)
            probabilities = self._sigmoid(logits)
            error = probabilities - y
            # Gradient of BCE + L2.
            # Divide by scale^2 because regularization
            # applies to original feature weights.
            gradient_weights = (Z.T @ error / n_samples + regularization * weights_scaled / (scale ** 2))
            gradient_intercept = np.mean(error)
            gradient = np.concatenate([gradient_weights, [gradient_intercept]])
            # Adam first moment.
            first_moment = (beta1 * first_moment + (1.0 - beta1) * gradient)
            # Adam second moment.
            second_moment = (beta2 * second_moment + (1.0 - beta2) * gradient ** 2)
            # Bias correction.
            corrected_first = (first_moment / (1.0 - beta1 ** step))
            corrected_second = (second_moment / (1.0 - beta2 ** step))
            # Update.
            parameters -= (self.learning_rate * corrected_first / (np.sqrt(corrected_second) + epsilon))
            # Project feature weights into nonnegative
            # space. The intercept is unconstrained.
            parameters[:-1] = np.maximum(parameters[:-1], 0.0)
        # Convert back to original likelihood ratio space.
        weights = parameters[:-1] / scale
        intercept = float(parameters[-1])
        return weights, intercept

    # WEIGHTED MODEL HYPERPARAMETER OPTIMIZATION
    def _optimize_weighted(self, keys, y, groups):
        train_idx, val_idx = (self._train_validation_split(y))
        train_keys = self._subset_keys(keys, train_idx)
        val_keys = self._subset_keys(keys, val_idx)
        y_train = y[train_idx]
        y_val = y[val_idx]
        # Fit all encoding parameters using the
        # internal training subset only.
        X_train, mappings, prior = (self._fit_encoder(train_keys, y_train, groups))
        X_val = self._transform_encoder(val_keys, groups, mappings, prior)
        mean, std = self._fit_scaler(X_train)
        X_train = self._transform_scaler(X_train, mean, std)
        X_val = self._transform_scaler(X_val, mean, std)
        best = None
        self.optimization_history_ = []
        # ------------------------------------------------------
        # Grid search variance smoothing and regularization
        # ------------------------------------------------------
        for smoothing in self.variance_candidates:
            means, variances, _, priors = (self._fit_gaussian(X_train, y_train, smoothing))
            train_ratios = (self._feature_log_likelihood_ratio(X_train, means, variances))
            val_ratios = (self._feature_log_likelihood_ratio(X_val, means, variances))
            for regularization in (self.regularization_candidates):
                weights, intercept = (
                    self._learn_feature_weights(train_ratios, y_train, priors, regularization)
                )
                scores = (val_ratios @ weights + intercept)
                accuracy, threshold = (self._find_best_threshold(scores, y_val))
                result = {
                    "accuracy": float(accuracy),
                    "var_smoothing": float(smoothing),
                    "regularization": float(regularization),
                    "threshold": float(threshold),
                    "active_features": int(np.sum(weights > 1e-8))
                }
                self.optimization_history_.append(result)
                if (best is None or accuracy > best["accuracy"]):
                    best = result.copy()
        return best

    # PUBLIC FIT API
    def fit(self, X, y=None):
        # ------------------------------------------------------
        # Stage 0: reproduce the original Gaussian NB
        # ------------------------------------------------------
        if self.optimization_stage == 0:
            super().fit(X, y)
            self.feature_weights_ = np.ones(self.n_features_in_)
            self.intercept_ = float(self.log_class_priors_[1] - self.log_class_priors_[0])
            self.regularization_ = None
            return self
        # ------------------------------------------------------
        # Stages 1, 2, and 3
        # ------------------------------------------------------
        self.is_fitted = False
        self._validate_dataframe(X)
        if y is None:
            if "is_canceled" not in X.columns:
                raise ValueError("Provide y or an is_canceled column.")
            y = X["is_canceled"].to_numpy()
        y = self._validate_target(X, y)
        # 1. Integrated feature engineering.
        keys = self._build_feature_keys(X)
        groups = self._candidate_groups(keys)
        self.candidate_groups_ = groups
        # 2. Optimize using internal validation.
        best = self._optimize_weighted(keys, y, groups)
        self.var_smoothing_ = best["var_smoothing"]
        self.regularization_ = best["regularization"]
        self.log_lr_threshold_ = best["threshold"]
        self.validation_accuracy_ = best["accuracy"]
        # Unlike stage 0, weighted stages use continuous
        # feature weighting instead of hard forward selection.
        self.selected_groups_ = groups
        self.feature_names_ = [" × ".join(group) for group in groups]
        # 3. Refit preprocessing on all supplied
        # training samples.
        (encoded, self.category_maps_, self.encoding_prior_) = self._fit_encoder(keys, y, groups)
        # 4. Standardization.
        (self.scaler_mean_, self.scaler_std_) = self._fit_scaler(encoded)
        X_train = self._transform_scaler(encoded, self.scaler_mean_, self.scaler_std_)
        # 5. Train Gaussian Naive Bayes parameters.
        (self.theta_, self.var_, self.class_counts_, self.class_priors_) = self._fit_gaussian(
            X_train,
            y,
            self.var_smoothing_
        )
        self.log_class_priors_ = np.log(self.class_priors_)
        # 6. Generate Gaussian likelihood ratios.
        ratios = self._feature_log_likelihood_ratio(X_train, self.theta_, self.var_)
        # 7. Learn final feature weights.
        (
            self.feature_weights_,
            self.intercept_
        ) = self._learn_feature_weights(ratios, y, self.class_priors_, self.regularization_)
        # 8. Convert optimized log-odds threshold
        # to probability threshold.
        self.threshold_ = float(self._sigmoid(self.log_lr_threshold_))
        # Preserve compatibility with the previous API.
        self.X_fit_ = X_train
        self.n_features_in_ = X_train.shape[1]
        self.is_fitted = True
        return self

    # WEIGHTED INFERENCE
    def decision_function(self, X):
        self._check_fitted()
        X_transformed = self.transform(X)
        ratios = self._feature_log_likelihood_ratio(X_transformed, self.theta_, self.var_)
        return (ratios @ self.feature_weights_ + self.intercept_)

    def _compute_log_likelihood(self, X):
        if self.optimization_stage == 0:
            return super()._compute_log_likelihood(X)
        scores = self.decision_function(X)
        # Equivalent binary class scores up to
        # a common additive constant.
        # The inherited predict_log_proba() will
        # normalize these using log-sum-exp.
        return np.column_stack([np.zeros(len(scores)), scores])

    def predict(self, X, threshold=None):
        self._check_fitted()
        if self.optimization_stage == 0:
            return super().predict(X, threshold=threshold)
        if threshold is None:
            scores = self.decision_function(X)
            return (scores >= self.log_lr_threshold_).astype(np.int64)
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1.")
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] >= threshold).astype(np.int64)

    # FEATURE INFORMATION
    def get_feature_info(self):
        self._check_fitted()
        information = super().get_feature_info()
        information["weight"] = (self.feature_weights_)
        information["encoding_smoothing"] = [
            self.feature_smoothing_[group]
            for group in self.selected_groups_
        ]
        information["is_active"] = (information["weight"] > 1e-8)
        return information

    def get_optimization_history(self):
        self._check_fitted()
        if self.optimization_stage == 0:
            return super().get_optimization_history()
        return pd.DataFrame(self.optimization_history_)

    # MODEL SUMMARY
    def summary(self):
        self._check_fitted()
        if self.optimization_stage == 0:
            return super().summary()
        active_features = int(np.sum(self.feature_weights_ > 1e-8))
        print("=" * 65)
        print("OPTIMIZED WEIGHTED GAUSSIAN NAIVE BAYES")
        print("=" * 65)
        print("Optimization stage      :", self.optimization_stage)
        print("Candidate features      :", len(self.candidate_groups_))
        print("Active features         :", active_features)
        print("Variance smoothing      :", self.var_smoothing_)
        print("L2 regularization       :", self.regularization_)
        print("Learning rate           :", self.learning_rate)
        print("Weight iterations       :", self.weight_iterations)
        print("Decision threshold      :", round(self.threshold_, 6))
        print("Internal validation     :", f"{self.validation_accuracy_:.4%}")
        print("Class priors            :", self.class_priors_)
        print("\nTop 15 features by weight:")
        information = (self.get_feature_info().sort_values("weight", ascending=False))
        print(information[["feature", "weight", "encoding_smoothing"]].head(15).to_string(index=False))
        print("=" * 65)

# COST-SENSITIVE CLASSIFICATION
def predict_cost_sensitive(model, X, cost_fp=2.0, cost_fn=1.0):
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
        raise ValueError("Costs must be finite, non-negative, " "and have a positive sum.")
    threshold = cost_fp / (cost_fp + cost_fn)
    probabilities = model.predict_proba(X)
    return (probabilities[:, 1] >= threshold).astype(np.int64)

# BAYESIAN NETWORK GRAPH VISUALIZATION
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
        raise ValueError("Invalid number of feature names.")
    graph = nx.DiGraph()
    target_node = "target"
    graph.add_node(target_node)
    labels = {target_node: target_name}
    feature_nodes = []
    for j, name in enumerate(feature_names):
        node = f"feature_{j}"
        graph.add_edge(target_node, node)
        feature_nodes.append(node)
        labels[node] = textwrap.fill(str(name), width=22)
    n_features = len(feature_nodes)
    positions = {target_node: (0, 0)}
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n_features, endpoint=False)
    for node, angle in zip(feature_nodes, angles):
        positions[node] = (3.0 * np.cos(angle), 3.0 * np.sin(angle))
    fig, ax = plt.subplots(figsize=(14, 11))
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
    nx.draw_networkx_edges(graph, positions, arrows=True, arrowsize=22, width=1.8, node_size=5000, ax=ax)
    nx.draw_networkx_labels(graph, positions, labels=labels, font_size=9, ax=ax)
    ax.set_title("Gaussian Naive Bayes - Bayesian Network", fontsize=17, fontweight="bold")
    ax.margins(0.2)
    ax.axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Graph Saved] {output_path}")
    return graph
