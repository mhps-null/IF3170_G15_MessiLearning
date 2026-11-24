"""Binary logistic regression with FISTA and backtracking.

The momentum and backtracking updates follow Section 4 of:

Beck, A., & Teboulle, M. (2009). A Fast Iterative Shrinkage-Thresholding
Algorithm for Linear Inverse Problems. SIAM Journal on Imaging Sciences,
2(1), 183-202. https://doi.org/10.1137/080716542
"""

from itertools import combinations
from operator import index

import numpy as np
import pandas as pd
from scipy import sparse

from .base import BaseModel


class LogisticRegression(BaseModel):
    """Fit binary labels 0 and 1 from dense or sparse features.

    Mean log-loss and L2 form the smooth objective. Soft-thresholding handles
    L1, and the intercept is unpenalized. Elastic Net gives L1 the fraction
    l1_ratio of reg_lambda and assigns the rest to L2.

    optimality_ is the largest KKT violation. Setting tol=0 uses the full
    update budget. The loss history and optional trajectory start at zero.
    """

    def __init__(
        self,
        lr: float = 1.0,
        n_iters: int = 1000,
        penalty: str = "l2",
        reg_lambda: float = 0.01,
        l1_ratio: float = 0.5,
        *,
        record_trajectory: bool = False,
        tol: float = 1e-6,
        backtracking_factor: float = 0.5,
        max_backtracking: int = 60,
    ):
        super().__init__()
        self.lr = lr
        self.n_iters = index(n_iters)
        self.penalty = penalty.lower()
        self.reg_lambda = reg_lambda
        self.l1_ratio = l1_ratio
        self.record_trajectory = record_trajectory
        self.tol = tol
        self.backtracking_factor = backtracking_factor
        self.max_backtracking = index(max_backtracking)
        self._validate_hyperparameters()
        self._reset_training_state()

    def _reset_training_state(self):
        self.is_fitted = False
        self.weights_ = None
        self.bias_ = 0.0
        self.loss_history_ = []
        self.trajectory_ = []
        self.step_sizes_ = []
        self.n_iter_ = 0
        self.converged_ = False
        self.optimality_ = np.inf

    def fit(self, X, y: np.ndarray):
        """Fit from zero parameters and return the fitted model."""
        X = self._validate_features(X)
        y = self._validate_labels(y, X.shape[0])
        self._reset_training_state()
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]

        label_sign = 1.0 - 2.0 * y
        l1_strength, l2_strength = self._penalty_strengths()

        # FISTA acts on one vector with the intercept in the final slot.
        accepted = np.zeros(self.n_features_in_ + 1)
        extrapolated = accepted
        step_size = self.lr
        momentum = 1.0

        # The zero initialization can already satisfy the stopping condition.
        signed_logits = np.zeros_like(label_sign)
        smooth_loss = self._smooth_loss(signed_logits, accepted[:-1], l2_strength)
        gradient = self._smooth_gradient(X, label_sign, signed_logits, accepted[:-1], l2_strength)
        self._record_accepted(accepted, smooth_loss, gradient, l1_strength)

        for iteration in range(1, self.n_iters + 1):
            if self.converged_:
                break
            updated, signed_logits, smooth_loss, step_size = self._backtracking_step(
                X, label_sign, extrapolated, step_size, l1_strength, l2_strength
            )

            # Extrapolate from the two successive accepted points.
            next_momentum = (1.0 + np.sqrt(1.0 + 4.0 * momentum**2)) / 2.0
            momentum_fraction = (momentum - 1.0) / next_momentum
            extrapolated = updated + momentum_fraction * (updated - accepted)
            accepted, momentum = updated, next_momentum

            gradient = self._smooth_gradient(X, label_sign, signed_logits, accepted[:-1], l2_strength)
            self._record_accepted(accepted, smooth_loss, gradient, l1_strength)
            self.n_iter_ = iteration
            self.step_sizes_.append(step_size)

        self.is_fitted = True
        return self

    def _backtracking_step(self, X, label_sign, extrapolated, step_size, l1_strength, l2_strength):
        """Accept a proximal step when the smooth loss satisfies its quadratic bound."""
        weights, bias = extrapolated[:-1], extrapolated[-1]
        signed_logits = label_sign * (X @ weights + bias)
        smooth_loss = self._smooth_loss(signed_logits, weights, l2_strength)
        gradient = self._smooth_gradient(X, label_sign, signed_logits, weights, l2_strength)
        relative_roundoff = 16 * np.finfo(float).eps

        # A nonfinite trial also calls for a smaller step.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            for _ in range(self.max_backtracking):
                trial = extrapolated - step_size * gradient
                trial[:-1] = self._soft_threshold(trial[:-1], step_size * l1_strength)
                trial_logits = label_sign * (X @ trial[:-1] + trial[-1])
                trial_loss = self._smooth_loss(trial_logits, trial[:-1], l2_strength)

                displacement = trial - extrapolated
                quadratic_bound = (
                    smooth_loss
                    + np.dot(gradient, displacement)
                    + np.dot(displacement, displacement) / (2.0 * step_size)
                )
                roundoff = relative_roundoff * max(1.0, smooth_loss, trial_loss)
                if (
                    np.isfinite(quadratic_bound)
                    and np.isfinite(trial_loss)
                    and trial_loss <= quadratic_bound + roundoff
                    and np.isfinite(trial_logits).all()
                ):
                    return trial, trial_logits, trial_loss, step_size
                step_size *= self.backtracking_factor
        raise FloatingPointError("Backtracking could not find a valid update. Rescale X or lower lr.")

    @staticmethod
    def _soft_threshold(values, threshold):
        """Shrink weights toward zero by the given threshold."""
        return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)

    @staticmethod
    def _sigmoid(logits):
        """Use exp(-abs(logits)) to keep the exponential bounded by one."""
        exponential = np.exp(-np.abs(logits))
        denominator = 1.0 + exponential
        return np.where(logits >= 0, 1.0 / denominator, exponential / denominator)

    @staticmethod
    def _smooth_loss(signed_logits, weights, l2_strength):
        """Evaluate mean softplus of (1 - 2*y) * logits plus the L2 penalty."""
        loss = np.logaddexp(0.0, signed_logits).mean()
        if l2_strength:
            loss += 0.5 * l2_strength * np.dot(weights, weights)
        return float(loss)

    @staticmethod
    def _total_loss(smooth_loss, weights, l1_strength):
        if l1_strength:
            smooth_loss += l1_strength * np.sum(np.abs(weights))
        return float(smooth_loss)

    def _smooth_gradient(self, X, label_sign, signed_logits, weights, l2_strength):
        """Return the weight and intercept gradients in FISTA's parameter order."""
        # This form of sigmoid(logits) - y retains small residuals near p = 1.
        residual = label_sign * self._sigmoid(signed_logits) / len(label_sign)
        gradient = np.empty(weights.size + 1)
        gradient[:-1] = X.T @ residual + l2_strength * weights
        gradient[-1] = np.sum(residual)
        return gradient

    @staticmethod
    def _kkt_residual(weights, gradient, l1_strength):
        """Return the largest first-order optimality violation.

        A nonzero weight's smooth gradient must cancel the L1 derivative.
        At zero the smooth gradient can lie anywhere in [-lambda_1, lambda_1].
        The intercept gradient must be zero.
        """
        weight_gradient = gradient[:-1]
        weight_residuals = np.where(
            weights == 0,
            np.maximum(np.abs(weight_gradient) - l1_strength, 0.0),
            np.abs(weight_gradient + l1_strength * np.sign(weights)),
        )
        return max(float(np.max(weight_residuals, initial=0.0)), abs(float(gradient[-1])))

    def _record_accepted(self, accepted, smooth_loss, gradient, l1_strength):
        """Measure the full objective and convergence at an accepted point."""
        weights, bias = accepted[:-1], float(accepted[-1])
        objective = self._total_loss(smooth_loss, weights, l1_strength)
        optimality = self._kkt_residual(weights, gradient, l1_strength)

        self.weights_, self.bias_ = weights, bias
        self.loss_history_.append(objective)
        if self.record_trajectory:
            self.trajectory_.append((bias, weights.copy(), objective))
        self.optimality_ = optimality
        self.converged_ = self.tol > 0 and optimality <= self.tol

    def _penalty_strengths(self):
        if self.penalty == "none":
            return 0.0, 0.0
        l1_fraction = {"l1": 1.0, "l2": 0.0, "elasticnet": self.l1_ratio}[self.penalty]
        return self.reg_lambda * l1_fraction, self.reg_lambda * (1.0 - l1_fraction)

    @property
    def coef_(self):
        """Fitted weights with shape (1, n_features)."""
        if not self.is_fitted:
            raise AttributeError("Fit LogisticRegression before reading coef_.")
        return self.weights_[None, :]

    @property
    def intercept_(self):
        """Fitted intercept with shape (1,)."""
        if not self.is_fitted:
            raise AttributeError("Fit LogisticRegression before reading intercept_.")
        return np.array([self.bias_])

    def compute_loss(self, X, y: np.ndarray) -> float:
        """Evaluate the full mean-loss objective at the current fitted parameters."""
        X = self._prediction_features(X)
        y = self._validate_labels(y, X.shape[0])
        l1_strength, l2_strength = self._penalty_strengths()
        signed_logits = (1.0 - 2.0 * y) * (X @ self.weights_ + self.bias_)
        smooth_loss = self._smooth_loss(signed_logits, self.weights_, l2_strength)
        return self._total_loss(smooth_loss, self.weights_, l1_strength)

    def predict_proba(self, X) -> np.ndarray:
        """Return (n_samples, 2) probabilities, columns ordered as classes 0, 1."""
        X = self._prediction_features(X)
        probabilities = self._sigmoid(X @ self.weights_ + self.bias_)
        return np.column_stack((1.0 - probabilities, probabilities))

    def predict(self, X) -> np.ndarray:
        """Return labels (n_samples,), class 1 at probability >= 0.5."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def _prediction_features(self, X):
        if not self.is_fitted:
            raise RuntimeError(
                "LogisticRegression must be fitted before prediction or loss evaluation."
            )
        X = self._validate_features(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}.")
        return X

    def _validate_hyperparameters(self):
        if self.penalty not in ("none", "l1", "l2", "elasticnet"):
            raise ValueError("penalty must be 'none', 'l1', 'l2', or 'elasticnet'.")
        if not np.isfinite([self.lr, self.reg_lambda, self.tol]).all():
            raise ValueError("lr, reg_lambda, and tol must be finite.")
        if self.lr <= 0 or self.reg_lambda < 0 or self.tol < 0:
            raise ValueError("Use positive lr and nonnegative reg_lambda and tol.")
        if not 0 <= self.l1_ratio <= 1:
            raise ValueError("l1_ratio must be in [0, 1].")
        if not 0 < self.backtracking_factor < 1:
            raise ValueError("backtracking_factor must be in (0, 1).")
        if self.n_iters < 1 or self.max_backtracking < 1:
            raise ValueError("n_iters and max_backtracking must be positive.")

    @staticmethod
    def _validate_features(X):
        is_sparse = sparse.issparse(X)
        features = X if is_sparse else np.asarray(X)
        if features.ndim != 2 or features.dtype.kind not in "biuf":
            raise ValueError("X must be a real matrix with shape (n_samples, n_features).")
        features = features.astype(np.float64, copy=False)
        if is_sparse:
            features = sparse.csr_matrix(features)
        if not np.isfinite(features.data if is_sparse else features).all():
            raise ValueError("X must contain only finite values.")
        return features

    @staticmethod
    def _validate_labels(y, n_samples):
        labels = np.asarray(y)
        if n_samples == 0 or labels.shape != (n_samples,):
            raise ValueError("y must contain one label per row of nonempty X.")
        if labels.dtype.kind not in "biuf" or not ((labels == 0) | (labels == 1)).all():
            raise ValueError("y must contain only numeric binary labels 0 and 1.")
        return labels.astype(np.float64, copy=False)


_QUANTILE_COLUMNS = ("days_before_arrival", "nightly_rate_idr")


class BookingFeatureTransformer:
    """Append frequent single and pair indicators to the team's shared booking features."""

    def __init__(self, n_bins=5, min_count=20):
        self.n_bins = n_bins
        self.min_count = min_count

    def fit(self, raw_bookings):
        from .preprocessing import DataPipeline, FEATURE_COLS

        self.pipeline_ = DataPipeline(use_pca=False).fit(raw_bookings)
        cleaned = self._clean_bookings(raw_bookings)
        quantiles = np.arange(1, self.n_bins) / self.n_bins
        self.bin_edges_ = {}
        # Lead time and price have many distinct values. Keep calendar and count levels.
        for column in _QUANTILE_COLUMNS:
            values = cleaned[column].to_numpy()
            cuts = np.unique(np.quantile(values, quantiles))
            self.bin_edges_[column] = cuts[cuts < values.max()]
        levels = self._levels(cleaned)

        self.feature_names_out_ = list(self.pipeline_.feature_names_out_)
        self.retained_combinations_ = {}
        groups = [(column,) for column in FEATURE_COLS] + list(combinations(FEATURE_COLS, 2))
        for columns in groups:
            is_pair = len(columns) == 2
            counts = self._group_values(levels, columns).value_counts(sort=False)
            retained = counts.index[counts >= self.min_count].sort_values()
            if is_pair:
                retained = retained.remove_unused_levels()
            self.retained_combinations_[columns] = retained
            for key in retained:
                values = key if is_pair else (key,)
                name = " & ".join(f"{column}={value}" for column, value in zip(columns, values))
                self.feature_names_out_.append(f"indicator {name}")
        return self

    def transform(self, raw_bookings):
        """Return a CSR matrix in fitted column order and the caller's row order."""
        shared = self.pipeline_.transform(raw_bookings)
        levels = self._levels(self._clean_bookings(raw_bookings))

        # Each row matches at most one retained value per group. An unseen or
        # infrequent key returns -1 and contributes no entry for that group.
        row_indices, column_indices = [], []
        offset = 0
        for columns, retained in self.retained_combinations_.items():
            matches = retained.get_indexer(self._group_values(levels, columns))
            active_rows = np.flatnonzero(matches >= 0)
            row_indices.append(active_rows)
            column_indices.append(offset + matches[active_rows])
            offset += len(retained)
        rows = np.concatenate(row_indices)
        indicators = sparse.csr_matrix(
            (np.ones(rows.size), (rows, np.concatenate(column_indices))),
            shape=(len(raw_bookings), offset),
        )
        shared_csr = sparse.csr_matrix(shared.to_numpy(dtype=float))
        return sparse.hstack((shared_csr, indicators), format="csr")

    def _clean_bookings(self, raw_bookings):
        """Use the team's fitted cleaning steps and keep values in their original units."""
        from .preprocessing import FEATURE_COLS

        pipeline = self.pipeline_
        imputed = pipeline._impute(raw_bookings[FEATURE_COLS])
        return pipeline._adjust_types(pipeline._cap_outliers(imputed))

    @staticmethod
    def _group_values(levels, columns):
        if len(columns) == 1:
            return pd.Index(levels[columns[0]])
        return pd.MultiIndex.from_frame(levels[list(columns)])

    def _levels(self, cleaned):
        from .preprocessing import FEATURE_COLS, RAW_CATEGORY_COLS

        levels = cleaned[FEATURE_COLS].copy()
        # Use the category values themselves so codes cannot vary with the batch.
        levels[RAW_CATEGORY_COLS] = levels[RAW_CATEGORY_COLS].astype(str)
        for column, cuts in self.bin_edges_.items():
            # Values on a cut go in the lower bin. End bins cover the full range.
            codes = np.searchsorted(cuts, cleaned[column].to_numpy(), side="left")
            levels[column] = pd.Index([f"bin {code}" for code in range(len(cuts) + 1)]).take(codes)
        return levels
