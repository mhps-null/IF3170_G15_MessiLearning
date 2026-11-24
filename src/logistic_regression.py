"""Binary logistic regression using NumPy and FISTA with backtracking.

FISTA accelerates proximal gradient steps. Mean log-loss and L2 form the
smooth part of the objective. Soft-thresholding handles the L1 penalty.
The momentum and backtracking updates follow Section 4 of:

Beck, A., & Teboulle, M. (2009). A Fast Iterative Shrinkage-Thresholding
Algorithm for Linear Inverse Problems. SIAM Journal on Imaging Sciences,
2(1), 183-202. https://doi.org/10.1137/080716542
"""

from math import isfinite
from numbers import Integral, Real

import numpy as np

from .base import BaseModel


class LogisticRegression(BaseModel):
    """Fit a binary model using mean log-loss and an optional L1 and/or L2 penalty.

    X is a real matrix with samples in rows, and y contains labels 0 or 1.
    The objective adds lambda_1 * ||w||_1 and lambda_2 * ||w||_2^2 / 2 to
    mean log-loss. The intercept is free. L1 and L2 assign all of
    ``reg_lambda`` to their respective terms. Elastic Net assigns the fraction
    ``l1_ratio`` to L1 and the remainder to L2.

    Each fit starts from zero with step size ``lr``. Backtracking multiplies
    the step by ``backtracking_factor`` and tries at most ``max_backtracking``
    candidates per update. The fit allows at most ``n_iters`` accepted updates.
    ``optimality_`` is the largest KKT violation at the last accepted point,
    and ``converged_`` means it reached ``tol``. With ``tol=0``, the fit uses
    the full update budget.

    The loss history starts at zero parameters. Optional trajectory entries
    contain (bias, weights.copy(), total loss) at the same points. Step sizes
    are recorded once per accepted update.
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
        self.n_iters = n_iters
        self.penalty = penalty.lower() if isinstance(penalty, str) else penalty
        self.reg_lambda = reg_lambda
        self.l1_ratio = l1_ratio
        self.record_trajectory = record_trajectory
        self.tol = tol
        self.backtracking_factor = backtracking_factor
        self.max_backtracking = max_backtracking
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

    def fit(self, X: np.ndarray, y: np.ndarray):
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

        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            # The zero initialization can already satisfy the stopping condition.
            signed_logits = np.zeros_like(label_sign)
            smooth_loss = self._smooth_loss(signed_logits, accepted[:-1], l2_strength)
            gradient = self._smooth_gradient(X, label_sign, signed_logits, accepted[:-1], l2_strength)
            self._record_accepted(accepted, smooth_loss, gradient, l1_strength)

            for iteration in range(1, int(self.n_iters) + 1):
                if self.converged_:
                    break
                candidate, signed_logits, smooth_loss, step_size = self._backtracking_step(
                    X, label_sign, extrapolated, step_size, l1_strength, l2_strength
                )

                # Extrapolate from the two successive accepted points.
                next_momentum = (1.0 + np.sqrt(1.0 + 4.0 * momentum**2)) / 2.0
                momentum_fraction = (momentum - 1.0) / next_momentum
                extrapolated = candidate + momentum_fraction * (candidate - accepted)
                accepted = candidate
                momentum = next_momentum

                gradient = self._smooth_gradient(X, label_sign, signed_logits, accepted[:-1], l2_strength)
                self._record_accepted(accepted, smooth_loss, gradient, l1_strength)
                self.n_iter_ = iteration
                self.step_sizes_.append(step_size)

        self.is_fitted = True
        return self

    def _backtracking_step(self, X, label_sign, extrapolated, step_size, l1_strength, l2_strength):
        """Accept a proximal step when the smooth loss satisfies its quadratic bound."""
        weights, bias = extrapolated[:-1], extrapolated[-1]
        signed_logits = label_sign * self._logits(X, weights, bias)
        smooth_loss = self._smooth_loss(signed_logits, weights, l2_strength)
        gradient = self._smooth_gradient(X, label_sign, signed_logits, weights, l2_strength)
        relative_roundoff = 16 * np.finfo(float).eps
        numerical_error = None

        for _ in range(self.max_backtracking):
            try:
                candidate = extrapolated - step_size * gradient
                candidate[:-1] = self._soft_threshold(candidate[:-1], step_size * l1_strength)
                candidate_signed_logits = label_sign * self._logits(X, candidate[:-1], candidate[-1])
                candidate_smooth_loss = self._smooth_loss(
                    candidate_signed_logits, candidate[:-1], l2_strength
                )

                displacement = candidate - extrapolated
                weight_change, bias_change = displacement[:-1], displacement[-1]
                squared_distance = np.dot(weight_change, weight_change) + bias_change**2
                quadratic_bound = (
                    smooth_loss
                    + np.dot(gradient[:-1], weight_change)
                    + gradient[-1] * bias_change
                    + squared_distance / (2.0 * step_size)
                )
                roundoff = relative_roundoff * max(1.0, smooth_loss, candidate_smooth_loss)
                if not np.isfinite(quadratic_bound):
                    raise FloatingPointError("The backtracking bound must be finite.")
                if candidate_smooth_loss <= quadratic_bound + roundoff:
                    return candidate, candidate_signed_logits, candidate_smooth_loss, step_size
            except FloatingPointError as error:
                # An unrepresentable trial also calls for a smaller step.
                numerical_error = error

            step_size *= self.backtracking_factor
            if step_size == 0:
                break

        raise FloatingPointError(
            "Backtracking could not find a valid update. Rescale X or lower lr."
        ) from numerical_error

    @staticmethod
    def _soft_threshold(values, threshold):
        """Shrink weights toward zero by the given threshold."""
        return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)

    @staticmethod
    def _logits(X, weights, bias):
        logits = X @ weights + bias
        if not np.isfinite(logits).all():
            raise FloatingPointError("Logits must be finite. Rescale features or parameters.")
        return logits

    @staticmethod
    def _sigmoid(logits):
        """Use exp(-abs(logits)) to keep the exponential bounded by one."""
        exponential = np.exp(-np.abs(logits))
        denominator = 1.0 + exponential
        return np.where(logits >= 0, 1.0 / denominator, exponential / denominator)

    @staticmethod
    def _smooth_loss(signed_logits, weights, l2_strength):
        """Evaluate mean softplus of (1 - 2*y) * logits plus the L2 penalty."""
        # Divide before summing to keep large finite losses representable.
        loss = np.sum(np.logaddexp(0.0, signed_logits) / len(signed_logits))
        if l2_strength:
            loss += 0.5 * l2_strength * np.dot(weights, weights)
            # dot can overflow in BLAS without raising a NumPy exception.
            if not np.isfinite(loss):
                raise FloatingPointError("Smooth loss must be finite. Rescale the data.")
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

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate the full mean-loss objective at the current fitted parameters."""
        X = self._prediction_features(X)
        y = self._validate_labels(y, X.shape[0])
        l1_strength, l2_strength = self._penalty_strengths()
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            signed_logits = (1.0 - 2.0 * y) * self._logits(X, self.weights_, self.bias_)
            smooth_loss = self._smooth_loss(signed_logits, self.weights_, l2_strength)
            return self._total_loss(smooth_loss, self.weights_, l1_strength)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (n_samples, 2) probabilities, columns ordered as classes 0, 1."""
        X = self._prediction_features(X)
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            probabilities = self._sigmoid(self._logits(X, self.weights_, self.bias_))
        return np.column_stack((1.0 - probabilities, probabilities))

    def predict(self, X: np.ndarray) -> np.ndarray:
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
        if not isinstance(self.penalty, str) or self.penalty not in ("none", "l1", "l2", "elasticnet"):
            raise ValueError("penalty must be 'none', 'l1', 'l2', or 'elasticnet'.")

        for name in ("lr", "reg_lambda", "l1_ratio", "tol", "backtracking_factor"):
            value = getattr(self, name)
            if not isinstance(value, Real) or not isfinite(value):
                raise ValueError(f"{name} must be a finite real number.")

        if self.lr <= 0 or self.reg_lambda < 0 or self.tol < 0:
            raise ValueError("Use a positive lr and nonnegative reg_lambda and tol.")
        if not 0 <= self.l1_ratio <= 1:
            raise ValueError("l1_ratio must be in [0, 1].")
        if not 0 < self.backtracking_factor < 1:
            raise ValueError("backtracking_factor must be in (0, 1).")

        for name in ("n_iters", "max_backtracking"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")

        if not isinstance(self.record_trajectory, (bool, np.bool_)):
            raise ValueError("record_trajectory must be boolean.")

    @staticmethod
    def _validate_features(X):
        features = np.asarray(X)
        if features.ndim != 2 or features.dtype.kind not in "biuf":
            raise ValueError("X must be a dense real matrix with shape (n_samples, n_features).")
        features = features.astype(np.float64, copy=False)
        if not np.isfinite(features).all():
            raise ValueError("X must contain only finite values.")
        return features

    @staticmethod
    def _validate_labels(y, n_samples):
        if n_samples == 0:
            raise ValueError("X must contain at least one sample.")
        labels = np.asarray(y)
        if labels.shape != (n_samples,):
            raise ValueError("y must have shape (n_samples,) matching X.")
        if labels.dtype.kind not in "biuf" or not ((labels == 0) | (labels == 1)).all():
            raise ValueError("y must contain only numeric binary labels 0 and 1.")
        return labels.astype(np.float64, copy=False)


# Booking features and model configuration

_LEAD_TIME_BANDS = ("0-7", "8-30", "31-90", "91+")
# Fixed categories keep every indicator column even when a split lacks a combination.
_LEAD_REQUEST_LEVELS = tuple(
    sorted(f"{band}_{count}" for band in _LEAD_TIME_BANDS for count in range(4))
)


class BookingFeatureTransformer:
    """Add 20 interaction indicators to the team's 52 shared booking features."""

    def fit(self, raw_bookings):
        from .preprocessing import DataPipeline

        self.pipeline_ = DataPipeline(use_pca=False).fit(raw_bookings)
        if len(self.pipeline_.feature_names_out_) != 52:
            raise ValueError("Booking Logistic Regression requires the team's 52-column schema.")
        return self

    def transform(self, raw_bookings):
        """Return the 72 features in the fitted schema and original row order."""
        import pandas as pd

        from .preprocessing import FEATURE_COLS

        pipeline = self.pipeline_
        shared_features = pipeline.transform(raw_bookings)

        # Build indicators in the original units using the fitted cleaning steps.
        cleaned = pipeline._impute(raw_bookings[FEATURE_COLS])
        cleaned = pipeline._cap_outliers(cleaned)
        cleaned = pipeline._adjust_types(cleaned)

        lead_bands = pd.cut(
            cleaned.days_before_arrival, [-1, 7, 30, 90, np.inf], labels=_LEAD_TIME_BANDS
        ).astype(str)
        requests = cleaned.special_requests.clip(upper=3).astype(str)
        lead_request_levels = (lead_bands + "_" + requests).astype(
            pd.CategoricalDtype(categories=_LEAD_REQUEST_LEVELS)
        )
        lead_request_flags = pd.get_dummies(lead_request_levels, dtype=float).add_prefix(
            "lead_requests_"
        )
        channel_request_flags = pd.DataFrame(
            {
                f"{channel}_requests_{count}": (
                    cleaned.booking_channel.eq(channel) & cleaned.special_requests.eq(count)
                ).astype(float)
                for channel in ("Online Travel Agent", "Offline Agent")
                for count in (1, 2)
            }
        )
        return pd.concat([shared_features, lead_request_flags, channel_request_flags], axis=1)


def make_booking_logistic(n_train_samples, *, record_trajectory=False):
    """Build the booking L1 baseline with lambda = 1 / n_train_samples (C=1).

    The integrated notebook also uses this configuration for the one-feature
    contour demonstration. It tunes the final booking model separately.
    """
    return LogisticRegression(
        penalty="l1",
        reg_lambda=1 / n_train_samples,
        lr=1.0,
        backtracking_factor=0.5,
        max_backtracking=60,
        n_iters=1500,
        tol=1e-5,
        record_trajectory=record_trajectory,
    )
