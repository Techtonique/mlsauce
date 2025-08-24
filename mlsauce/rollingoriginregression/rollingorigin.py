import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class RollingOriginForecaster(BaseEstimator, RegressorMixin):
    """
    A flexible rolling origin forecaster that supports both autoregressive and
    exogenous features modes, with multiple prediction strategies.
    """

    def __init__(
        self,
        estimator,
        max_horizon=1,
        n_lags=1,
        mode="auto",
        multi_output="auto",
        recursive=True,
    ):
        self.estimator = estimator
        self.max_horizon = max_horizon
        self.n_lags = n_lags
        self.mode = mode
        self.multi_output = multi_output
        self.recursive = recursive

    def fit(self, X=None, y=None):
        if y is None:
            raise ValueError("y cannot be None")

        # Determine mode
        if self.mode == "auto":
            self.mode_ = "ar" if X is None else "exog"
        else:
            self.mode_ = self.mode

        # Fit in appropriate mode
        if self.mode_ == "ar":
            return self._fit_ar(y)
        else:
            return self._fit_exog(X, y)

    def predict(self, X=None, h=None):
        check_is_fitted(self)

        # Validate horizon
        if h is None:
            h = self.max_horizon
        elif h > self.max_horizon:
            raise ValueError(
                f"Requested horizon {h} exceeds max_horizon {self.max_horizon}"
            )

        if self.mode_ == "ar":
            return self._predict_ar(h)
        else:
            if X is None:
                raise ValueError("X cannot be None in exog mode")
            X = check_array(X)
            return self._predict_exog(X, h)

    def _fit_ar(self, y):
        """Fit autoregressive model."""
        y = check_array(y, ensure_2d=False)

        # Validate series length
        if len(y) < self.n_lags + 1:
            raise ValueError(
                f"y must have at least n_lags+1 ({self.n_lags+1}) samples"
            )

        # Create lagged features matrix
        X, y = self._create_lagged_features(y)

        # Fit model
        self.estimator_ = self._fit_model(X, y)

        # Store last window for predictions
        self.last_window_ = y[-self.n_lags :].reshape(1, -1)

        return self

    def _fit_exog(self, X, y):
        """Fit model with exogenous features."""
        X, y = check_X_y(X, y, multi_output=True)
        self.n_features_in_ = X.shape[1]

        # Validate series length
        if len(y) < self.max_horizon:
            raise ValueError(
                f"Need at least max_horizon ({self.max_horizon}) samples"
            )

        # Fit model
        self.estimator_ = self._fit_model(X, y)

        return self

    def _fit_model(self, X, y):
        """Internal method to fit model with selected strategy."""
        # Determine multi-output strategy
        if self.multi_output == "auto":
            try:
                # Test if estimator supports multi-output
                dummy = clone(self.estimator)
                test_shape = (min(10, len(X)), self.max_horizon)
                dummy.fit(X[: min(10, len(X))], np.zeros(test_shape))
                self.multi_output_ = True
            except Exception:
                self.multi_output_ = False
        else:
            self.multi_output_ = self.multi_output

        # Prepare targets for multi-step forecasting
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if self.multi_output_:
            # Create shifted targets matrix
            T = len(y)
            if T < self.max_horizon:
                raise ValueError("Time series too short for max_horizon")
            y_shifted = np.zeros((T - self.max_horizon + 1, self.max_horizon))
            for i in range(self.max_horizon):
                y_shifted[:, i] = y[i : T - self.max_horizon + 1 + i].ravel()

            # Trim X to match
            X = X[: len(y_shifted)]
            y = y_shifted

            return clone(self.estimator).fit(X, y)

        else:
            if self.recursive:
                # Single model for recursive predictions
                return clone(self.estimator).fit(X, y.ravel())
            else:
                # Separate models for each horizon
                self.estimators_ = []
                for i in range(self.max_horizon):
                    X_i = X[: len(y) - i]
                    y_i = y[i:].ravel()
                    est = clone(self.estimator)
                    est.fit(X_i, y_i)
                    self.estimators_.append(est)
                return self

    def _predict_ar(self, h):
        """Make autoregressive predictions."""
        current_window = self.last_window_.copy()
        predictions = np.zeros((1, h))

        for i in range(h):
            pred = self.estimator_.predict(current_window)[0]
            predictions[0, i] = pred
            # Update window
            current_window = np.roll(current_window, -1)
            current_window[0, -1] = pred

        return predictions

    def _predict_exog(self, X, h):
        """Make predictions with exogenous features."""
        if hasattr(self, "estimators_"):  # Direct strategy
            preds = np.zeros((X.shape[0], h))
            for i in range(h):
                preds[:, i] = self.estimators_[i].predict(X).ravel()
            return preds
        else:
            if self.multi_output_:
                pred_out = self.estimator_.predict(X)
                if pred_out.shape[1] < h:
                    # Pad if model outputs fewer horizons than requested
                    pad_width = ((0, 0), (0, h - pred_out.shape[1]))
                    pred_out = np.pad(pred_out, pad_width, mode="constant")
                return pred_out[:, :h]
            else:  # Recursive
                preds = np.zeros((X.shape[0], h))
                current_pred = self.estimator_.predict(X)
                if current_pred.ndim == 1:
                    current_pred = current_pred.reshape(-1, 1)
                preds[:, 0] = current_pred.ravel()

                for i in range(1, h):
                    if X.shape[1] > 1:
                        X_new = np.hstack([X[:, 1:], current_pred])
                    else:
                        X_new = current_pred
                    current_pred = self.estimator_.predict(X_new)
                    if current_pred.ndim == 1:
                        current_pred = current_pred.reshape(-1, 1)
                    preds[:, i] = current_pred.ravel()

                return preds

    def _create_lagged_features(self, y):
        """Create lagged features matrix for AR mode."""
        if len(y) <= self.n_lags:
            raise ValueError("Not enough samples to create lagged features")
        n_samples = len(y) - self.n_lags
        X = np.zeros((n_samples, self.n_lags))
        for i in range(self.n_lags):
            X[:, i] = y[i : i + n_samples]
        y = y[self.n_lags :]
        return X, y

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        params = {
            "estimator": self.estimator,
            "max_horizon": self.max_horizon,
            "n_lags": self.n_lags,
            "mode": self.mode,
            "multi_output": self.multi_output,
            "recursive": self.recursive,
        }
        if deep:
            for key, value in params.items():
                if hasattr(value, "get_params"):
                    params[key] = value.get_params(deep)
        return params

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self
