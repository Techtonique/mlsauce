import numpy as np
import pandas as pd
from sklearn.decomposition import (
    PCA,
    TruncatedSVD,
    FactorAnalysis,
    FastICA,
    NMF,
    KernelPCA,
    MiniBatchSparsePCA,
)
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from statsmodels.tsa.ar_model import AutoReg
from sklearn.utils.validation import check_is_fitted, check_array
from typing import Optional, Literal, Union, Any
import matplotlib.pyplot as plt
import warnings


class GenericFunctionalForecaster(BaseEstimator, RegressorMixin):
    """
    Functional time series forecaster using dimensionality reduction and regression.

    Following Hyndman-Ullah methodology:
    1. Extract functional components using dimensionality reduction
    2. Model relationships between components and functional data using regression
    3. Forecast future functional curves

    Parameters
    ----------
    n_components : int, default=8
        Number of components to extract.
    reduction_method : str, default='pca'
        Dimensionality reduction method.
    reduction_params : dict, optional
        Additional parameters for the reduction method.
    rolling_window : int, optional
        Window size for rolling regression. If None, uses full training set.
    forecast_method : {'ar', 'last_value'}, default='ar'
        Method for forecasting coefficients.
    regressor : sklearn regressor, optional
        Any sklearn regressor. If None, uses LinearRegression.
    regressor_params : dict, optional
        Additional parameters for the regressor.
    """

    def __init__(
        self,
        n_components: int = 8,
        reduction_method: str = "pca",
        reduction_params: Optional[dict] = None,
        rolling_window: Optional[int] = None,
        forecast_method: Literal["ar", "last_value"] = "ar",
        regressor: Optional[BaseEstimator] = None,
        regressor_params: Optional[dict] = None,
    ):
        self.n_components = n_components
        self.reduction_method = reduction_method
        self.reduction_params = reduction_params or {}
        self.rolling_window = rolling_window
        self.forecast_method = forecast_method
        self.regressor = (
            regressor if regressor is not None else LinearRegression()
        )
        self.regressor_params = regressor_params or {}

        # Available reduction methods
        self._reduction_methods = {
            "pca": PCA,
            "kernel_pca": KernelPCA,
            "truncated_svd": TruncatedSVD,
            "factor_analysis": FactorAnalysis,
            "fast_ica": FastICA,
            "nmf": NMF,
            "minibatch_sparse_pca": MiniBatchSparsePCA,
            "mds": MDS,
            "isomap": Isomap,
            "lle": LocallyLinearEmbedding,
        }

        if reduction_method not in self._reduction_methods:
            raise ValueError(
                f"reduction_method must be one of {list(self._reduction_methods.keys())}"
            )

    def _create_regressor(self):
        """Create a fresh regressor instance with parameters."""
        if hasattr(self.regressor, "__class__"):
            # Create new instance from class
            regressor = self.regressor.__class__(**self.regressor_params)
        else:
            # Clone existing instance
            from sklearn.base import clone

            regressor = clone(self.regressor)
            # Apply additional parameters
            for param, value in self.regressor_params.items():
                setattr(regressor, param, value)

        return regressor

    def fit(
        self, X: Union[np.ndarray, pd.DataFrame]
    ) -> "GenericFunctionalForecaster":
        """
        Fit the functional forecaster.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame, shape (n_samples, n_points)
            Functional time series data.

        Returns
        -------
        self : object
            Fitted forecaster.
        """
        # Input validation and conversion
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = check_array(X)

        self.X_ = X.copy()
        self.n_samples_, self.n_points_ = X.shape

        # 1. Standardize the functional data
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # 2. Fit dimensionality reduction
        self._fit_reduction_method(X_scaled)

        # 3. Extract components (reduced features)
        self.reduced_features_ = self.reducer_.transform(X_scaled)

        # 4. Fit regression models
        if self.rolling_window is not None:
            self._fit_rolling_regression(X_scaled)
        else:
            self._fit_full_regression(X_scaled)

        self.is_fitted_ = True
        return self

    def _fit_reduction_method(self, X_scaled):
        """Fit the dimensionality reduction method."""
        reduction_class = self._reduction_methods[self.reduction_method]

        # Handle method-specific parameters
        if self.reduction_method == "kernel_pca":
            if "kernel" not in self.reduction_params:
                self.reduction_params["kernel"] = "rbf"
            if "fit_inverse_transform" not in self.reduction_params:
                self.reduction_params["fit_inverse_transform"] = True
        elif self.reduction_method == "minibatch_sparse_pca":
            if "alpha" not in self.reduction_params:
                self.reduction_params["alpha"] = 1.0
            if "batch_size" not in self.reduction_params:
                self.reduction_params["batch_size"] = min(3, self.n_samples_)

        # Initialize and fit the reducer
        self.reducer_ = reduction_class(
            n_components=self.n_components, **self.reduction_params
        )
        self.reducer_.fit(X_scaled)

        # Store components/basis functions for reconstruction
        if hasattr(self.reducer_, "components_"):
            self.components_ = (
                self.reducer_.components_
            )  # Shape: (n_components, n_points)
        elif hasattr(self.reducer_, "inverse_transform"):
            # For methods like KernelPCA, create identity mapping to get components
            try:
                identity_matrix = np.eye(self.n_components)
                reconstructed = self.reducer_.inverse_transform(identity_matrix)
                if reconstructed.shape == (self.n_components, self.n_points_):
                    self.components_ = reconstructed
                else:
                    self.components_ = reconstructed.T
            except Exception as e:
                warnings.warn(
                    f"Could not extract components for {self.reduction_method}: {e}"
                )
                self.components_ = None
        else:
            warnings.warn(
                f"No reconstruction available for {self.reduction_method}"
            )
            self.components_ = None

    def _fit_rolling_regression(self, X_scaled):
        """
        Fit rolling regression models.

        For each window, fit: reduced_features[window] -> next_scaled_curve
        This maintains scale consistency throughout.
        """
        if self.n_samples_ <= self.rolling_window:
            raise ValueError(
                f"Need more than {self.rolling_window} samples for rolling window, "
                f"got {self.n_samples_}"
            )

        self.rolling_models_ = []
        self.rolling_coefs_ = []

        n_windows = self.n_samples_ - self.rolling_window

        for i in range(n_windows):
            # Input: window of reduced features
            X_window = self.reduced_features_[
                i : i + self.rolling_window
            ]  # (window, n_components)

            # Target: next scaled functional curve
            y_next_scaled = X_scaled[i : i + self.rolling_window]  # (n_points,)

            # Create and fit regressor
            regressor = self._create_regressor()

            try:
                # Fit regression: reduced_features_window -> scaled_functional_curve
                regressor.fit(X_window, y_next_scaled)

                # Store model and coefficients
                self.rolling_models_.append(regressor)

                # Extract coefficients - shape depends on regressor type
                if hasattr(regressor, "coef_"):
                    coef = regressor.coef_
                    # For multioutput: coef shape is (n_outputs, n_features) = (n_points, n_components)
                    # For single output with multiple features: (n_features,) = (n_components,)
                    # We expect multioutput here since y_next_scaled is (n_points,)
                    if coef.ndim == 1:
                        # This shouldn't happen with multioutput, but handle gracefully
                        warnings.warn(
                            f"Unexpected single output coefficients at window {i}"
                        )
                        coef = coef.reshape(1, -1)  # (1, n_components)
                    self.rolling_coefs_.append(coef)  # (n_points, n_components)
                else:
                    # Fallback: use least squares
                    warnings.warn(
                        f"Regressor has no coef_ attribute, using least squares at window {i}"
                    )
                    coef = np.linalg.lstsq(X_window, y_next_scaled, rcond=None)[
                        0
                    ].T
                    if coef.ndim == 1:
                        coef = coef.reshape(1, -1)
                    self.rolling_coefs_.append(coef)

            except Exception as e:
                warnings.warn(
                    f"Regression failed at window {i}: {e}. Using least squares fallback."
                )
                # Least squares fallback
                coef = np.linalg.lstsq(X_window, y_next_scaled, rcond=None)[
                    0
                ].T  # (n_points, n_components)
                if coef.ndim == 1:
                    coef = coef.reshape(1, -1)
                self.rolling_coefs_.append(coef)
                self.rolling_models_.append(None)

        # Convert to array for easier manipulation
        # Shape: (n_windows, n_points, n_components)
        self.rolling_coefs_ = np.array(self.rolling_coefs_)

    def _fit_full_regression(self, X_scaled):
        """
        Fit regression using full training set.

        Fit: reduced_features -> scaled_functional_data
        """
        # Create regressor
        regressor = self._create_regressor()

        try:
            # Fit: all reduced features -> all scaled functional curves
            regressor.fit(self.reduced_features_, X_scaled)
            self.full_model_ = regressor

            # Store coefficients
            if hasattr(regressor, "coef_"):
                self.coefs_ = (
                    regressor.coef_
                )  # (n_points, n_components) for multioutput
            else:
                # Fallback to least squares
                warnings.warn(
                    "Regressor has no coef_ attribute, using least squares"
                )
                self.coefs_ = np.linalg.lstsq(
                    self.reduced_features_, X_scaled, rcond=None
                )[0].T

        except Exception as e:
            warnings.warn(
                f"Full regression failed: {e}. Using least squares fallback."
            )
            # Least squares fallback
            self.coefs_ = np.linalg.lstsq(
                self.reduced_features_, X_scaled, rcond=None
            )[0].T
            self.full_model_ = None

    def forecast(self, steps: int = 5) -> np.ndarray:
        """
        Forecast functional time series.

        Parameters
        ----------
        steps : int
            Number of steps to forecast.

        Returns
        -------
        np.ndarray, shape (steps, n_points)
            Forecasted functional curves.
        """
        check_is_fitted(self, "is_fitted_")

        if self.rolling_window is not None:
            return self._forecast_rolling(steps)
        else:
            return self._forecast_full(steps)

    def _forecast_rolling(self, steps: int) -> np.ndarray:
        """Forecast using rolling regression approach."""
        # rolling_coefs_ shape: (n_windows, n_points, n_components)
        n_windows, n_points, n_components = self.rolling_coefs_.shape
        # Forecast coefficients for each point and component
        forecasted_coefs = np.zeros((steps, n_points, n_components))

        for point_idx in range(n_points):
            for comp_idx in range(n_components):
                # Get time series of coefficients for this (point, component)
                coef_series = self.rolling_coefs_[:, point_idx, comp_idx]
                # Forecast this coefficient series
                if self.forecast_method == "ar" and len(coef_series) > 1:
                    try:
                        # Fit AR model to coefficient series
                        ar_model = AutoReg(
                            coef_series, lags=min(2, len(coef_series) - 1)
                        ).fit()
                        forecasted_values = ar_model.predict(
                            start=len(coef_series),
                            end=len(coef_series) + steps - 1,
                        )
                        forecasted_coefs[:, point_idx, comp_idx] = (
                            forecasted_values
                        )
                    except Exception as e:
                        warnings.warn(
                            f"AR forecasting failed for point {point_idx}, component {comp_idx}: {e}"
                        )
                        # Use last value
                        forecasted_coefs[:, point_idx, comp_idx] = coef_series[
                            -1
                        ]
                else:
                    # Use last value
                    forecasted_coefs[:, point_idx, comp_idx] = coef_series[-1]

        # Reconstruct functional forecasts from predicted coefficients
        forecasts_scaled = np.zeros((steps, n_points))

        if self.components_ is not None:
            # Use learned components for reconstruction
            # For each forecast step and each point, sum over components
            for step in range(steps):
                for point_idx in range(n_points):
                    # forecasted_coefs[step, point_idx, :] has shape (n_components,)
                    # self.components_[:, point_idx] has shape (n_components,)
                    forecasts_scaled[step, point_idx] = np.dot(
                        forecasted_coefs[step, point_idx, :],
                        self.components_[:, point_idx],
                    )
        else:
            # No reconstruction available - use direct prediction
            warnings.warn(
                f"No reconstruction available for {self.reduction_method}. Using last known values."
            )
            last_scaled = self.scaler_.transform(self.X_[-1:])
            forecasts_scaled = np.tile(last_scaled, (steps, 1))
        # Transform back to original scale
        return self.scaler_.inverse_transform(forecasts_scaled)

    def _forecast_full(self, steps: int) -> np.ndarray:
        """Forecast using full training set approach."""
        # First, forecast the reduced features themselves
        forecasted_features = np.zeros((steps, self.n_components))

        for comp in range(self.n_components):
            # Get time series of this component
            feature_series = self.reduced_features_[:, comp]

            if self.forecast_method == "ar" and len(feature_series) > 1:
                try:
                    # Fit AR model to feature series
                    ar_model = AutoReg(
                        feature_series, lags=min(2, len(feature_series) - 1)
                    ).fit()
                    forecasted_values = ar_model.predict(
                        start=len(feature_series),
                        end=len(feature_series) + steps - 1,
                    )
                    forecasted_features[:, comp] = forecasted_values
                except Exception as e:
                    warnings.warn(
                        f"AR forecasting failed for component {comp}: {e}"
                    )
                    # Use last value
                    forecasted_features[:, comp] = feature_series[-1]
            else:
                # Use last value
                forecasted_features[:, comp] = feature_series[-1]

        # Reconstruct functional data from forecasted features
        if hasattr(self, "full_model_") and self.full_model_ is not None:
            # Use the fitted model to predict
            try:
                forecasts_scaled = self.full_model_.predict(forecasted_features)
            except:
                # Fallback to coefficient multiplication
                forecasts_scaled = forecasted_features @ self.coefs_.T
        else:
            # Use stored coefficients
            forecasts_scaled = forecasted_features @ self.coefs_.T

        # Transform back to original scale
        forecasts = self.scaler_.inverse_transform(forecasts_scaled)
        return forecasts

    def plot_components(self, n_plot: int = 3) -> None:
        """Plot functional components."""
        check_is_fitted(self, "is_fitted_")

        if self.components_ is None:
            print(f"Components not available for {self.reduction_method}")
            return

        plt.figure(figsize=(12, 6))
        for i in range(min(n_plot, self.n_components)):
            plt.plot(self.components_[i], label=f"Component {i+1}", linewidth=2)

        plt.title(f"{self.reduction_method.upper()} Components")
        plt.xlabel("Domain Point")
        plt.ylabel("Component Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_reduced_features(self, n_plot: int = 4) -> None:
        """Plot reduced features over time."""
        check_is_fitted(self, "is_fitted_")

        plt.figure(figsize=(12, 8))
        n_subplot_cols = 2
        n_subplot_rows = (min(n_plot, self.n_components) + 1) // 2

        for i in range(min(n_plot, self.n_components)):
            plt.subplot(n_subplot_rows, n_subplot_cols, i + 1)
            plt.plot(
                self.reduced_features_[:, i], "o-", linewidth=2, markersize=4
            )
            plt.title(f"Reduced Feature {i+1}")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_forecast(
        self, actual: Optional[np.ndarray] = None, steps: int = 5
    ) -> None:
        """Plot forecasted curves."""
        forecasts = self.forecast(steps=steps)

        plt.figure(figsize=(12, 6))

        # Plot some historical curves
        n_history = min(5, len(self.X_))
        for i in range(n_history):
            idx = -(n_history - i)
            plt.plot(
                self.X_[idx],
                "b-",
                alpha=0.3,
                linewidth=1,
                label="Historical" if i == 0 else "",
            )

        # Plot actual test data if provided
        if actual is not None:
            for i in range(min(3, len(actual))):
                plt.plot(
                    actual[i],
                    "k-",
                    alpha=0.7,
                    linewidth=2,
                    label="Actual" if i == 0 else "",
                )

        # Plot forecasts
        for i in range(steps):
            plt.plot(
                forecasts[i],
                "r--",
                linewidth=2,
                alpha=0.7,
                label="Forecast" if i == 0 else "",
            )

        plt.title("Functional Time Series Forecast")
        plt.xlabel("Domain Point")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def get_model_info(self) -> dict:
        """Get information about the fitted model."""
        info = {
            "n_components": self.n_components,
            "reduction_method": self.reduction_method,
            "rolling_window": self.rolling_window,
            "forecast_method": self.forecast_method,
            "regressor": self.regressor.__class__.__name__,
            "regressor_params": self.regressor_params,
            "is_fitted": getattr(self, "is_fitted_", False),
        }

        if hasattr(self, "reduced_features_"):
            info.update(
                {
                    "n_samples": self.n_samples_,
                    "n_points": self.n_points_,
                    "explained_variance_ratio": getattr(
                        self.reducer_, "explained_variance_ratio_", None
                    ),
                    "has_components": self.components_ is not None,
                    "coefficient_shape": (
                        getattr(self, "rolling_coefs_", np.array([])).shape
                        if hasattr(self, "rolling_coefs_")
                        else getattr(self, "coefs_", np.array([])).shape
                    ),
                }
            )

        return info
