import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm
from typing import Optional, Union, Tuple, Dict, List
import warnings


class ContextAwareThetaForecaster:
    """
    Context-Aware Theta Method (Theta3/Theta4) Forecaster

    Extends the classical Theta method of Assimakopoulos and Nikolopoulos (2000)
    with temporal attention-based context modulation.

    The standard Theta method is equivalent to Simple Exponential Smoothing (SES)
    with drift. This implementation adds a context-aware drift modulation term:

        ŷ(h) = l_n + 0.5 × b₀ × (1 + γ × z*_h) × D_n(h)

    where:
        - l_n: final SES level
        - b₀: baseline drift (OLS slope / 2)
        - γ: context sensitivity parameter
        - z*_h: standardized temporal attention context
        - D_n(h): drift multiplier accounting for smoothing

    When γ=0, reduces to standard Theta method. When use_context=False,
    forces γ=0 (standard Theta). Otherwise, γ is estimated from data.

    Parameters
    ----------
    h : int, optional
        Forecast horizon. If None, defaults to 2×frequency for seasonal data,
        or 10 for non-seasonal data.
    level : list of float, optional
        Confidence levels for prediction intervals (default: [80, 95])
    seasonal_test : bool, optional
        Whether to test for seasonality (default: True)
    seasonal_method : str, optional
        Method for seasonal decomposition: 'multiplicative' or 'additive'
        (default: 'multiplicative')
    tau : int, optional
        Temporal attention decay parameter (default: 12)
    gamma_method : str, optional
        Method to estimate gamma: 'cox' (Cox partial likelihood),
        'regression' (Ridge regression) (default: 'cox')
    risk_set_size : int, optional
        Size of risk set for Cox partial likelihood (default: 15)
    stability_factor : float, optional
        Safety factor for stability constraint, 0 < factor ≤ 1 (default: 0.8)
    use_context : bool, optional
        Whether to use context-aware version (estimate γ). If False, uses
        standard Theta method with γ=0 (default: True)

    Attributes
    ----------
    alpha_ : float
        Estimated SES smoothing parameter
    b0_ : float
        Baseline drift parameter
    gamma_ : float
        Context sensitivity parameter (0 if use_context=False)
    l_n_ : float
        Final SES level
    sigma2_ : float
        Innovation variance
    is_seasonal_ : bool
        Whether seasonality was detected
    seasonal_period_ : int
        Detected seasonal period
    method_ : str
        'Context-Aware Theta' or 'Standard Theta'
    mu_z_ : float
        Mean of context variable (for standardization)
    sigma_z_ : float
        Std dev of context variable (for standardization)

    References
    ----------
    Assimakopoulos, V. and Nikolopoulos, K. (2000). The theta model: a
    decomposition approach to forecasting. International Journal of
    Forecasting, 16, 521-530.

    Hyndman, R.J., and Billah, B. (2003). Unmasking the Theta method.
    International Journal of Forecasting, 19, 287-290.
    """

    def __init__(
        self,
        h: Optional[int] = None,
        level: List[float] = [80, 95],
        seasonal_test: bool = True,
        seasonal_method: str = "multiplicative",
        tau: int = 12,
        gamma_method: str = "cox",
        risk_set_size: int = 15,
        stability_factor: float = 0.8,
        use_context: bool = True,
    ):
        self.h = h
        self.level = sorted(level)
        self.seasonal_test = seasonal_test
        self.seasonal_method = seasonal_method
        self.tau = tau
        self.gamma_method = gamma_method
        self.risk_set_size = risk_set_size
        self.stability_factor = stability_factor
        self.use_context = use_context

        # To be set during fitting
        self.alpha_ = None
        self.b0_ = None
        self.gamma_ = None
        self.l_n_ = None
        self.sigma2_ = None
        self.is_seasonal_ = False
        self.seasonal_period_ = None
        self.method_ = None
        self.mu_z_ = None
        self.sigma_z_ = None

        # Internal storage
        self._y_original = None
        self._y_adjusted = None
        self._ses_level_array = None  # Changed name to avoid conflict
        self._seasonal_component = None
        self._z_star = None
        self._trend = None
        self._n = None
        self._frequency = None
        self._forecast_deseason = None
        self._lower_deseason = None
        self._upper_deseason = None

    def _test_seasonality(self, y: np.ndarray, m: int) -> bool:
        """
        Test for seasonality using ACF-based test from A&N (2000).

        Parameters
        ----------
        y : array-like
            Time series data
        m : int
            Seasonal period

        Returns
        -------
        bool
            True if seasonal, False otherwise
        """
        n = len(y)

        if m <= 1 or n <= 2 * m:
            return False

        # Check if series is constant
        if np.std(y) < 1e-10:
            return False

        # Compute ACF
        y_mean = np.mean(y)
        c0 = np.sum((y - y_mean) ** 2) / n

        acf_values = []
        for lag in range(1, m + 1):
            ck = np.sum((y[lag:] - y_mean) * (y[:-lag] - y_mean)) / n
            acf_values.append(ck / c0)

        acf_values = np.array(acf_values)

        # Test statistic
        r_non_seasonal = acf_values[:-1]  # All except seasonal lag
        stat = np.sqrt((1 + 2 * np.sum(r_non_seasonal**2)) / n)

        # Test if seasonal ACF is significant
        seasonal = (np.abs(acf_values[-1]) / stat) > norm.ppf(0.95)

        return seasonal

    def _classical_decomposition(
        self, y: np.ndarray, period: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Classical seasonal decomposition (multiplicative or additive).

        Parameters
        ----------
        y : array-like
            Time series data
        period : int
            Seasonal period

        Returns
        -------
        adjusted : array
            Seasonally adjusted series
        seasonal : array
            Seasonal component
        trend : array
            Trend component
        """
        n = len(y)

        # Compute trend using centered moving average
        trend = np.full(n, np.nan)
        half = period // 2

        for i in range(half, n - half):
            if period % 2 == 0:
                # Even period: weighted average
                trend[i] = (
                    0.5 * y[i - half]
                    + np.sum(y[i - half + 1 : i + half])
                    + 0.5 * y[i + half]
                ) / period
            else:
                trend[i] = np.mean(y[i - half : i + half + 1])

        # Fill edges with extrapolation
        valid_idx = np.where(~np.isnan(trend))[0]
        if len(valid_idx) > 0:
            trend[: valid_idx[0]] = trend[valid_idx[0]]
            trend[valid_idx[-1] + 1 :] = trend[valid_idx[-1]]

        # Compute seasonal indices
        if self.seasonal_method == "multiplicative":
            detrended = y / (trend + 1e-10)  # Avoid division by zero
        else:  # additive
            detrended = y - trend

        seasonal_indices = np.zeros(period)
        for i in range(period):
            seasonal_indices[i] = np.nanmean(detrended[i::period])

        # Normalize seasonal component
        if self.seasonal_method == "multiplicative":
            seasonal_indices = seasonal_indices / np.mean(seasonal_indices)
        else:  # additive
            seasonal_indices = seasonal_indices - np.mean(seasonal_indices)

        # Extend to full length
        seasonal = np.tile(seasonal_indices, n // period + 1)[:n]

        # Compute seasonally adjusted series
        if self.seasonal_method == "multiplicative":
            adjusted = y / (seasonal + 1e-10)
        else:  # additive
            adjusted = y - seasonal

        return adjusted, seasonal, trend

    def _compute_ses_level(self, y: np.ndarray, alpha: float) -> np.ndarray:
        """
        Compute SES level at each time point.

        Parameters
        ----------
        y : array-like
            Time series data
        alpha : float
            Smoothing parameter

        Returns
        -------
        level : array
            SES level at each time
        """
        n = len(y)
        level = np.zeros(n)
        level[0] = y[0]

        for t in range(1, n):
            level[t] = alpha * y[t] + (1 - alpha) * level[t - 1]

        return level

    def _ses_nll(self, alpha: float, y: np.ndarray) -> float:
        """
        Negative log-likelihood for SES.

        Parameters
        ----------
        alpha : float
            Smoothing parameter
        y : array-like
            Time series data

        Returns
        -------
        nll : float
            Negative log-likelihood
        """
        level = self._compute_ses_level(y, alpha)
        residuals = y[1:] - level[:-1]
        sigma2 = np.var(residuals, ddof=1)

        if sigma2 < 1e-10:
            return 1e10

        nll = 0.5 * len(residuals) * (np.log(2 * np.pi * sigma2) + 1)
        return nll

    def _estimate_alpha(self, y: np.ndarray) -> float:
        """
        Estimate SES smoothing parameter via maximum likelihood.

        Parameters
        ----------
        y : array-like
            Time series data

        Returns
        -------
        alpha : float
            Estimated smoothing parameter
        """
        result = minimize(
            lambda a: self._ses_nll(a[0], y),
            [0.3],
            bounds=[(1e-6, 0.999)],
            method="L-BFGS-B",
        )
        return result.x[0]

    def _estimate_drift(self, y: np.ndarray) -> float:
        """
        Estimate baseline drift parameter (OLS slope / 2).

        Parameters
        ----------
        y : array-like
            Time series data

        Returns
        -------
        b0 : float
            Baseline drift
        """
        n = len(y)
        t = np.arange(n)
        t_bar = np.mean(t)
        y_bar = np.mean(y)

        beta = np.sum((t - t_bar) * (y - y_bar)) / np.sum((t - t_bar) ** 2)
        return beta / 2.0

    def _compute_attention_context(self, y: np.ndarray, tau: int) -> np.ndarray:
        """
        Compute temporal attention context at each time point.

        Parameters
        ----------
        y : array-like
            Time series data
        tau : int
            Decay parameter

        Returns
        -------
        z : array
            Context at each time
        """
        n = len(y)
        z = np.zeros(n)

        for t in range(n):
            past = np.arange(t + 1)
            weights = np.exp(-(t - past) / tau)
            weights = weights / np.sum(weights)
            z[t] = np.dot(weights, y[: t + 1])

        return z

    def _stable_partial_nll(
        self, gamma: float, z_star: np.ndarray, k: int
    ) -> float:
        """
        Negative Cox partial log-likelihood (numerically stable).

        Parameters
        ----------
        gamma : float
            Context sensitivity parameter
        z_star : array
            Standardized context
        k : int
            Risk set size

        Returns
        -------
        nll : float
            Negative partial log-likelihood
        """
        n = len(z_star)
        nll = 0.0

        for t in range(k, n):
            idx = slice(max(0, t - k), t + 1)
            z_risk = z_star[idx]

            # Use logsumexp for numerical stability
            log_den = logsumexp(gamma * z_risk)
            nll -= gamma * z_star[t] - log_den

        return nll

    def _estimate_gamma_cox(self, z_star: np.ndarray, k: int) -> float:
        """
        Estimate gamma via Cox partial likelihood.

        Parameters
        ----------
        z_star : array
            Standardized context
        k : int
            Risk set size

        Returns
        -------
        gamma : float
            Estimated context sensitivity
        """
        result = minimize(
            lambda g: self._stable_partial_nll(g[0], z_star, k),
            [0.0],
            method="BFGS",
        )
        return result.x[0]

    def _estimate_gamma_regression(
        self,
        y: np.ndarray,
        z_star: np.ndarray,
        ses_level: np.ndarray,
        alpha: float,
        b0: float,
        n: int,
    ) -> float:
        """
        Estimate gamma via Ridge regression.

        Parameters
        ----------
        y : array
            Time series data
        z_star : array
            Standardized context
        ses_level : array
            SES level at each time
        alpha : float
            SES smoothing parameter
        b0 : float
            Baseline drift
        n : int
            Number of observations

        Returns
        -------
        gamma : float
            Estimated context sensitivity
        """
        try:
            from sklearn.linear_model import Ridge
        except ImportError:
            warnings.warn(
                "scikit-learn not available. Using Cox method instead."
            )
            return self._estimate_gamma_cox(z_star, self.risk_set_size)

        residuals = y - ses_level

        # Build design matrix
        X_design = []
        y_resid = []

        for t in range(20, n):
            h = 1
            D_h = self._D_n(h, alpha, t)
            x_t = 0.5 * b0 * z_star[t - 1] * D_h
            X_design.append(x_t)
            y_resid.append(residuals[t])

        X_design = np.array(X_design).reshape(-1, 1)
        y_resid = np.array(y_resid)

        # Ridge regression
        ridge = Ridge(alpha=10.0, fit_intercept=False)
        ridge.fit(X_design, y_resid)

        return ridge.coef_[0]

    def _D_n(self, h: int, alpha: float, n: int) -> float:
        """
        Drift multiplier function.

        Parameters
        ----------
        h : int
            Forecast horizon
        alpha : float
            SES smoothing parameter
        n : int
            Number of observations

        Returns
        -------
        D : float
            Drift multiplier
        """
        return (h - 1) + (1 - (1 - alpha) ** n) / alpha

    def _recursive_forecast(
        self,
        y_adj: np.ndarray,
        l_n: float,
        alpha: float,
        b0: float,
        gamma: float,
        mu_z: float,
        sigma_z: float,
        H: int,
        tau: int,
        n: int,
    ) -> np.ndarray:
        """
        Generate recursive forecasts with recomputed attention.

        Parameters
        ----------
        y_adj : array
            Seasonally adjusted data
        l_n : float
            Final SES level
        alpha : float
            SES smoothing parameter
        b0 : float
            Baseline drift
        gamma : float
            Context sensitivity
        mu_z : float
            Mean of context (for standardization)
        sigma_z : float
            Std dev of context
        H : int
            Forecast horizon
        tau : int
            Attention decay parameter
        n : int
            Number of training observations

        Returns
        -------
        forecasts : array
            Forecasts for h=1, ..., H
        """
        forecasts = []
        history = list(y_adj)

        for h in range(1, H + 1):
            # Recompute attention on current history
            t_now = len(history) - 1
            past = np.arange(t_now + 1)
            weights = np.exp(-(t_now - past) / tau)
            weights = weights / np.sum(weights)
            z_h = np.dot(weights, history)
            z_h_star = (z_h - mu_z) / sigma_z

            # Compute forecast
            drift_mult = self._D_n(h, alpha, n)
            context_factor = 1.0 + gamma * z_h_star
            forecast = l_n + 0.5 * b0 * context_factor * drift_mult

            forecasts.append(forecast)
            history.append(forecast)

        return np.array(forecasts)

    def _prediction_intervals(
        self,
        forecasts: np.ndarray,
        alpha: float,
        sigma2: float,
        b0: float,
        gamma: float,
        sigma_z: float,
        n: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute prediction intervals.

        Parameters
        ----------
        forecasts : array
            Point forecasts
        alpha : float
            SES smoothing parameter
        sigma2 : float
            Innovation variance
        b0 : float
            Baseline drift
        gamma : float
            Context sensitivity
        sigma_z : float
            Std dev of context
        n : int
            Number of training observations

        Returns
        -------
        lower : array
            Lower bounds for each confidence level (H x nlevels)
        upper : array
            Upper bounds for each confidence level (H x nlevels)
        """
        H = len(forecasts)
        nlevels = len(self.level)

        lower = np.zeros((H, nlevels))
        upper = np.zeros((H, nlevels))

        for i, lev in enumerate(self.level):
            z_quant = norm.ppf(1 - (1 - lev / 100) / 2)

            for h in range(1, H + 1):
                # SES variance component
                var_ses = sigma2 * ((h - 1) * alpha**2 + 1)

                # Context variance component
                D_h = self._D_n(h, alpha, n)
                var_context = ((0.5 * gamma * b0 * sigma_z * D_h) ** 2) / n

                # Total variance
                var_total = var_ses + var_context
                se = np.sqrt(var_total)

                lower[h - 1, i] = forecasts[h - 1] - z_quant * se
                upper[h - 1, i] = forecasts[h - 1] + z_quant * se

        return lower, upper

    def fit(
        self,
        y: Union[np.ndarray, pd.Series, list],
        frequency: Optional[int] = None,
    ) -> "ContextAwareThetaForecaster":
        """
        Fit the Context-Aware Theta model.

        Parameters
        ----------
        y : array-like
            Time series data
        frequency : int, optional
            Seasonal period. If None, attempts to infer from data.

        Returns
        -------
        self : ContextAwareThetaForecaster
            Fitted model
        """
        # Convert to numpy array
        if isinstance(y, pd.Series):
            y = y.values
        elif isinstance(y, list):
            y = np.array(y)

        self._y_original = y.copy()
        self._n = len(y)

        # Infer frequency if not provided
        if frequency is None:
            # Simple heuristic: look for common frequencies
            if self._n >= 24:
                self._frequency = 12  # Monthly
            elif self._n >= 8:
                self._frequency = 4  # Quarterly
            else:
                self._frequency = 1  # Non-seasonal
        else:
            self._frequency = frequency

        self.seasonal_period_ = self._frequency

        # Set default forecast horizon if not provided
        if self.h is None:
            if self._frequency > 1:
                self.h = 2 * self._frequency
            else:
                self.h = 10

        # Test for seasonality
        if self.seasonal_test and self._frequency > 1:
            self.is_seasonal_ = self._test_seasonality(y, self._frequency)
        else:
            self.is_seasonal_ = False

        # Seasonal decomposition if needed
        if self.is_seasonal_:
            self._y_adjusted, self._seasonal_component, self._trend = (
                self._classical_decomposition(y, self._frequency)
            )

            # Check for near-zero seasonal indices
            if self.seasonal_method == "multiplicative":
                if np.any(np.abs(self._seasonal_component) < 1e-4):
                    warnings.warn(
                        "Seasonal indexes close to zero. Using non-seasonal Theta method"
                    )
                    self.is_seasonal_ = False
                    self._y_adjusted = y.copy()
        else:
            self._y_adjusted = y.copy()
            self._seasonal_component = np.ones(self._n)
            self._trend = y.copy()

        # Estimate SES parameter alpha
        self.alpha_ = self._estimate_alpha(self._y_adjusted)
        self._ses_level_array = self._compute_ses_level(
            self._y_adjusted, self.alpha_
        )
        self.l_n_ = self._ses_level_array[-1]

        # Estimate baseline drift
        self.b0_ = self._estimate_drift(self._y_adjusted)

        # Compute context variable
        z_raw = self._compute_attention_context(self._y_adjusted, self.tau)
        self.mu_z_ = np.mean(z_raw)
        self.sigma_z_ = np.std(z_raw)
        self._z_star = (z_raw - self.mu_z_) / self.sigma_z_

        # Estimate gamma (or set to 0 if use_context=False)
        if self.use_context:
            if self.gamma_method == "cox":
                gamma_raw = self._estimate_gamma_cox(
                    self._z_star, self.risk_set_size
                )
            elif self.gamma_method == "regression":
                gamma_raw = self._estimate_gamma_regression(
                    self._y_adjusted,
                    self._z_star,
                    self._ses_level_array,
                    self.alpha_,
                    self.b0_,
                    self._n,
                )
            else:
                raise ValueError(f"Unknown gamma_method: {self.gamma_method}")

            # Apply stability constraint
            D_max = self._D_n(self.h, self.alpha_, self._n)
            stability_bound = 2.0 / (abs(self.b0_) * D_max + 1e-10)
            self.gamma_ = np.clip(
                gamma_raw,
                -self.stability_factor * stability_bound,
                self.stability_factor * stability_bound,
            )

            if abs(gamma_raw) > stability_bound:
                warnings.warn(
                    f"Gamma clipped for stability: {gamma_raw:.6f} -> {self.gamma_:.6f}"
                )

            self.method_ = "Context-Aware Theta"
        else:
            self.gamma_ = 0.0
            self.method_ = "Standard Theta"

        # Compute innovation variance
        residuals = self._y_adjusted[1:] - self._ses_level_array[:-1]
        self.sigma2_ = np.var(residuals, ddof=1)

        # Generate forecasts (deseasonalized)
        self._forecast_deseason = self._recursive_forecast(
            self._y_adjusted,
            self.l_n_,
            self.alpha_,
            self.b0_,
            self.gamma_,
            self.mu_z_,
            self.sigma_z_,
            self.h,
            self.tau,
            self._n,
        )

        # Compute prediction intervals (deseasonalized)
        self._lower_deseason, self._upper_deseason = self._prediction_intervals(
            self._forecast_deseason,
            self.alpha_,
            self.sigma2_,
            self.b0_,
            self.gamma_,
            self.sigma_z_,
            self._n,
        )

        return self

    def predict(self) -> Dict[str, np.ndarray]:
        """
        Generate forecasts and prediction intervals.

        Returns
        -------
        forecast : dict
            Dictionary containing:
            - 'mean': Point forecasts (H,)
            - 'lower': Lower prediction intervals (H, nlevels)
            - 'upper': Upper prediction intervals (H, nlevels)
            - 'level': Confidence levels
        """
        if self.alpha_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Reseasonalize if needed
        if self.is_seasonal_:
            # Extend seasonal pattern
            seasonal_indices = self._seasonal_component[-self._frequency :]
            seasonal_fc = np.tile(
                seasonal_indices, (self.h // self._frequency) + 1
            )[: self.h]

            if self.seasonal_method == "multiplicative":
                forecast_mean = self._forecast_deseason * seasonal_fc
                forecast_lower = (
                    self._lower_deseason * seasonal_fc[:, np.newaxis]
                )
                forecast_upper = (
                    self._upper_deseason * seasonal_fc[:, np.newaxis]
                )
            else:  # additive
                forecast_mean = self._forecast_deseason + seasonal_fc
                forecast_lower = (
                    self._lower_deseason + seasonal_fc[:, np.newaxis]
                )
                forecast_upper = (
                    self._upper_deseason + seasonal_fc[:, np.newaxis]
                )
        else:
            forecast_mean = self._forecast_deseason
            forecast_lower = self._lower_deseason
            forecast_upper = self._upper_deseason

        return {
            "mean": forecast_mean,
            "lower": forecast_lower,
            "upper": forecast_upper,
            "level": self.level,
        }

    @property
    def fitted_(self) -> np.ndarray:
        """Get fitted values (in-sample)."""
        if self.alpha_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if self.is_seasonal_:
            if self.seasonal_method == "multiplicative":
                return self._ses_level_array * self._seasonal_component
            else:  # additive
                return self._ses_level_array + self._seasonal_component
        else:
            return self._ses_level_array

    @property
    def residuals_(self) -> np.ndarray:
        """Get residuals (in-sample)."""
        if self.alpha_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        return self._y_original - self.fitted_

    def get_results(self) -> Dict:
        """
        Get detailed model results and diagnostics.

        Returns
        -------
        results : dict
            Dictionary containing:
            - 'parameters': Model parameters
            - 'diagnostics': Fit diagnostics
            - 'forecast': Forecast results
        """
        if self.alpha_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        forecast = self.predict()
        residuals = self.residuals_

        # Compute diagnostics
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals**2))

        D_max = self._D_n(self.h, self.alpha_, self._n)
        stability_bound = 2.0 / (abs(self.b0_) * D_max + 1e-10)
        stable = abs(self.gamma_) < stability_bound

        if self.gamma_ > 0:
            context_effect = "Amplifies drift when z* > 0"
        elif self.gamma_ < 0:
            context_effect = "Dampens drift when z* > 0"
        else:
            context_effect = "No context effect (standard Theta)"

        return {
            "parameters": {
                "alpha": self.alpha_,
                "b0": self.b0_,
                "gamma": self.gamma_,
                "sigma2": self.sigma2_,
                "l_n": self.l_n_,
            },
            "diagnostics": {
                "mae": mae,
                "rmse": rmse,
                "n_obs": self._n,
                "stable": stable,
                "stability_bound": stability_bound,
                "context_effect": context_effect,
            },
            "forecast": forecast,
            "method": self.method_,
            "seasonal": self.is_seasonal_,
            "seasonal_period": self.seasonal_period_,
        }

    def plot(self, ax=None, title=None, figsize=(12, 6)):
        """
        Plot fitted values, forecasts, and prediction intervals.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        title : str, optional
            Plot title. If None, uses method name.
        figsize : tuple, optional
            Figure size if creating new figure
        """
        if self.alpha_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        forecast = self.predict()

        # Create time indices
        train_idx = np.arange(self._n)
        forecast_idx = np.arange(self._n, self._n + self.h)

        # Plot training data
        ax.plot(
            train_idx,
            self._y_original,
            "o-",
            color="black",
            label="Observed",
            linewidth=1.5,
            markersize=3,
        )

        # Plot fitted values
        ax.plot(
            train_idx,
            self.fitted_,
            "-",
            color="orange",
            label="Fitted",
            linewidth=1,
            alpha=0.7,
        )

        # Plot forecasts
        ax.plot(
            forecast_idx,
            forecast["mean"],
            "s-",
            color="blue",
            label="Forecast",
            linewidth=2,
            markersize=4,
        )

        # Plot prediction intervals
        colors = plt.cm.Blues(np.linspace(0.3, 0.7, len(self.level)))
        for i, (lev, color) in enumerate(zip(self.level, colors)):
            ax.fill_between(
                forecast_idx,
                forecast["lower"][:, i],
                forecast["upper"][:, i],
                color=color,
                alpha=0.3,
                label=f"{lev}% PI",
            )

        # Vertical line at forecast origin
        ax.axvline(
            self._n - 0.5, color="red", linestyle="--", alpha=0.5, linewidth=1
        )

        # Labels and title
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        if title is None:
            title = f"{self.method_} (α={self.alpha_:.4f}, γ={self.gamma_:.6f})"
        ax.set_title(title, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        return ax
