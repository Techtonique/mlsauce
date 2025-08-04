import numpy as np
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.ar_model import AutoReg
from sklearn.utils.validation import check_is_fitted, check_array
from typing import Optional, Literal
import matplotlib.pyplot as plt


class FunctionalForecaster(BaseEstimator, RegressorMixin):
    """
    Functional time series forecaster using Functional Principal Component Regression (FPCR).
    
    Two approaches:
    1. Use principal components as regressors (method='components')
    2. Use reduced features at inference (method='features')
    
    Parameters
    ----------
    n_components : int, default=8
        Number of principal components to extract.
    method : {'components', 'features'}, default='components'
        'components': Use principal components as regressors
        'features': Use reduced features directly for forecasting
    rolling_window : int, optional
        Window size for rolling regression. If None, uses full training set.
    forecast_method : {'ar', 'last_value'}, default='ar'
        Method for forecasting coefficients.
    """
    
    def __init__(
        self,
        n_components: int = 8,
        method: Literal['components', 'features'] = 'components',
        rolling_window: Optional[int] = None,
        forecast_method: Literal['ar', 'last_value'] = 'ar'
    ):
        self.n_components = n_components
        self.method = method
        self.rolling_window = rolling_window
        self.forecast_method = forecast_method
        
    def fit(self, X: np.ndarray) -> 'FunctionalForecaster':
        """
        Fit the functional forecaster.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_points)
            Functional time series data.
            
        Returns
        -------
        self : object
            Fitted forecaster.
        """
        # Input validation
        X = check_array(X)
        
        self.X_ = X
        self.n_samples_, self.n_points_ = X.shape
        
        # Reduce dimensionality with PCA
        self.X_mean_ = X.mean(axis=0)
        X_centered = X - self.X_mean_
        
        self.pca_ = PCA(n_components=self.n_components)
        self.reduced_features_ = self.pca_.fit_transform(X_centered)  # (n_samples, n_components)
        self.components_ = self.pca_.components_.T  # (n_points, n_components)
        
        # Fit based on method
        if self.method == 'components':
            self._fit_components_method()
        else:  # features
            self._fit_features_method()
            
        self.is_fitted_ = True
        return self
    
    def _fit_components_method(self):
        """Method 1: Use principal components as regressors."""
        if self.rolling_window is not None:
            self._fit_rolling_components()
        else:
            self._fit_full_components()
    
    def _fit_features_method(self):
        """Method 2: Use reduced features directly."""
        if self.rolling_window is not None:
            self._fit_rolling_features()
        else:
            self._fit_full_features()
    
    def _fit_rolling_components(self):
        """Rolling regression using principal components as regressors."""
        self.rolling_coefs_ = []
        
        for i in range(len(self.X_) - self.rolling_window):
            # Use principal components as regressors to predict next functional curve
            X_regressors = self.components_.T  # (n_components, n_points) - the basis functions
            y_target = self.X_[i+self.rolling_window]  # Next functional curve
            
            # Fit regression: components -> functional_data
            coefs = np.linalg.lstsq(X_regressors.T, y_target, rcond=None)[0]
            self.rolling_coefs_.append(coefs)  # (n_components,)
    
    def _fit_full_components(self):
        """Full regression using principal components as regressors."""
        # Use principal components as regressors for all functional data
        X_regressors = self.components_.T  # (n_components, n_points)
        coefs = np.linalg.lstsq(X_regressors.T, self.X_, rcond=None)[0].T
        self.coefs_ = coefs  # (n_samples, n_components)
    
    def _fit_rolling_features(self):
        """Rolling regression using reduced features."""
        self.rolling_coefs_ = []
        
        for i in range(len(self.reduced_features_) - self.rolling_window):
            # Use window of reduced features to predict next time step
            X_window = self.reduced_features_[i:i+self.rolling_window]
            y_window = self.X_[i+self.rolling_window]  # Predict next functional curve
            
            # Fit regression: reduced_features -> functional_data
            coefs = np.linalg.lstsq(X_window, y_window, rcond=None)[0]
            self.rolling_coefs_.append(coefs)  # (n_components,)
    
    def _fit_full_features(self):
        """Full regression using reduced features."""
        # Use all reduced features to predict functional data
        coefs = np.linalg.lstsq(self.reduced_features_, self.X_, rcond=None)[0].T
        self.coefs_ = coefs  # (n_points, n_components)
    
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
        check_is_fitted(self, 'is_fitted_')
        
        if self.method == 'components':
            return self._forecast_components(steps)
        else:  # features
            return self._forecast_features(steps)
    
    def _forecast_components(self, steps: int) -> np.ndarray:
        """Forecast using principal components method."""
        if self.rolling_window is not None:
            # Forecast rolling coefficients
            coef_series = np.array(self.rolling_coefs_)  # (n_windows, n_components)
            
            forecasted_coefs = np.zeros((steps, self.n_components))
            for comp in range(self.n_components):
                if self.forecast_method == 'ar':
                    try:
                        ar_model = AutoReg(coef_series[:, comp], lags=1).fit()
                        forecasted_coefs[:, comp] = ar_model.predict(
                            start=len(coef_series),
                            end=len(coef_series) + steps - 1
                        )
                    except:
                        forecasted_coefs[:, comp] = coef_series[-1, comp]
                else:  # last_value
                    forecasted_coefs[:, comp] = coef_series[-1, comp]
            
            # Reconstruct: forecasted_coefficients @ components
            forecasts = self.X_mean_ + forecasted_coefs @ self.components_.T
            
        else:
            # Forecast full coefficients
            forecasted_coefs = np.zeros((steps, self.n_components))
            for comp in range(self.n_components):
                if self.forecast_method == 'ar':
                    try:
                        ar_model = AutoReg(self.coefs_[:, comp], lags=1).fit()
                        forecasted_coefs[:, comp] = ar_model.predict(
                            start=len(self.coefs_),
                            end=len(self.coefs_) + steps - 1
                        )
                    except:
                        forecasted_coefs[:, comp] = self.coefs_[-1, comp]
                else:  # last_value
                    forecasted_coefs[:, comp] = self.coefs_[-1, comp]
            
            # Reconstruct: forecasted_coefficients @ components
            forecasts = self.X_mean_ + forecasted_coefs @ self.components_.T
        
        return forecasts
    
    def _forecast_features(self, steps: int) -> np.ndarray:
        """Forecast using reduced features method."""
        if self.rolling_window is not None:
            # Forecast rolling coefficients
            coef_series = np.array(self.rolling_coefs_)  # (n_windows, n_components)
            
            forecasted_coefs = np.zeros((steps, self.n_components))
            for comp in range(self.n_components):
                if self.forecast_method == 'ar':
                    try:
                        ar_model = AutoReg(coef_series[:, comp], lags=1).fit()
                        forecasted_coefs[:, comp] = ar_model.predict(
                            start=len(coef_series),
                            end=len(coef_series) + steps - 1
                        )
                    except:
                        forecasted_coefs[:, comp] = coef_series[-1, comp]
                else:  # last_value
                    forecasted_coefs[:, comp] = coef_series[-1, comp]
            
            # Reconstruct using forecasted coefficients
            forecasts = self.X_mean_ + forecasted_coefs @ self.components_.T
            
        else:
            # Forecast reduced features directly
            forecasted_features = np.zeros((steps, self.n_components))
            for comp in range(self.n_components):
                if self.forecast_method == 'ar':
                    try:
                        ar_model = AutoReg(self.reduced_features_[:, comp], lags=1).fit()
                        forecasted_features[:, comp] = ar_model.predict(
                            start=len(self.reduced_features_),
                            end=len(self.reduced_features_) + steps - 1
                        )
                    except:
                        forecasted_features[:, comp] = self.reduced_features_[-1, comp]
                else:  # last_value
                    forecasted_features[:, comp] = self.reduced_features_[-1, comp]
            
            # Reconstruct using coefficient matrix
            forecasts = self.X_mean_ + forecasted_features @ self.coefs_.T
        
        return forecasts
    
    def plot_components(self, n_plot: int = 3) -> None:
        """Plot functional principal components."""
        if not hasattr(self, 'components_'):
            raise ValueError("Model must be fitted before plotting components.")
        
        plt.figure(figsize=(12, 6))
        for i in range(min(n_plot, self.n_components)):
            plt.plot(self.components_[:, i], label=f'Component {i+1}', linewidth=2)
        
        plt.title(f'Functional Principal Components (n_components={self.n_components})')
        plt.xlabel('Domain Point')
        plt.ylabel('Component Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_reduced_features(self, n_plot: int = 4) -> None:
        """Plot reduced features over time."""
        if not hasattr(self, 'reduced_features_'):
            raise ValueError("Model must be fitted before plotting reduced features.")
        
        plt.figure(figsize=(12, 8))
        for i in range(min(n_plot, self.n_components)):
            plt.subplot(2, 2, i+1)
            plt.plot(self.reduced_features_[:, i], 'o-', linewidth=2, markersize=4)
            plt.title(f'Reduced Feature {i+1}')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_forecast(self, actual: Optional[np.ndarray] = None, steps: int = 5) -> None:
        """Plot forecasted curves."""
        forecasts = self.forecast(steps=steps)
        
        plt.figure(figsize=(12, 6))
        
        if actual is not None:
            # Plot actual data
            for i in range(min(5, len(actual))):
                plt.plot(actual[i], 'k-', alpha=0.3, linewidth=1)
        
        # Plot forecasts
        for i in range(steps):
            plt.plot(forecasts[i], 'r--', linewidth=2, alpha=0.7, label=f'Forecast {i+1}' if i == 0 else "")
        
        plt.title(f'Functional Forecast ({self.method} method)')
        plt.xlabel('Domain Point')
        plt.ylabel('Value')
        if actual is not None:
            plt.legend(['Actual', 'Forecast'])
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def get_model_info(self) -> dict:
        """Get information about the fitted model."""
        info = {
            'n_components': self.n_components,
            'method': self.method,
            'rolling_window': self.rolling_window,
            'forecast_method': self.forecast_method,
            'is_fitted': getattr(self, 'is_fitted_', False)
        }
        
        if hasattr(self, 'reduced_features_'):
            info.update({
                'n_samples': len(self.reduced_features_),
                'n_points': self.n_points_,
                'explained_variance_ratio': self.pca_.explained_variance_ratio_.tolist()
            })
        
        return info