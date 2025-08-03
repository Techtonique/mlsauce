import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.base import clone, RegressorMixin
from sklearn.gaussian_process.kernels import Matern, RBF
from statsmodels.tsa.ar_model import AutoReg
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import RidgeCV
from typing import Optional, Literal, Any
import matplotlib.pyplot as plt
from sklearn.utils.validation import check_is_fitted

class FunctionalForecaster:
    def __init__(
        self,
        n_components: int = 8,
        kernel: Literal['rbf', 'matern'] = 'rbf',
        length_scale: float = 50.0,
        nu: float = 1.5,
        decomposition_method: Literal['kernel_eigen', 'kpca', 'both'] = 'both',
        regression_model: Optional[RegressorMixin] = None,
        **regression_kwargs: Any
    ):
        """
        Functional time series forecaster with rolling coefficient estimation.

        Parameters
        ----------
        n_components : int
            Number of latent components to extract.
        kernel : {'rbf', 'matern'}
            Kernel type for functional decomposition.
        length_scale : float
            Kernel length scale parameter.
        nu : float
            Matern kernel smoothness parameter.
        decomposition_method : {'kernel_eigen', 'kpca', 'both'}
            Method for functional decomposition.
        regression_model : RegressorMixin, optional
            Custom sklearn regressor (must have coef_ attribute).
        **regression_kwargs
            Arguments for default RidgeCV if no model provided.
        """
        self.n_components = n_components
        self.kernel = kernel
        self.length_scale = length_scale
        self.nu = nu
        self.decomposition_method = decomposition_method
        self.regression_model = regression_model or RidgeCV(**regression_kwargs)

    def _compute_kernel_matrix(self, grid: np.ndarray) -> np.ndarray:
        """Compute kernel matrix for functional decomposition."""
        if self.kernel == 'matern':
            return Matern(length_scale=self.length_scale, nu=self.nu)(grid.reshape(-1, 1))
        elif self.kernel == 'rbf':
            return RBF(length_scale=self.length_scale)(grid.reshape(-1, 1))
        else:
            raise ValueError("Kernel must be 'rbf' or 'matern'.")

    def fit(
        self,
        X: np.ndarray,
        grid: np.ndarray,
        rolling_window: Optional[int] = None
    ) -> 'FunctionalForecaster':
        """
        Fit model with optional rolling window training.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_points)
            Functional time series data.
        grid : np.ndarray, shape (n_points,)
            Domain points of the functional data.
        rolling_window : int, optional
            If provided, fits regression on sliding windows to track time-varying coefficients.
        """
        self.grid = grid
        self.X_ = X
        self.X_mean = X.mean(axis=0)
        X_centered = X - self.X_mean

        # Kernel decomposition
        kernel_mat = self._compute_kernel_matrix(grid)
        
        # Mercer eigenfunctions
        if self.decomposition_method in ['kernel_eigen', 'both']:
            eigenvalues, eigenfunctions = np.linalg.eigh(kernel_mat)
            idx = np.argsort(eigenvalues)[::-1]
            self.kernel_eigenvalues_ = eigenvalues[idx][:self.n_components]
            self.kernel_eigenfunctions_ = eigenfunctions[:, idx][:, :self.n_components]
            self.kernel_scores_ = X_centered @ self.kernel_eigenfunctions_

        # Kernel PCA
        if self.decomposition_method in ['kpca', 'both']:
            self.kpca = KernelPCA(
                n_components=self.n_components,
                kernel='precomputed' if self.kernel == 'matern' else self.kernel,
                gamma=1/(2*self.length_scale**2),
                fit_inverse_transform=True
            )
            self.kpca.fit(kernel_mat if self.kernel == 'matern' else X_centered)
            self.kpca_scores_ = X_centered @ self.kpca.alphas_ if self.kernel == 'matern' else self.kpca.transform(X_centered)
            self.kpca_components_ = self.kpca.inverse_transform(np.eye(self.n_components)).T

        # Select active decomposition
        self.scores_ = getattr(self, f"{self.decomposition_method}_scores_")
        self.components_ = getattr(self, f"{self.decomposition_method}_components_")

        # Rolling coefficient estimation
        if rolling_window is not None:
            self.rolling_coefs_ = []
            for i in range(len(self.scores_) - rolling_window + 1):
                window_scores = self.scores_[i:i+rolling_window]
                window_X = X_centered[i:i+rolling_window]
                
                model = clone(self.regression_model)
                model.fit(window_scores, window_X)
                self.rolling_coefs_.append(model.coef_)
            
            # Forecast coefficients using AR(1)
            self.coef_forecaster_ = [
                AutoReg(np.array(self.rolling_coefs_)[:, k, :], lags=1).fit()
                for k in range(self.n_components)
            ]
        else:
            # Standard full-batch fit
            self.regression_model.fit(self.scores_, X_centered)

        return self

    def forecast(self, steps: int = 5) -> np.ndarray:
        """
        Forecast using time-varying coefficients.

        Parameters
        ----------
        steps : int
            Number of steps to forecast.

        Returns
        -------
        np.ndarray, shape (steps, n_points)
            Forecasted curves.
        """
        check_is_fitted(self)
        
        # Case 1: Use pre-computed coefficient forecasts
        if hasattr(self, 'coef_forecaster_'):
            forecast_coefs = np.zeros((steps, self.n_components, len(self.grid)))
            for k in range(self.n_components):
                forecast_coefs[:, k, :] = self.coef_forecaster_[k].predict(
                    start=len(self.rolling_coefs_),
                    end=len(self.rolling_coefs_)+steps-1
                )
            return self.X_mean + np.einsum('tk,kd->td', self.scores_[-steps:], forecast_coefs.mean(axis=0))
        
        # Case 2: Standard AR on scores
        forecast_scores = np.zeros((steps, self.n_components))
        for k in range(self.n_components):
            model = AutoReg(self.scores_[:, k], lags=1).fit()
            forecast_scores[:, k] = model.predict(
                start=len(self.scores_),
                end=len(self.scores_)+steps-1
            )
        return self.X_mean + forecast_scores @ self.regression_model.coef_.T

    def select_components(
        self,
        k: int = 3,
        decomposition_source: Optional[Literal['kernel_eigen', 'kpca']] = None
    ) -> np.ndarray:
        """
        Select top-k most predictive components using F-statistics.

        Parameters
        ----------
        k : int
            Number of components to select.
        decomposition_source : {'kernel_eigen', 'kpca'}, optional
            Which decomposition to use (defaults to active method in fit).

        Returns
        -------
        np.ndarray
            Indices of selected components.
        """
        if decomposition_source is None:
            scores = self.scores_
        else:
            scores = getattr(self, f"{decomposition_source}_scores_")

        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(scores, self.X_ - self.X_mean)
        return selector.get_support(indices=True)

    def plot_components(
        self,
        n_plot: int = 3,
        decomposition_source: Optional[Literal['kernel_eigen', 'kpca']] = None
    ) -> None:
        """
        Visualize functional components.

        Parameters
        ----------
        n_plot : int
            Number of components to plot.
        decomposition_source : {'kernel_eigen', 'kpca'}, optional
            Which decomposition to visualize (defaults to active method in fit).
        """
        if decomposition_source is None:
            components = self.components_
            title = f"{self.decomposition_method} Components"
        else:
            components = getattr(self, f"{decomposition_source}_components_")
            title = f"{decomposition_source} Components"

        plt.figure(figsize=(10, 5))
        for i in range(min(n_plot, self.n_components)):
            plt.plot(self.grid, components[:, i], label=f'Component {i+1}')
        plt.title(title)
        plt.xlabel("Domain")
        plt.ylabel("Component Value")
        plt.legend()
        plt.show()