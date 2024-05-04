import numpy as np
import platform
import warnings
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from numpy.linalg import inv
from ..utils import get_beta
from ._enet import fit_elasticnet, predict_elasticnet

if platform.system() in ("Linux", "Darwin"):
    import jax.numpy as jnp
    from jax import device_put
    from jax.numpy.linalg import inv as jinv


class ElasticNetRegressor(BaseEstimator, RegressorMixin):
    """Elasticnet.

    Attributes:

        reg_lambda: float
            regularization parameter.

        alpha: float
            compromise between L1 and L2 regularization (must be in [0, 1]),
            for `solver` == 'enet'.

        backend: str
            type of backend; must be in ('cpu', 'gpu', 'tpu')

    """

    def __init__(self, reg_lambda=0.1, alpha=0.5, backend="cpu"):
        assert backend in (
            "cpu",
            "gpu",
            "tpu",
        ), "`backend` must be in ('cpu', 'gpu', 'tpu')"

        sys_platform = platform.system()

        if (sys_platform == "Windows") and (backend in ("gpu", "tpu")):
            warnings.warn(
                "No GPU/TPU computing on Windows yet, backend set to 'cpu'"
            )
            backend = "cpu"

        self.reg_lambda = reg_lambda
        self.alpha = alpha
        self.backend = backend

    def fit(self, X, y, **kwargs):
        """Fit matrixops (classifier) to training data (X, y)

        Args:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features.

            y: array-like, shape = [n_samples]
                Target values.

            **kwargs: additional parameters to be passed to self.cook_training_set.

        Returns:

            self: object.

        """
        fit_result = fit_elasticnet(X, y, lam=self.reg_lambda, alpha=self.alpha)
        self.coef_ = fit_result.coef_
        self.y_train_mean = fit_result.y_train_mean
        self.scaler = fit_result.scaler
        self.converged = fit_result.converged
        return self

    def predict(self, X, **kwargs):
        """Predict test data X.

        Args:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features.

            **kwargs: additional parameters to be passed to `predict_proba`

        Returns:

            model predictions: {array-like}

        """
        return predict_elasticnet(X, self)
