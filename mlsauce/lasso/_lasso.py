import numpy as np
import pickle
import platform
import warnings
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from numpy.linalg import inv

try:
    from . import _lassoc as mo
except ImportError:
    import _lassoc as mo
from ..utils import get_beta, check_and_install

try:
    import jax.numpy as jnp
    from jax import device_put
    from jax.numpy.linalg import inv as jinv
except ImportError:
    pass


class LassoRegressor(BaseEstimator, RegressorMixin):
    """Lasso.

    Attributes:

        reg_lambda: float
            L1 regularization parameter.

        max_iter: int
            number of iterations of lasso shooting algorithm.

        tol: float
            tolerance for convergence of lasso shooting algorithm.

        backend: str
            type of backend; must be in ('cpu', 'gpu', 'tpu').

    """

    def __init__(self, reg_lambda=0.1, max_iter=10, tol=1e-3, backend="cpu"):
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
        self.max_iter = max_iter
        self.tol = tol
        self.backend = backend
        if self.backend in ("gpu", "tpu"):
            check_and_install("jax")
            check_and_install("jaxlib")

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

        self.ym, centered_y = mo.center_response(y)
        self.xm = X.mean(axis=0)
        self.xsd = X.std(axis=0)
        self.xsd[self.xsd == 0] = 1
        X_ = (X - self.xm[None, :]) / self.xsd[None, :]
        XX = mo.crossprod(X_, backend=self.backend)
        Xy = mo.crossprod(X_, centered_y, backend=self.backend)
        XX2 = 2 * XX
        Xy2 = 2 * Xy

        if self.backend == "cpu":
            # beta0, _, _, _ = np.linalg.lstsq(X_, centered_y, rcond=None)
            beta0 = get_beta(X_, centered_y)
            if len(np.asarray(y).shape) == 1:
                res = mo.get_beta_1D(
                    beta0=np.asarray(beta0),
                    XX2=np.asarray(XX2),
                    Xy2=np.asarray(Xy2),
                    reg_lambda=self.reg_lambda,
                    max_iter=self.max_iter,
                    tol=self.tol,
                )
                self.beta = res[0]
                return self

            res = mo.get_beta_2D(
                beta0=np.asarray(beta0),
                XX2=np.asarray(XX2),
                Xy2=np.asarray(Xy2),
                reg_lambda=self.reg_lambda,
                max_iter=self.max_iter,
                tol=self.tol,
            )
            self.beta = res[0]
            return self

        invXX = jinv(XX + self.reg_lambda * jnp.eye(X_.shape[1]))
        beta0 = mo.safe_sparse_dot(invXX, Xy, backend=self.backend)
        if len(np.asarray(y).shape) == 1:
            res = mo.get_beta_1D(
                beta0=np.asarray(beta0),
                XX2=np.asarray(XX2),
                Xy2=np.asarray(Xy2),
                reg_lambda=self.reg_lambda,
                max_iter=self.max_iter,
                tol=self.tol,
            )
            self.beta = res[0]
            return self

        res = mo.get_beta_2D(
            beta0=np.asarray(beta0),
            XX2=np.asarray(XX2),
            Xy2=np.asarray(Xy2),
            reg_lambda=self.reg_lambda,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        self.beta = res[0]
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
        X_ = (X - self.xm[None, :]) / self.xsd[None, :]

        if self.backend == "cpu":
            if isinstance(self.ym, float):
                return self.ym + mo.safe_sparse_dot(X_, self.beta)
            return self.ym[None, :] + mo.safe_sparse_dot(X_, self.beta)

        # if self.backend in ("gpu", "tpu"):
        if isinstance(self.ym, float):
            return self.ym + mo.safe_sparse_dot(
                X_, self.beta, backend=self.backend
            )
        return self.ym[None, :] + mo.safe_sparse_dot(
            X_, self.beta, backend=self.backend
        )
