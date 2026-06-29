import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time
from collections import namedtuple
from time import time
from sklearn.base import BaseEstimator, RegressorMixin

# Define the return type
Prediction = namedtuple("Prediction", ["mean", "lower", "upper"])


class RVFLJackknifePlus(BaseEstimator, RegressorMixin):
    """RVFL network with closed-form jackknife+ prediction intervals.

    Parameters
    ----------
    n_hidden : int
        Number of random hidden features. Set to 0 for plain ridge regression.
    lambda_ : float
        Ridge penalty for the read-out layer.
    activation : {"tanh", "relu", "sigmoid"}
        Nonlinearity for the random hidden layer.
    random_state : int
        Seed for random hidden-layer weights.
    symmetric : bool
        If True, use symmetric jackknife+ (absolute residuals).
        If False (default), use asymmetric jackknife+.
    """

    def __init__(
        self,
        n_hidden=200,
        lambda_=1.0,
        activation="tanh",
        random_state=0,
        symmetric=False,
    ):
        self.n_hidden = n_hidden
        self.lambda_ = lambda_
        self.activation = activation
        self.random_state = random_state
        self.symmetric = symmetric

    def _act(self, U):
        if self.activation == "tanh":
            return np.tanh(U)
        if self.activation == "relu":
            return np.maximum(U, 0.0)
        if self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-U))
        raise ValueError(f"unknown activation '{self.activation}'")

    def _augment(self, X):
        if self.n_hidden == 0:
            return X
        H = self._act(X @ self.W_ + self.b_)
        return np.hstack([X, H])

    def fit(self, X, y):
        n, p = X.shape

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = np.asarray(y).ravel()

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Fixed random hidden layer
        rng = np.random.default_rng(self.random_state)
        self.W_ = rng.normal(scale=1.0 / np.sqrt(p), size=(p, self.n_hidden))
        self.b_ = rng.normal(scale=1.0, size=(self.n_hidden,))

        Z = self._augment(X)
        q = Z.shape[1]

        # Center response (intercept not penalized)
        self.ybar_ = y.mean()
        yc = y - self.ybar_

        # Ridge solution
        A = Z.T @ Z + self.lambda_ * np.eye(q)
        self.A_inv_ = np.linalg.inv(A)
        self.beta_ = self.A_inv_ @ Z.T @ yc
        self.Z_train_ = Z

        # Closed-form LOO residuals (memory efficient)
        h = np.sum((Z @ self.A_inv_) * Z, axis=1)  # diag(Z @ A_inv @ Z.T)
        e = yc - Z @ self.beta_  # in-sample residuals
        self.r_ = e / np.maximum(1.0 - h, 1e-10)  # LOO residuals

        return self

    def predict(self, X, alpha=0.1, return_pi=False):
        """Predict and optionally return prediction intervals.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Test data.
        alpha : float, default=0.1
            Significance level for prediction intervals (1-alpha coverage).
        return_pi : bool, default=False
            If True, return Prediction namedtuple with mean, lower, and upper.
            If False, return only the mean predictions.

        Returns
        -------
        If return_pi=False:
            y_pred : array, shape (n_samples,)
                Mean predictions.
        If return_pi=True:
            Prediction : namedtuple
                Named tuple with fields 'mean', 'lower', 'upper'.
        """
        Z = self._augment(X)
        yhat = Z @ self.beta_ + self.ybar_

        if not return_pi:
            return yhat

        # Cross-term: n_test x n_train
        G = Z @ self.A_inv_ @ self.Z_train_.T

        # f^{-i}(x_j) = f(x_j) - G[j,i] * r_i (Sherman-Morrison)
        loo_pred = yhat[:, None] - G * self.r_[None, :]

        if self.symmetric:
            # Symmetric version: use absolute residuals
            scores = np.abs(loo_pred - yhat[:, None]) + np.abs(self.r_[None, :])
            q = np.quantile(scores, 1 - alpha, axis=1)
            lo = yhat - q
            hi = yhat + q
        else:
            # Asymmetric version: use signed residuals
            scores = loo_pred + self.r_[None, :]
            lo = np.quantile(scores, alpha / 2, axis=1)
            hi = np.quantile(scores, 1 - alpha / 2, axis=1)

        return Prediction(mean=yhat, lower=lo, upper=hi)
