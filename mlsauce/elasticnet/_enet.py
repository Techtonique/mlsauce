import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from time import time
try:
    import jax.numpy as jnp
    from jax import device_put
    from jax.numpy.linalg import inv as jinv
except ImportError:
    pass
from ..utils.get_beta import get_beta

def soft_thresholding(z, gamma):
    return np.sign(z) * np.maximum(np.abs(z) - gamma, 0)


def l1_norm(x):
    return np.sum(np.abs(x))


def fit_elasticnet(X_train, y_train, lam=1, alpha=0.5, tol=1e-5, max_iter=50, backend="cpu"):

    lam_alpha = lam * alpha

    scaler = StandardScaler(with_mean=True, with_std=True)

    scaled_X_train = scaler.fit_transform(X_train, y_train)

    y_train_mean = y_train.mean()

    centered_y_train = y_train - y_train_mean

    beta_prev = get_beta(scaled_X_train, centered_y_train, backend=backend) # /!\ this is key

    p_train = X_train.shape[1]
    beta_prevs = np.repeat(beta_prev, repeats=X_train.shape[1]).reshape(
        p_train, p_train
    )
    np.fill_diagonal(beta_prevs, 0)

    centered_y_train_tilde = scaled_X_train @ beta_prevs

    weighted_residuals = np.mean(
        scaled_X_train * centered_y_train[:, None] - centered_y_train_tilde,
        axis=0,
    )

    beta = soft_thresholding(weighted_residuals, gamma=lam_alpha) / (
        1 + lam_alpha
    )

    converged = l1_norm(beta_prev - beta) < tol

    iteration = 0

    while converged == False and iteration < max_iter:
        beta_prev = beta
        beta_prevs = np.repeat(beta_prev, repeats=p_train).reshape(
            p_train, p_train
        )
        np.fill_diagonal(beta_prevs, 0)
        centered_y_train_tilde = scaled_X_train @ beta_prevs
        weighted_residuals = np.mean(
            scaled_X_train * centered_y_train[:, None] - centered_y_train_tilde,
            axis=0,
        )
        beta = soft_thresholding(weighted_residuals, gamma=lam_alpha) / (
            1 + lam_alpha
        )
        tol_ = l1_norm(beta_prev - beta)
        converged = tol_ < tol
        iteration += 1

    DescribeResult = namedtuple(
        "DescribeResult",
        ("coef_", "iteration_", "tol_", "converged", "y_train_mean", "scaler"),
    )

    return DescribeResult(
        beta, iteration, tol_, converged, y_train_mean, scaler
    )


def predict_elasticnet(X_test, fitted_elasticnet, backend="cpu"):
    if backend == "cpu":
        return (
            fitted_elasticnet.y_train_mean
            + fitted_elasticnet.scaler.transform(X_test) @ fitted_elasticnet.coef_
        )
    X_test_ = fitted_elasticnet.scaler.transform(X_test)
    device_put(X_test_)
    device_put(fitted_elasticnet.coef_)
    return (fitted_elasticnet.y_train_mean + jnp.matmul(X_test_, fitted_elasticnet.coef_))
