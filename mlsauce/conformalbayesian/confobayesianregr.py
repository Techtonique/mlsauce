import numpy as np
import matplotlib.pyplot as plt
import joblib

from collections import namedtuple
from sklearn.datasets import fetch_openml
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_random_state, shuffle
from time import time
from tqdm import tqdm


class ConformalBayesianRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        obj=Ridge,
        level=95,
        hyperparameter_bounds=None,
        n_samples=20,
        calibration_fraction=0.2,
        scaling_method="standard",
        random_state=None,
        show_progress=False,
        verbose=True,
        n_jobs=-1,
    ):
        self.obj = obj
        self.level = level
        self.alpha_ = 1 - self.level / 100
        self.hyperparameter_bounds = hyperparameter_bounds
        self.n_samples = n_samples
        self.calibration_fraction = calibration_fraction
        self.scaling_method = scaling_method
        self.random_state = random_state
        self.verbose = verbose
        self.show_progress = show_progress
        self.n_jobs = n_jobs

        self.is_fitted_ = False

    def _sample_hyperparameters(self):
        # Simple uniform sampling or use fixed bounds
        configs = []
        for _ in range(self.n_samples):
            if self.hyperparameter_bounds:
                cfg = {}
                for k, v in self.hyperparameter_bounds.items():
                    if isinstance(v, list) and len(v) == 2:
                        # Always sample as float first
                        sampled_value = np.random.uniform(v[0], v[1])

                        # If both bounds are integers, assume integer parameter
                        if isinstance(v[0], (int, np.integer)) and isinstance(
                            v[1], (int, np.integer)
                        ):
                            cfg[k] = int(sampled_value)
                        else:
                            cfg[k] = sampled_value
                    else:
                        # Assume fixed value
                        cfg[k] = v
            else:
                cfg = {}
            configs.append(cfg)
        return configs

    def _train_models_parallel(self, X, y, configs):
        def train_model(cfg):
            self.obj.set_params(**cfg)
            self.obj.fit(X, y)
            return self.obj

        if self.show_progress == False:
            models = joblib.Parallel(n_jobs=self.n_jobs)(
                joblib.delayed(train_model)(cfg)
                for cfg in tqdm(configs, disable=not self.verbose)
            )
        else:
            models = joblib.Parallel(n_jobs=self.n_jobs)(
                joblib.delayed(train_model)(cfg) for cfg in configs
            )
        return models

    def _predict_models(self, models, X):
        if self.show_progress == False:
            preds = [m.predict(X) for m in models]
        else:
            preds = [
                m.predict(X) for m in tqdm(models, disable=not self.verbose)
            ]
        return np.column_stack(preds)

    def fit(self, X, y):
        if hasattr(X, "values"):  # keep DataFrame if possible
            X = X.copy()
        else:
            X = np.asarray(X)
        X = np.asarray(X)
        y = np.asarray(y)
        rng = check_random_state(self.random_state)
        X, y = shuffle(X, y, random_state=rng)
        # 1. Cluster with GMM (full covariance)
        n_clusters = min(10, X.shape[0] // 30)
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            random_state=self.random_state,
        )
        clusters = gmm.fit_predict(X)
        # 2. Stratified train/calibration split
        X_train, X_calib, y_train, y_calib = train_test_split(
            X,
            y,
            test_size=self.calibration_fraction,
            random_state=self.random_state,
            stratify=clusters,
        )
        # 3. Scale features
        if self.scaling_method == "standard":
            scaler = StandardScaler().fit(X_train)
        else:
            raise ValueError("Scaling method must be 'standard'")
        self.scaler_ = scaler
        X_train_s = scaler.transform(X_train)
        X_calib_s = scaler.transform(X_calib)
        # 4. Train ensemble
        configs = self._sample_hyperparameters()
        self.models_ = self._train_models_parallel(X_train_s, y_train, configs)
        # 5. Calibration residuals
        preds_calib = self._predict_models(self.models_, X_calib_s)
        self.calibration_residuals_ = y_calib - np.median(preds_calib, axis=1)
        self.is_fitted_ = True
        return self

    def predict(self, X, return_pi=False):
        """Obtain predictions and prediction intervals

        Args:

            X: array-like, shape = [n_samples, n_features];
                Testing set vectors, where n_samples is the number
                of samples and n_features is the number of features.

            return_pi: boolean
                Whether the prediction interval is returned or not.
                Default is False, for compatibility with other _estimators_.
                If True, a tuple containing the predictions + lower and upper
                bounds is returned.

        """
        if not self.is_fitted_:
            raise RuntimeError("Fit the model first")
        X_s = self.scaler_.transform(X)
        preds = self._predict_models(self.models_, X_s)
        self.mean_ = np.median(preds, axis=1)
        if return_pi == False:
            return self.mean_
        DescribeResult = namedtuple(
            "DescribeResult", ("mean", "lower", "upper")
        )
        q = np.quantile(self.calibration_residuals_, q=self.alpha_ / 200)
        return DescribeResult(self.mean_, self.mean_ + q, self.mean_ - q)

    def get_coverage(self, y_true, lower, upper):
        return np.mean((y_true >= lower) & (y_true <= upper))
