import numpy as np
import matplotlib.pyplot as plt
import nnetsauce as ns
import joblib

from .confobayesianregr import ConformalBayesianRegressor
from sklearn.datasets import fetch_openml
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_random_state, shuffle
from time import time
from tqdm import tqdm


class ConformalBayesianClassifier(BaseEstimator, ClassifierMixin):

    # construct the object -----
    _estimator_type = "classifier"

    def __init__(
        self,
        obj=Ridge(),
        hyperparameter_bounds=None,
        n_samples=20,
        calibration_fraction=0.2,
        scaling_method="standard",
        random_state=None,
        verbose=True,
        n_jobs=-1,
        calibrated=False,
        calibration_method="sigmoid",
        calibration_cv=3,
    ):
        """
        Conformal Bayesian Classifier with optional probability calibration.

        Parameters
        ----------
        obj : estimator object
            Base regression estimator (default: Ridge())
        hyperparameter_bounds : dict, optional
            Bounds for hyperparameter sampling
        n_samples : int
            Number of ensemble models to train
        calibration_fraction : float
            Fraction of data to use for conformal calibration
        scaling_method : str
            Feature scaling method (default: "standard")
        random_state : int, optional
            Random seed for reproducibility
        verbose : bool
            Whether to show progress bars
        n_jobs : int
            Number of parallel jobs (-1 for all cores)
        calibrated : bool
            Whether to apply probability calibration (default: False)
        calibration_method : str
            Method for calibration: "sigmoid" or "isotonic" (default: "sigmoid")
        calibration_cv : int
            Number of CV folds for calibration (default: 3)
        """
        self.obj_base = obj
        self.hyperparameter_bounds = hyperparameter_bounds
        self.n_samples = n_samples
        self.calibration_fraction = calibration_fraction
        self.scaling_method = scaling_method
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.calibrated = calibrated
        self.calibration_method = calibration_method
        self.calibration_cv = calibration_cv

        self.is_fitted_ = False

    def fit(self, X, y):
        """
        Fit the conformal Bayesian classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self : object
            Fitted classifier
        """
        # Build the base conformal regressor
        base_regressor = ConformalBayesianRegressor(
            self.obj_base,
            hyperparameter_bounds=self.hyperparameter_bounds,
            n_samples=self.n_samples,
            calibration_fraction=self.calibration_fraction,
            scaling_method=self.scaling_method,
            random_state=self.random_state,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
        )

        # Wrap in multitask classifier
        self.obj = ns.SimpleMultitaskClassifier(base_regressor)

        if self.calibrated:
            # Fit the base classifier first
            self.obj.fit(X, y)

            # Then wrap with calibration
            self.calibrated_obj_ = CalibratedClassifierCV(
                self.obj,
                method=self.calibration_method,
                cv="prefit",  # Use prefit to avoid refitting
                ensemble=False,  # Don't create ensemble, use single model
            )

            # Fit calibration on the same data
            # (in practice, you might want to use a held-out calibration set)
            self.calibrated_obj_.fit(X, y)
            self.is_fitted_ = True
        else:
            self.obj.fit(X, y)
            self.is_fitted_ = True

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data

        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Predicted probabilities for each class
        """
        if not self.is_fitted_:
            raise RuntimeError("Fit the model first")

        if self.calibrated:
            return self.calibrated_obj_.predict_proba(X)
        else:
            return self.obj.predict_proba(X)

    def predict(self, X):
        """
        Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    @property
    def _estimator_type(self):
        return "classifier"
