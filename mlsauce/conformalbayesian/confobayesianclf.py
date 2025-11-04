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

    def __init__(self, obj, hyperparameter_bounds=None,
                 n_samples=20, calibration_fraction=0.2,
                 scaling_method="standard", random_state=None,
                 verbose=True, n_jobs=-1):
        self.obj = ConformalBayesianRegressor(obj, 
                                               hyperparameter_bounds=hyperparameter_bounds,
                                               n_samples=n_samples,
                                               calibration_fraction=calibration_fraction,
                                               scaling_method=scaling_method,
                                               random_state=random_state,
                                               verbose=verbose,
                                               n_jobs=n_jobs)
        self.obj = ns.SimpleMultitaskClassifier(self.obj)
        self.obj = CalibratedClassifierCV(self.obj, cv=3)
        self.is_fitted_ = False
    
    def fit(self, X, y):
        self.obj.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        if not self.is_fitted_:
            raise RuntimeError("Fit the model first")
        return self.obj.predict_proba(X)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    @property
    def _estimator_type(self):
        return "classifier"
