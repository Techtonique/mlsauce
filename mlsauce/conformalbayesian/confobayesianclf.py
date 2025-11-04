import numpy as np
import matplotlib.pyplot as plt
import nnetsauce as ns
import joblib

from .confobayesianregr import ConformalBayesianRegressor
from sklearn.datasets import fetch_openml
from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_classifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
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
        use_classifier=True,
    ):
        """
        Conformal Bayesian Classifier with optional probability calibration.

        Parameters
        ----------
        obj : estimator object
            Base estimator (default: Ridge() for regression-based, or classifier if use_classifier=True)
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
        use_classifier : bool
            If True, use sklearn-like classifier with probability averaging.
            If False, use regression-based approach with SimpleMultitaskClassifier (default: False)
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
        self.use_classifier = use_classifier

        self.is_fitted_ = False

    def _sample_hyperparameters(self):
        """Sample hyperparameters for ensemble members"""
        configs = []

        # Get parameter constraints from the model if available
        param_constraints = {}
        if hasattr(self.obj_base, "_parameter_constraints"):
            param_constraints = self.obj_base._parameter_constraints

        for _ in range(self.n_samples):
            if self.hyperparameter_bounds:
                cfg = {}
                for k, v in self.hyperparameter_bounds.items():
                    if isinstance(v, list) and len(v) == 2:
                        # If both bounds are integers, assume integer parameter
                        if isinstance(v[0], (int, np.integer)) and isinstance(
                            v[1], (int, np.integer)
                        ):
                            cfg[k] = np.random.randint(v[0], v[1] + 1)
                        else:
                            cfg[k] = np.random.uniform(v[0], v[1])
                    else:
                        # Assume fixed value
                        cfg[k] = v
            else:
                cfg = {}
            configs.append(cfg)
        return configs

    def _train_classifiers_parallel(self, X, y, configs):
        """Train ensemble of classifiers in parallel"""

        def train_classifier(cfg):
            clf = clone(self.obj_base)
            clf.set_params(**cfg)
            clf.fit(X, y)
            return clf

        if self.verbose:
            models = joblib.Parallel(n_jobs=self.n_jobs)(
                joblib.delayed(train_classifier)(cfg)
                for cfg in tqdm(configs, desc="Training classifiers")
            )
        else:
            models = joblib.Parallel(n_jobs=self.n_jobs)(
                joblib.delayed(train_classifier)(cfg) for cfg in configs
            )
        return models

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
        X = np.asarray(X)
        y = np.asarray(y)

        # Store classes
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        if self.use_classifier:
            # Simpler approach: use sklearn-like classifier with probability averaging
            if not is_classifier(self.obj_base):
                raise ValueError(
                    "use_classifier=True requires obj to be a classifier "
                    "(e.g., LogisticRegression, RandomForestClassifier)"
                )

            rng = check_random_state(self.random_state)
            X, y = shuffle(X, y, random_state=rng)

            # Stratified train/calibration split
            X_train, X_calib, y_train, y_calib = train_test_split(
                X,
                y,
                test_size=self.calibration_fraction,
                random_state=self.random_state,
                stratify=y,
            )

            # Scale features
            if self.scaling_method == "standard":
                self.scaler_ = StandardScaler().fit(X_train)
            else:
                raise ValueError("Scaling method must be 'standard'")

            X_train_s = self.scaler_.transform(X_train)
            X_calib_s = self.scaler_.transform(X_calib)

            # Train ensemble of classifiers
            configs = self._sample_hyperparameters()
            self.classifiers_ = self._train_classifiers_parallel(
                X_train_s, y_train, configs
            )

            # Store calibration data for potential conformal prediction
            self.X_calib_ = X_calib_s
            self.y_calib_ = y_calib

            if self.calibrated:
                # Calibrate each classifier
                if self.verbose:
                    print("Calibrating classifiers...")

                self.calibrated_classifiers_ = []
                for clf in tqdm(
                    self.classifiers_,
                    disable=not self.verbose,
                    desc="Calibrating",
                ):
                    cal_clf = CalibratedClassifierCV(
                        clf,
                        method=self.calibration_method,
                        cv="prefit",
                        ensemble=False,
                    )
                    cal_clf.fit(X_calib_s, y_calib)
                    self.calibrated_classifiers_.append(cal_clf)

        else:
            # Original approach: use regression-based with SimpleMultitaskClassifier
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
                    cv="prefit",
                    ensemble=False,
                )

                # Fit calibration on the same data
                self.calibrated_obj_.fit(X, y)
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

        X = np.asarray(X)

        if self.use_classifier:
            # Average probabilities from ensemble
            X_s = self.scaler_.transform(X)

            if self.calibrated:
                # Use calibrated classifiers
                probas = np.array(
                    [
                        clf.predict_proba(X_s)
                        for clf in self.calibrated_classifiers_
                    ]
                )
            else:
                # Use uncalibrated classifiers
                probas = np.array(
                    [clf.predict_proba(X_s) for clf in self.classifiers_]
                )

            # Average probabilities across ensemble
            mean_proba = np.mean(probas, axis=0)

            return mean_proba
        else:
            # Use regression-based approach
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
        return self.classes_[np.argmax(proba, axis=1)]

    @property
    def _estimator_type(self):
        return "classifier"
