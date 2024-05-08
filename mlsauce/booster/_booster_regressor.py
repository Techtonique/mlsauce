import numpy as np
import platform
import warnings
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.preprocessing import PolynomialFeatures

try:
    from . import _boosterc as boosterc
except ImportError:
    import _boosterc as boosterc
from ..predictioninterval import PredictionInterval
from ..utils import cluster


class LSBoostRegressor(BaseEstimator, RegressorMixin):
    """LSBoost regressor.

    Attributes:

        n_estimators: int
            number of boosting iterations.

        learning_rate: float
            controls the learning speed at training time.

        n_hidden_features: int
            number of nodes in successive hidden layers.

        reg_lambda: float
            L2 regularization parameter for successive errors in the optimizer
            (at training time).

        alpha: float
            compromise between L1 and L2 regularization (must be in [0, 1]),
            for `solver` == 'enet'

        row_sample: float
            percentage of rows chosen from the training set.

        col_sample: float
            percentage of columns chosen from the training set.

        dropout: float
            percentage of nodes dropped from the training set.

        tolerance: float
            controls early stopping in gradient descent (at training time).

        direct_link: bool
            indicates whether the original features are included (True) in model's
            fitting or not (False).

        verbose: int
            progress bar (yes = 1) or not (no = 0) (currently).

        seed: int
            reproducibility seed for nodes_sim=='uniform', clustering and dropout.

        backend: str
            type of backend; must be in ('cpu', 'gpu', 'tpu')

        solver: str
            type of 'weak' learner; currently in ('ridge', 'lasso')

        activation: str
            activation function: currently 'relu', 'relu6', 'sigmoid', 'tanh'

        type_pi: str.
            type of prediction interval; currently "kde" (default) or "bootstrap".
            Used only in `self.predict`, for `self.replications` > 0 and `self.kernel`
            in ('gaussian', 'tophat'). Default is `None`.

        replications: int.
            number of replications (if needed) for predictive simulation.
            Used only in `self.predict`, for `self.kernel` in ('gaussian',
            'tophat') and `self.type_pi = 'kde'`. Default is `None`.

        n_clusters: int
            number of clusters for clustering the features

        clustering_method: str
            clustering method: currently 'kmeans', 'gmm'

        cluster_scaling: str
            scaling method for clustering: currently 'standard', 'robust', 'minmax'

        degree: int
            degree of features interactions to include in the model

        weights_distr: str
            distribution of weights for constructing the model's hidden layer;
            either 'uniform' or 'gaussian'

    """

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        n_hidden_features=5,
        reg_lambda=0.1,
        alpha=0.5,
        row_sample=1,
        col_sample=1,
        dropout=0,
        tolerance=1e-4,
        direct_link=1,
        verbose=1,
        seed=123,
        backend="cpu",
        solver="ridge",
        activation="relu",
        type_pi=None,
        replications=None,
        kernel=None,
        n_clusters=0,
        clustering_method="kmeans",
        cluster_scaling="standard",
        degree=None,
        weights_distr="uniform",
    ):
        if n_clusters > 0:
            assert clustering_method in (
                "kmeans",
                "gmm",
            ), "`clustering_method` must be in ('kmeans', 'gmm')"
            assert cluster_scaling in (
                "standard",
                "robust",
                "minmax",
            ), "`cluster_scaling` must be in ('standard', 'robust', 'minmax')"

        assert backend in (
            "cpu",
            "gpu",
            "tpu",
        ), "`backend` must be in ('cpu', 'gpu', 'tpu')"

        assert solver in (
            "ridge",
            "lasso",
            "enet",
        ), "`solver` must be in ('ridge', 'lasso', 'enet')"

        sys_platform = platform.system()

        if (sys_platform == "Windows") and (backend in ("gpu", "tpu")):
            warnings.warn(
                "No GPU/TPU computing on Windows yet, backend set to 'cpu'"
            )
            backend = "cpu"

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.n_hidden_features = n_hidden_features
        self.reg_lambda = reg_lambda
        assert alpha >= 0 and alpha <= 1, "`alpha` must be in [0, 1]"
        self.alpha = alpha
        self.row_sample = row_sample
        self.col_sample = col_sample
        self.dropout = dropout
        self.tolerance = tolerance
        self.direct_link = direct_link
        self.verbose = verbose
        self.seed = seed
        self.backend = backend
        self.obj = None
        self.solver = solver
        self.activation = activation
        self.type_pi = type_pi
        self.replications = replications
        self.kernel = kernel
        self.n_clusters = n_clusters
        self.clustering_method = clustering_method
        self.cluster_scaling = cluster_scaling
        self.scaler_, self.label_encoder_, self.clusterer_ = None, None, None
        self.degree = degree
        self.poly_ = None
        self.weights_distr = weights_distr

    def fit(self, X, y, **kwargs):
        """Fit Booster (regressor) to training data (X, y)

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

        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.degree is not None:
            assert isinstance(self.degree, int), "`degree` must be an integer"
            self.poly_ = PolynomialFeatures(
                degree=self.degree, interaction_only=True, include_bias=False
            )
            X = self.poly_.fit_transform(X)

        if self.n_clusters > 0:
            clustered_X, self.scaler_, self.label_encoder_, self.clusterer_ = (
                cluster(
                    X,
                    n_clusters=self.n_clusters,
                    method=self.clustering_method,
                    type_scaling=self.cluster_scaling,
                    training=True,
                    seed=self.seed,
                )
            )
            X = np.column_stack((X, clustered_X))

        try:
            self.obj = boosterc.fit_booster_regressor(
                X=np.asarray(X, order="C"),
                y=np.asarray(y, order="C"),
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                n_hidden_features=self.n_hidden_features,
                reg_lambda=self.reg_lambda,
                alpha=self.alpha,
                row_sample=self.row_sample,
                col_sample=self.col_sample,
                dropout=self.dropout,
                tolerance=self.tolerance,
                direct_link=self.direct_link,
                verbose=self.verbose,
                seed=self.seed,
                backend=self.backend,
                solver=self.solver,
                activation=self.activation,
            )
        except ValueError:
            self.obj = _boosterc.fit_booster_regressor(
                X=np.asarray(X, order="C"),
                y=np.asarray(y, order="C"),
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                n_hidden_features=self.n_hidden_features,
                reg_lambda=self.reg_lambda,
                alpha=self.alpha,
                row_sample=self.row_sample,
                col_sample=self.col_sample,
                dropout=self.dropout,
                tolerance=self.tolerance,
                direct_link=self.direct_link,
                verbose=self.verbose,
                seed=self.seed,
                backend=self.backend,
                solver=self.solver,
                activation=self.activation,
            )

        self.n_estimators = self.obj["n_estimators"]

        self.X_ = X

        self.y_ = y

        return self

    def predict(self, X, level=95, method=None, **kwargs):
        """Predict probabilities for test data X.

        Args:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features.

            level: int
                Level of confidence (default = 95)

            method: str
                `None`, or 'splitconformal', 'localconformal'
                prediction (if you specify `return_pi = True`)

            **kwargs: additional parameters to be passed to
                self.cook_test_set

        Returns:

            probability estimates for test data: {array-like}
        """

        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.degree > 0:
            X = self.poly_.transform(X)

        if self.n_clusters > 0:
            X = np.column_stack(
                (
                    X,
                    cluster(
                        X,
                        training=False,
                        scaler=self.scaler_,
                        label_encoder=self.label_encoder_,
                        clusterer=self.clusterer_,
                        seed=self.seed,
                    ),
                )
            )
        if "return_pi" in kwargs:
            assert method in (
                "splitconformal",
                "localconformal",
            ), "method must be in ('splitconformal', 'localconformal')"
            self.pi = PredictionInterval(
                obj=self,
                method=method,
                level=level,
                type_pi=self.type_pi,
                replications=self.replications,
                kernel=self.kernel,
            )
            self.pi.fit(self.X_, self.y_)
            self.X_ = None
            self.y_ = None
            preds = self.pi.predict(X, return_pi=True)
            return preds

        try:
            return boosterc.predict_booster_regressor(
                self.obj, np.asarray(X, order="C")
            )
        except ValueError:
            return _boosterc.predict_booster_regressor(
                self.obj, np.asarray(X, order="C")
            )
