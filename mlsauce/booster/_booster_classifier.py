import numpy as np
import platform
import warnings
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.tree import ExtraTreeRegressor
from sklearn.preprocessing import PolynomialFeatures

try:
    from . import _boosterc as boosterc
except ImportError:
    import _boosterc as boosterc
from ..utils import cluster, check_and_install, get_histo_features


class LSBoostClassifier(BaseEstimator, ClassifierMixin):
    """LSBoost classifier.

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
            for `solver` == 'enet'.

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
            type of 'weak' learner; currently in ('ridge', 'lasso', 'enet').
            'enet' is a combination of 'ridge' and 'lasso' called Elastic Net.

        activation: str
            activation function: currently 'relu', 'relu6', 'sigmoid', 'tanh'

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
            currently 'uniform', 'gaussian'
        
        hist: bool
            indicates whether histogram features are used or not (default is False)
        
        bins: int or str
            number of bins for histogram features (same as numpy.histogram, default is 'auto')

    Examples:

        ```python
        import numpy as np
        from sklearn.datasets import load_digits, load_breast_cancer, load_wine, load_iris
        from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.kernel_ridge import KernelRidge
        from time import time
        from os import chdir
        from sklearn import metrics
        import os

        import mlsauce as ms

        print("\n")
        print("GenericBoosting Decision tree -----")
        print("\n")

        print("\n")
        print("breast_cancer data -----")

        # data 1
        breast_cancer = load_breast_cancer()
        X = breast_cancer.data
        y = breast_cancer.target
        # split data into training test and test set
        np.random.seed(15029)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2)

        clf = DecisionTreeRegressor()
        clf2 = KernelRidge()

        obj = ms.GenericBoostingClassifier(clf, tolerance=1e-2)
        print(obj.get_params())
        start = time()
        obj.fit(X_train, y_train)
        print(time()-start)
        start = time()
        print(obj.score(X_test, y_test))
        print(time()-start)

        print(obj.obj['loss'])

        obj = ms.GenericBoostingClassifier(clf, tolerance=1e-2, n_clusters=2)
        print(obj.get_params())
        start = time()
        obj.fit(X_train, y_train)
        print(time()-start)
        start = time()
        print(obj.score(X_test, y_test))
        print(time()-start)

        print(obj.obj['loss'])


        # data 2
        print("\n")
        print("wine data -----")

        wine = load_wine()
        Z = wine.data
        t = wine.target
        np.random.seed(879423)
        X_train, X_test, y_train, y_test = train_test_split(Z, t,
                                                            test_size=0.2)

        obj = ms.GenericBoostingClassifier(clf)
        print(obj.get_params())
        start = time()
        obj.fit(X_train, y_train)
        print(time()-start)
        start = time()
        print(obj.score(X_test, y_test))
        print(time()-start)

        print(obj.obj['loss'])

        obj = ms.GenericBoostingClassifier(clf, n_clusters=3)
        print(obj.get_params())
        start = time()
        obj.fit(X_train, y_train)
        print(time()-start)
        start = time()
        print(obj.score(X_test, y_test))
        print(time()-start)

        print(obj.obj['loss'])

        # data 3
        print("\n")
        print("iris data -----")

        iris = load_iris()
        Z = iris.data
        t = iris.target
        np.random.seed(734563)
        X_train, X_test, y_train, y_test = train_test_split(Z, t,
                                                            test_size=0.2)


        obj = ms.GenericBoostingClassifier(clf)
        print(obj.get_params())
        start = time()
        obj.fit(X_train, y_train)
        print(time()-start)
        start = time()
        print(obj.score(X_test, y_test))
        print(time()-start)

        print(obj.obj['loss'])


        print("\n")
        print("GenericBoosting  KRR -----")
        print("\n")

        obj = ms.GenericBoostingClassifier(clf2, tolerance=1e-2)
        print(obj.get_params())
        start = time()
        obj.fit(X_train, y_train)
        print(time()-start)
        start = time()
        print(obj.score(X_test, y_test))
        print(time()-start)

        print(obj.obj['loss'])

        obj = ms.GenericBoostingClassifier(clf2, tolerance=1e-2, n_clusters=2)
        print(obj.get_params())
        start = time()
        obj.fit(X_train, y_train)
        print(time()-start)
        start = time()
        print(obj.score(X_test, y_test))
        print(time()-start)

        print(obj.obj['loss'])


        # data 2
        print("\n")
        print("wine data -----")

        wine = load_wine()
        Z = wine.data
        t = wine.target
        np.random.seed(879423)
        X_train, X_test, y_train, y_test = train_test_split(Z, t,
                                                            test_size=0.2)

        obj = ms.GenericBoostingClassifier(clf2)
        print(obj.get_params())
        start = time()
        obj.fit(X_train, y_train)
        print(time()-start)
        start = time()
        print(obj.score(X_test, y_test))
        print(time()-start)

        print(obj.obj['loss'])

        obj = ms.GenericBoostingClassifier(clf2, n_clusters=3)
        print(obj.get_params())
        start = time()
        obj.fit(X_train, y_train)
        print(time()-start)
        start = time()
        print(obj.score(X_test, y_test))
        print(time()-start)

        print(obj.obj['loss'])

        # data 3
        print("\n")
        print("iris data -----")

        iris = load_iris()
        Z = iris.data
        t = iris.target
        np.random.seed(734563)
        X_train, X_test, y_train, y_test = train_test_split(Z, t,
                                                            test_size=0.2)


        obj = ms.GenericBoostingClassifier(clf2)
        print(obj.get_params())
        start = time()
        obj.fit(X_train, y_train)
        print(time()-start)
        start = time()
        print(obj.score(X_test, y_test))
        print(time()-start)

        print(obj.obj['loss'])
    ```

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
        n_clusters=0,
        clustering_method="kmeans",
        cluster_scaling="standard",
        degree=None,
        weights_distr="uniform",
        base_model=None,
        hist=False,
        bins="auto",
    ):

        self.base_model = base_model
        self.hist = hist
        self.bins = bins
        self.hist_bins_ = None

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
        self.n_clusters = n_clusters
        self.clustering_method = clustering_method
        self.cluster_scaling = cluster_scaling
        self.scaler_, self.label_encoder_, self.clusterer_ = None, None, None
        self.degree = degree
        self.poly_ = None
        self.weights_distr = weights_distr
        if self.backend in ("gpu", "tpu"):
            check_and_install("jax")
            check_and_install("jaxlib")

    def fit(self, X, y, **kwargs):
        """Fit Booster (classifier) to training data (X, y)

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
        
        if self.hist == True:
            X, self.hist_bins_ = get_histo_features(X)
        
        if isinstance(y, pd.Series):
            y = y.values.ravel()
        else:
            y = y.ravel()

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

        self.obj = boosterc.fit_booster_classifier(
            np.asarray(X, order="C", dtype=np.float64),
            np.asarray(y, order="C", dtype=np.int64),
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
            obj=self.base_model,
        )

        self.classes_ = np.unique(y)  # for compatibility with sklearn
        self.n_classes_ = len(self.classes_)  # for compatibility with sklearn
        self.n_estimators = self.obj["n_estimators"]
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

        return np.argmax(self.predict_proba(X, **kwargs), axis=1)

    def predict_proba(self, X, **kwargs):
        """Predict probabilities for test data X.

        Args:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features.

            **kwargs: additional parameters to be passed to
                self.cook_test_set

        Returns:

            probability estimates for test data: {array-like}
        """

        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.hist == True:
            X = get_histo_features(X, bins=self.hist_bins_)

        if self.degree is not None:
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
        try:
            return boosterc.predict_proba_booster_classifier(
                self.obj, np.asarray(X, order="C")
            )
        except ValueError:
            pass

    def update(self, X, y, eta=0.9):
        """Update model with new data.

        Args:

            X: {array-like}, shape = [n_samples=1, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features.

            y: float = [n_samples=1]
               Target value.

            eta: float
                Inverse power applied to number of observations
                (defines a learning rate).

        Returns:

            self: object.
        """

        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.degree is not None:
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

        self.obj = boosterc.update_booster(
            self.obj, np.asarray(X, order="C"), 
            np.asarray(y, order="C").ravel(), eta
        )

        return self


class GenericBoostingClassifier(LSBoostClassifier):
    """Generic Boosting classifier (using any classifier as base learner).

    Attributes:

        base_model: object
            base learner (default is ExtraTreeRegressor) to be boosted.

        n_estimators: int
            number of boosting iterations.

        learning_rate: float
            controls the learning speed at training time.

        n_hidden_features: int
            number of nodes in successive hidden layers.

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

        activation: str
            activation function: currently 'relu', 'relu6', 'sigmoid', 'tanh'

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
            currently 'uniform', 'gaussian'
        
        hist: bool
            indicates whether histogram features are used or not (default is False)
        
        bins: int or str
            number of bins for histogram features (same as numpy.histogram, default is 'auto')

    """

    def __init__(
        self,
        base_model=ExtraTreeRegressor(),
        n_estimators=100,
        learning_rate=0.1,
        n_hidden_features=5,
        row_sample=1,
        col_sample=1,
        dropout=0,
        tolerance=1e-4,
        direct_link=1,
        verbose=1,
        backend="cpu",
        seed=123,
        activation="relu",
        n_clusters=0,
        clustering_method="kmeans",
        cluster_scaling="standard",
        degree=None,
        weights_distr="uniform",
        hist=False,
        bins="auto",
    ):
        self.base_model = base_model
        self.hist = hist
        self.bins = bins
        self.hist_bins_ = None

        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            n_hidden_features=n_hidden_features,
            row_sample=row_sample,
            col_sample=col_sample,
            dropout=dropout,
            tolerance=tolerance,
            direct_link=direct_link,
            verbose=verbose,
            backend=backend,
            seed=seed,
            activation=activation,
            n_clusters=n_clusters,
            clustering_method=clustering_method,
            cluster_scaling=cluster_scaling,
            degree=degree,
            weights_distr=weights_distr,
            base_model=self.base_model,
        )