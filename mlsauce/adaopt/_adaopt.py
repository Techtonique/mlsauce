import numpy as np
import pickle
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from numpy.linalg import norm
from tqdm import tqdm
from ..utils import subsample
from ..utils import cluster 

try:
    from . import _adaoptc as adaoptc
except ImportError:
    pass


class AdaOpt(BaseEstimator, ClassifierMixin):
    """AdaOpt classifier.

    Attributes:

        n_iterations: int
            number of iterations of the optimizer at training time.

        learning_rate: float
            controls the speed of the optimizer at training time.

        reg_lambda: float
            L2 regularization parameter for successive errors in the optimizer
            (at training time).

        reg_alpha: float
            L1 regularization parameter for successive errors in the optimizer
            (at training time).

        eta: float
            controls the slope in gradient descent (at training time).

        gamma: float
            controls the step size in gradient descent (at training time).

        k: int
            number of nearest neighbors selected at test time for classification.

        tolerance: float
            controls early stopping in gradient descent (at training time).

        n_clusters: int
            number of clusters, if MiniBatch k-means is used at test time
            (for faster prediction).

        batch_size: int
            size of the batch, if MiniBatch k-means is used at test time
            (for faster prediction).

        row_sample: float
            percentage of rows chosen from training set (by stratified subsampling,
            for faster prediction).

        type_dist: str
            distance used for finding the nearest neighbors; currently `euclidean-f`
            (euclidean distances calculated as whole), `euclidean` (euclidean distances
            calculated row by row), `cosine` (cosine distance).

        n_jobs: int
            number of cpus for parallel processing (default: None)

        verbose: int
            progress bar for parallel processing (yes = 1) or not (no = 0)

        cache: boolean
            if the nearest neighbors are cached or not, for faster retrieval in
            subsequent calls.
        
        n_clusters_input: int
            number of clusters (a priori) for clustering the features
        
        clustering_method: str
            clustering method: currently 'kmeans', 'gmm'
        
        cluster_scaling: str
            scaling method for clustering: currently 'standard', 'robust', 'minmax'    

        seed: int
            reproducibility seed for nodes_sim=='uniform', clustering and dropout.

    """

    def __init__(
        self,
        n_iterations=50,
        learning_rate=0.3,
        reg_lambda=0.1,
        reg_alpha=0.5,
        eta=0.01,
        gamma=0.01,
        k=3,
        tolerance=0,
        n_clusters=0,
        batch_size=100,
        row_sample=0.8,
        type_dist="euclidean-f",
        n_jobs=None,
        verbose=0,
        cache=True,
        n_clusters_input = 0,
        clustering_method = "kmeans",
        cluster_scaling = "standard",
        seed=123,
    ):
        if n_clusters_input > 0: 
            assert clustering_method in (
                "kmeans",
                "gmm",
            ), "`clustering_method` must be in ('kmeans', 'gmm')"
            assert cluster_scaling in (
                "standard",
                "robust",
                "minmax",
            ), "`cluster_scaling` must be in ('standard', 'robust', 'minmax')"

        assert type_dist in (
            "euclidean",
            "manhattan",
            "euclidean-f",
            "cosine",
        ), "must have: `type_dist` in ('euclidean', 'manhattan', 'euclidean-f', 'cosine') "

        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.eta = eta
        self.gamma = gamma
        self.k = k
        self.tolerance = tolerance
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.row_sample = row_sample
        self.type_dist = type_dist
        self.n_jobs = n_jobs
        self.cache = cache
        self.verbose = verbose
        self.n_clusters_input = n_clusters_input
        self.clustering_method = clustering_method
        self.cluster_scaling = cluster_scaling
        self.scaler_, self.label_encoder_, self.clusterer_ = None, None, None 
        self.seed = seed

    def fit(self, X, y, **kwargs):
        """Fit AdaOpt to training data (X, y)

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

        if self.n_clusters_input > 0: 
            clustered_X, self.scaler_, self.label_encoder_, self.clusterer_ = cluster(X, n_clusters=self.n_clusters_input, 
                method=self.clustering_method, 
                type_scaling=self.cluster_scaling,
                training=True, 
                seed=self.seed)
            X = np.column_stack((X.copy(), clustered_X))

        if self.row_sample < 1:
            index_subsample = subsample(
                y, row_sample=self.row_sample, seed=self.seed
            )
            y_ = y[index_subsample]
            X_ = X[index_subsample, :]
        else:
            y_ = pickle.loads(pickle.dumps(y, -1))
            X_ = pickle.loads(pickle.dumps(X, -1))

        n, p = X_.shape

        n_classes = len(np.unique(y_))

        assert n == len(y_), "must have X.shape[0] == len(y)"

        res = adaoptc.fit_adaopt(
            X=np.asarray(X_).astype(np.float64),
            y=np.asarray(y_).astype(np.int64),
            n_iterations=self.n_iterations,
            n_X=n,
            p_X=p,
            n_classes=n_classes,
            learning_rate=self.learning_rate,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            eta=self.eta,
            gamma=self.gamma,
            tolerance=self.tolerance,
        )

        self.probs_training = res["probs"]
        self.training_accuracy = res["training_accuracy"]
        self.alphas = res["alphas"]
        self.n_iterations = res["n_iterations"]
        self.scaled_X_train = np.array(res["scaled_X_train"], dtype=np.float64)
        self.n_classes_ = len(np.unique(y)) # for compatibility with sklearn 
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

        n_train, p_train = self.scaled_X_train.shape

        if self.n_clusters_input > 0:
            X = np.column_stack((X.copy(), cluster(
                X, training=False, 
                scaler=self.scaler_, 
                label_encoder=self.label_encoder_, 
                clusterer=self.clusterer_,
                seed=self.seed
            )))

        n_test = X.shape[0]

        if self.n_jobs is None:
            return adaoptc.predict_proba_adaopt(
                X_test=np.asarray(X, order="C").astype(np.float64),
                scaled_X_train=np.asarray(
                    self.scaled_X_train, order="C"
                ).astype(np.float64),
                n_test=n_test,
                n_train=n_train,
                probs_train=self.probs_training,
                k=self.k,
                n_clusters=self.n_clusters,
                batch_size=self.batch_size,
                type_dist=self.type_dist,
                cache=self.cache,
                seed=self.seed,
            )

        # parallel: self.n_jobs is not None
        assert self.type_dist in (
            "euclidean",
            "manhattan",
            "cosine",
        ), "must have: `self.type_dist` in ('euclidean', 'manhattan', 'cosine') "

        scaled_X_test = X / norm(X, ord=2, axis=1)[:, None]

        if self.type_dist == "euclidean":

            @delayed
            @wrap_non_picklable_objects
            def multiproc_func(i):
                dists_test_i = adaoptc.distance_to_mat_euclidean2(
                    np.asarray(scaled_X_test.astype(np.float64), order="C")[
                        i, :
                    ],
                    np.asarray(
                        self.scaled_X_train.astype(np.float64), order="C"
                    ),
                    np.zeros(n_train),
                    n_train,
                    p_train,
                )

                kmin_test_i = adaoptc.find_kmin_x(
                    dists_test_i, n_x=n_train, k=self.k, cache=self.cache
                )

                weights_test_i = adaoptc.calculate_weights(kmin_test_i[0])

                probs_test_i = adaoptc.calculate_probs(
                    kmin_test_i[1], self.probs_training
                )

                return adaoptc.average_probs(
                    probs=probs_test_i, weights=weights_test_i
                )

        if self.type_dist == "manhattan":

            @delayed
            @wrap_non_picklable_objects
            def multiproc_func(i):
                dists_test_i = adaoptc.distance_to_mat_manhattan2(
                    np.asarray(scaled_X_test.astype(np.float64), order="C")[
                        i, :
                    ],
                    np.asarray(
                        self.scaled_X_train.astype(np.float64), order="C"
                    ),
                    np.zeros(n_train),
                    n_train,
                    p_train,
                )

                kmin_test_i = adaoptc.find_kmin_x(
                    dists_test_i, n_x=n_train, k=self.k, cache=self.cache
                )

                weights_test_i = adaoptc.calculate_weights(kmin_test_i[0])

                probs_test_i = adaoptc.calculate_probs(
                    kmin_test_i[1], self.probs_training
                )

                return adaoptc.average_probs(
                    probs=probs_test_i, weights=weights_test_i
                )

        if self.type_dist == "cosine":

            @delayed
            @wrap_non_picklable_objects
            def multiproc_func(i, *args):
                dists_test_i = adaoptc.distance_to_mat_cosine2(
                    np.asarray(scaled_X_test.astype(np.float64), order="C")[
                        i, :
                    ],
                    np.asarray(
                        self.scaled_X_train.astype(np.float64), order="C"
                    ),
                    np.zeros(n_train),
                    n_train,
                    p_train,
                )

                kmin_test_i = adaoptc.find_kmin_x(
                    dists_test_i, n_x=n_train, k=self.k, cache=self.cache
                )

                weights_test_i = adaoptc.calculate_weights(kmin_test_i[0])

                probs_test_i = adaoptc.calculate_probs(
                    kmin_test_i[1], self.probs_training
                )

                return adaoptc.average_probs(
                    probs=probs_test_i, weights=weights_test_i
                )

        if self.verbose == 1:
            res = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                (multiproc_func)(m) for m in tqdm(range(n_test))
            )

        else:
            res = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                (multiproc_func)(m) for m in range(n_test)
            )

        return np.asarray(res)
