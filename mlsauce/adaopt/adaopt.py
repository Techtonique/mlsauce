import numpy as np
import pickle
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from ..utils import subsample
from ..adaopt_cython import fit_adaopt, predict_proba_adaopt



class AdaOpt(BaseEstimator, ClassifierMixin):
    """AdaOpt classifier
        
       Attributes
       ----------
       n_iterations: int
           number of iterations of the optimizer at training time
       learning_rate: float
           controls the speed of the optimizer at training time  
       reg_lambda: float
           L2 regularization parameter for successive errors in the optimizer 
           (at training time)
       reg_alpha: float
           L1 regularization parameter for successive errors in the optimizer 
           (at training time)
       eta: float
           controls the slope in gradient descent (at training time)
       gamma: float
           controls the step size in gradient descent (at training time)
       k: int
           number of nearest neighbors selected at test time for classification
       tolerance: float
           controls early stopping in gradient descent (at training time)  
       n_clusters: int
            number of clusters, if MiniBatch k-means is used at test time 
            (for faster prediction)
       batch_size: int
            size of the batch, if MiniBatch k-means is used at test time 
            (for faster prediction)
       row_sample: float
           percentage of rows chosen from training set (by stratified subsampling, 
           for faster prediction)
       type_dist: str
           distance used for finding the nearest neighbors; currently `euclidean-f`
           (euclidean distances calculated as whole), `euclidean` (euclidean distances 
           calculated row by row), `cosine` (cosine distance)
       cache: boolean
           if the nearest neighbors are cached or not, for faster retrieval in 
           subsequent calls 
       seed: int 
           reproducibility seed for nodes_sim=='uniform', clustering and dropout
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
        cache=True,
        seed=123,
    ):

        assert type_dist in (
            "euclidean",
            "euclidean-f",
            "cosine",
        ), "must have: `type_dist` in ('euclidean', 'euclidean-f', 'cosine') "

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
        self.cache = cache
        self.seed = seed


    def fit(self, X, y, **kwargs):
        """Fit AdaOpt to training data (X, y)
        
        Parameters
        ----------
        X: {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number 
            of samples and n_features is the number of features
        
        y: array-like, shape = [n_samples]
               Target values
    
        **kwargs: additional parameters to be passed to self.cook_training_set
               
        Returns
        -------
        self: object
        """
        
        if self.row_sample < 1:
            index_subsample = subsample(y, row_sample=self.row_sample, 
                                       seed=self.seed)
            y_ = y[index_subsample]
            X_ = X[index_subsample, :]
        else:
            y_ = pickle.loads(pickle.dumps(y, -1))
            X_ = pickle.loads(pickle.dumps(X, -1))
        
        n, p = X_.shape
            
        n_classes = len(np.unique(y_))

        assert n == len(y_), "must have X.shape[0] == len(y)"

        res = fit_adaopt(
            X=X_,
            y=y_,
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
        self.scaled_X_train = res["scaled_X_train"]

        return self


    def predict(self, X, **kwargs):
        """Predict test data X.
        
        Parameters
        ----------
        X: {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number 
            of samples and n_features is the number of features.
        
        **kwargs: additional parameters to be passed to `predict_proba`
                
               
        Returns
        -------
        model predictions: {array-like}
        """

        return np.argmax(self.predict_proba(X, **kwargs), 
                         axis=1)


    def predict_proba(self, X, **kwargs):
        """Predict probabilities for test data X.
        
        Parameters
        ----------
        X: {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number 
            of samples and n_features is the number of features.
        
        **kwargs: additional parameters to be passed to 
                  self.cook_test_set
               
        Returns
        -------
        probability estimates for test data: {array-like}        
        """

        n_train = self.scaled_X_train.shape[0]

        n_test = X.shape[0]
        
        return predict_proba_adaopt(X_test=X, 
                                scaled_X_train=self.scaled_X_train,
                                n_test=n_test, n_train=n_train,
                                probs_train=self.probs_training,
                                k=self.k, n_clusters=self.n_clusters,
                                batch_size=self.batch_size, 
                                type_dist=self.type_dist, 
                                cache=self.cache,
                                seed=self.seed)
