import numpy as np
from ..utils import adaopt as ao
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin


class AdaOpt(BaseEstimator, ClassifierMixin):
    
    def __init__(
        self,
        n_iterations=50,
        learning_rate=0.3,
        reg_lambda=0.1,
        reg_alpha=0.5,
        eta=0.01,
        gamma=0.01,
        k=3,
        tolerance=1e-3,
        type_dist="euclidean",
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
        self.type_dist = type_dist
        self.seed = seed


    def fit(self, X, y, **kwargs):

        n, p = X.shape
        n_classes = len(np.unique(y))

        assert n == len(y), "must have X.shape[0] == len(y)"

        res = ao.fit(
            X=X,
            y=y,
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


    def predict(self, X, type_dist="euclidean", **kwargs):
        """Predict test data X.
        
        Parameters
        ----------
        X: {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number 
            of samples and n_features is the number of features.
        
        **kwargs: additional parameters to be passed to 
                  self.cook_test_set
               
        Returns
        -------
        model predictions: {array-like}
        """

        return np.argmax(
            self.predict_proba(X, type_dist=self.type_dist, 
                               **kwargs), axis=1
        )


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

        return ao.predict_proba(
            X_test=X,
            scaled_X_train=self.scaled_X_train,
            n_test=n_test,
            n_train=n_train,
            probs_train=self.probs_training,
            k=self.k,
            type_dist=self.type_dist,
        )
