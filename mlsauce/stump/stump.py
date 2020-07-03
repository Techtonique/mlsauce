import numpy as np
import pickle
from joblib.externals.loky import set_loky_pickler
from joblib import parallel_backend
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from scipy.special import expit
from numpy.linalg import norm
from tqdm import tqdm
from ..utils import subsample
from ..stump_cython import fit_stump_classifier, predict_proba_stump_classifier


class Stump(BaseEstimator, ClassifierMixin):
    """Stump classifier.
        
     Parameters
     ----------
     bins: int
         As in numpy.histogram.        
     sample_weight: array_like
         Observations weights.                                    
    """
    
    def __init__(
        self,
        bins="auto",
        sample_weight=None
    ):

        self.bins = bins
        self.obj = None

    def fit(self, X, y, 
            sample_weight=None, **kwargs):
        """Fit Stump to training data (X, y)
        
        Parameters
        ----------
        X: {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number 
            of samples and n_features is the number of features.
        
        y: array-like, shape = [n_samples]
               Target values.
    
        **kwargs: additional parameters to be passed to self.cook_training_set.
               
        Returns
        -------
        self: object.
        """
                
        n, p = X.shape
            
        n_classes = len(np.unique(y_))

        assert n == len(y), "must have X.shape[0] == len(y)"
                
        self.obj = fit_stump_classifier(X=np.asarray(X, order='C'), 
                                        y=np.asarray(y, order='C'), 
                                        sample_weight=sample_weight, 
                                        bins=self.bins)                   
      
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

        
        return predict_proba_stump_classifier(X_test=np.asarray(X, order='C'), 
                                    scaled_X_train=np.asarray(self.scaled_X_train, order='C'),
                                    n_test=n_test, n_train=n_train,
                                    probs_train=self.probs_training,
                                    k=self.k, n_clusters=self.n_clusters,
                                    batch_size=self.batch_size, 
                                    type_dist=self.type_dist, 
                                    cache=self.cache,
                                    seed=self.seed)
          
        return np.asarray(res)
         