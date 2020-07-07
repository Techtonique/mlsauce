import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from . import _boosterc as boosterc 


class LSBoostRegressor(BaseEstimator, RegressorMixin):
    """ LSBoost regressor.
        
     Parameters
     ----------
     n_estimators: int
         number of boosting iterations.
     learning_rate: float
         controls the learning speed at training time.  
     n_hidden_features: int 
         number of nodes in successive hidden layers.
     reg_lambda: float
         L2 regularization parameter for successive errors in the optimizer 
         (at training time).
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
         
    """
    
    def __init__(
        self,
        n_estimators=100, 
        learning_rate=0.1, 
        n_hidden_features=5, 
        reg_lambda=0.1, 
        row_sample=1, 
        col_sample=1,
        dropout=0, 
        tolerance=1e-4, 
        direct_link=1, 
        verbose=1,
        seed=123, 
    ):
        self.n_estimators=n_estimators
        self.learning_rate=learning_rate
        self.n_hidden_features=n_hidden_features
        self.reg_lambda=reg_lambda 
        self.row_sample=row_sample 
        self.col_sample=col_sample
        self.dropout=dropout
        self.tolerance=tolerance
        self.direct_link=direct_link
        self.verbose=verbose
        self.seed=seed 
        self.obj = None


    def fit(self, X, y, **kwargs):
        """Fit Booster (regressor) to training data (X, y)
        
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
                
        self.obj = boosterc.fit_booster_regressor(X=np.asarray(X, order='C'), 
                                          y=np.asarray(y, order='C'), 
                                          n_estimators=self.n_estimators, 
                                          learning_rate=self.learning_rate, 
                                          n_hidden_features=self.n_hidden_features, 
                                          reg_lambda=self.reg_lambda, 
                                          row_sample=self.row_sample, 
                                          col_sample=self.col_sample,
                                          dropout=self.dropout, 
                                          tolerance=self.tolerance, 
                                          direct_link=self.direct_link, 
                                          verbose=self.verbose,
                                          seed=self.seed)

        return self


    def predict(self, X, **kwargs):
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
        
        

        return(boosterc.predict_booster_regressor(self.obj, 
                                                  np.asarray(X, order='C')))