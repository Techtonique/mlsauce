import numpy as np
import pickle
import platform
import warnings
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from numpy.linalg import inv
from . import _lassoc as mo
if platform.system() in ('Linux', 'Darwin'):
    import jax.numpy as jnp
    from jax import device_put
    from jax.numpy.linalg import inv as jinv


class LassoRegressor(BaseEstimator, RegressorMixin):
    """ Lasso.
        
     Parameters
     ----------
     reg_lambda: float
         regularization parameter.
     backend: str    
         type of backend; must be in ('cpu', 'gpu', 'tpu')          
    """

    def __init__(
        self,        
        reg_lambda=0.1,
        backend="cpu"
    ):

        assert backend in ("cpu", "gpu", "tpu"),\
             "`backend` must be in ('cpu', 'gpu', 'tpu')"

        sys_platform = platform.system()

        if (sys_platform == "Windows") and (backend in ("gpu", "tpu")):
            warnings.warn("No GPU/TPU computing on Windows yet, backend set to 'cpu'")
            backend = "cpu"

        self.reg_lambda = reg_lambda    
        self.backend = backend
        

    def fit(self, X, y, **kwargs):
        """Fit matrixops (classifier) to training data (X, y)
        
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

        self.ym, centered_y  = mo.center_response(y)                                   
        self.xm = X.mean(axis=0)
        self.xsd = X.std(axis=0)
        X_ = (X - self.xm[None, :])/self.xsd[None, :]  
        XX = mo.crossprod(X_, backend=self.backend)  
        Xy = mo.crossprod(X_, centered_y, backend=self.backend)     
        XX2 = 2*XX        
        Xy2 = 2*Xy 

        if self.backend == "cpu":
            invXX = inv(XX + self.reg_lambda*np.eye(X_.shape[1]))            
            beta0 = mo.safe_sparse_dot(invXX, Xy)
            self.beta = mo.get_beta(beta0 = np.asarray(beta0), 
                                    XX2 = np.asarray(XX2), 
                                    Xy2 = np.asarray(Xy2), 
                                    reg_lambda = self.reg_lambda,
                                    max_iter = 10000, 
                                    tol = 1e-5)                
            return self        

        invXX = jinv(XX + self.reg_lambda*jnp.eye(X_.shape[1]))           
        beta0 = mo.safe_sparse_dot(invXX, Xy, backend=self.backend)
        self.beta = mo.get_beta(beta0 = np.asarray(beta0), 
                                XX2 = np.asarray(XX2), 
                                Xy2 = np.asarray(Xy2), 
                                reg_lambda = self.reg_lambda,
                                max_iter = 10000, 
                                tol = 1e-5)                
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
        X_ = (X - self.xm[None, :])/self.xsd[None, :]
    
        if self.backend == "cpu":
            if (isinstance(self.ym, float)):
                return self.ym + mo.safe_sparse_dot(X_, self.beta)    
            return self.ym[None, :] + mo.safe_sparse_dot(X_, self.beta)

        # if self.backend in ("gpu", "tpu"):
        if (isinstance(self.ym, float)):
            return self.ym + mo.safe_sparse_dot(X_, self.beta, backend=self.backend)
        return self.ym[None, :] + mo.safe_sparse_dot(X_, self.beta, backend=self.backend)