import numpy as np
import platform
import warnings
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from numpy.linalg import pinv
from . import _ridgec as mo
if platform.system() in ('Linux', 'Darwin'):
    import jax.numpy as jnp
    from jax import device_put
    from jax.numpy.linalg import pinv as jpinv


class RidgeRegressor(BaseEstimator, RegressorMixin):
    """ Ridge.
        
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

        if self.backend == "cpu":

            x = pinv(mo.crossprod(X_) + self.reg_lambda*np.eye(X_.shape[1]))
            hat_matrix = np.dot(x, X_.T)
            self.beta = np.dot(hat_matrix, centered_y)

            return self        

        x = device_put(jpinv(mo.crossprod(X_, backend=self.backend) + self.reg_lambda*jnp.eye(X_.shape[1])))
        hat_matrix = jnp.dot(x, device_put(jnp.transpose(X_))).block_until_ready()
        self.beta = jnp.dot(device_put(hat_matrix), device_put(centered_y)).block_until_ready()

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
                return self.ym + np.dot(X_, self.beta)    
            return self.ym[None, :] + np.dot(X_, self.beta)

        # if self.backend in ("gpu", "tpu"):
        if (isinstance(self.ym, float)):
            return self.ym + jnp.dot(device_put(X_), device_put(self.beta)).block_until_ready()
        return self.ym[None, :] + jnp.dot(device_put(X_), device_put(self.beta)).block_until_ready()