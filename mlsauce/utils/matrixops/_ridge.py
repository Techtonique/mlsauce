import numpy as np
import platform
import warnings
from numpy.linalg import pinv
from . import _matrixopsc as mo
if platform.system() in ('Linux', 'Darwin'):
    import jax.numpy as jnp
    from jax import device_put
    from jax.numpy.linalg import pinv as jpinv

class Ridge():
    """ Ridge.
        
     Parameters
     ----------
     seed: int 
         reproducibility seed for nodes_sim=='uniform', clustering and dropout.
     backend: str    
         type of backend; must be in ('cpu', 'gpu', 'tpu')          
    """

    def __init__(
        self,        
        reg_lambda=0.1,
        backend="cpu"
    ):

        sys_platform = platform.system()

        if (sys_platform == "Windows") and (backend in ("gpu", "tpu")):
            warnings.warn("No GPU/TPU computing on Windows yet, backend set to 'cpu'")
            backend = "cpu"

        self.reg_lambda = reg_lambda
        self.seed = seed
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

        assert self.backend in ("cpu", "gpu", "tpu"),\
             "`backend` must be in ('cpu', 'gpu', 'tpu')"

        
        self.ym, centered_y  = mo.center_response(y)                                   
        self.xm = np.asarray(X).mean(axis=0)
        self.xsd = np.asarray(X).std(axis=0)
        X_ = (X - self.xm[None, :])/self.xsd[None, :]
        n, p = X_.shape()

        if self.backend == "cpu":

            x = pinv(mo.crossprod(X_) + self.reg_lambda*np.diag(p)))
            z = X_.T
            hat_matrix = np.dot(x, z))
            self.beta = np.dot(hat_matrix, centered_y)

        else:    

            x = device_put(jpinv(mo.crossprod(X_, backend=self.backend) + self.reg_lambda*jnp.diag(p)))
            z = device_put(jnp.transpose(X_)) 
            hat_matrix = jnp.dot(x, z).block_until_ready()

            hat_matrix = device_put(hat_matrix)
            centered_y = device_put(centered_y)
            self.beta = jnp.dot(hat_matrix, centered_y).block_until_ready()

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
            if (len(self.ym) == 1):
                return self.ym + np.dot(X_, self.beta)    
            return self.ym[None, :] + np.dot(X_, self.beta)

        # if self.backend in ("gpu", "tpu"):
        if (len(self.ym) == 1):
            return self.ym + jnp.dot(X_, self.beta)                                
        return self.ym[None, :] + jnp.dot(X_, self.beta)