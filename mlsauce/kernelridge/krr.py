import numpy as np
import jax.numpy as jnp

from functools import partial
try:
    import jax.numpy as jnp
    from jax import device_put
    from jax.numpy.linalg import inv as jinv
except ImportError:
    pass
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler


class KRLSRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, regularization=0.1, 
                 kernel=None, backend="cpu"):  
              
        if kernel is None:
            if backend == "cpu":
                def kernel(x, y): 
                    return np.sqrt(np.sum(np.square(x - y)))        
            else:
                def kernel(x, y):
                    device_put(x)
                    device_put(y) 
                    return jnp.sqrt(jnp.sum(jnp.square(x - y)))
                
        self.backend = backend
        self.kernel = kernel
        self.regularization = regularization
        self.ym_ = None
        self.scaler_ = StandardScaler()
        self.X_ = None
        self.coef_ = None

    def fit(self, X, y):
        self.ym_ = np.mean(y)        
        centered_y = y - self.ym_
        X_ = self.scaler_.fit_transform(X)
        if self.backend == "cpu":
            K = np.array([[self.kernel(x1, x2) for x2 in X_] for x1 in X_]) + self.regularization * np.eye(X_.shape[0])
            self.coef_ = np.linalg.solve(K, centered_y)
        else: 
            device_put(X_)
            device_put(centered_y)
            K = jnp.array([[self.kernel(x1, x2) for x2 in X_] for x1 in X_]) + self.regularization * jnp.eye(X_.shape[0])
            self.coef_ = jnp.linalg.solve(K, centered_y)
        self.X_ = X_ 
        return self

    def predict(self, X):
        X_ = self.scaler_.transform(X)
        if self.backend == "cpu":
            return np.dot(np.array([[self.kernel(x1, x2) for x2 in self.X_] for x1 in X_]), self.coef_) + self.ym_
        else:
            device_put(X_)
            device_put(self.X_)
            device_put(self.coef_)
            device_put(self.ym_)
            return jnp.dot(jnp.array([[self.kernel(x1, x2) for x2 in self.X_] for x1 in X_]), self.coef_) + self.ym_