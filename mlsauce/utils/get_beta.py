import numpy as np
try:
    import jax.numpy as jnp
    from jax import device_put
    from jax.numpy.linalg import inv as jinv
except ImportError:
    pass

def get_beta(X, y, backend="cpu"):
    if backend == "cpu":
        return np.linalg.solve(X.T @ X, X.T @ y)
    device_put(X)
    device_put(y)    
    return jnp.linalg.solve(X.T @ X, X.T @ y)
