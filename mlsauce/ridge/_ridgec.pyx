# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False

# Authors: T. Moudiki
#
# License: BSD 3 Clause Clear

import numpy as np
cimport numpy as np
cimport cython

import platform
from scipy import sparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
try:
    from jax import device_put
    import jax.numpy as jnp
except ImportError:
    pass 
from ..utils import safe_sparse_dot


# column bind
def cbind(x, y, backend="cpu"):
    # if len(x.shape) == 1 or len(y.shape) == 1:
    sys_platform = platform.system()
    if backend in ("gpu", "tpu") and (sys_platform in ('Linux', 'Darwin')):
        return jnp.column_stack((x, y))
    return np.column_stack((x, y))


# center... response
def center_response(y):
    if (len(np.asarray(y).shape)==1):
        y_mean = np.mean(y)
        return y_mean, (y - y_mean)
    y_mean = np.asarray(y).mean(axis=0)
    return y_mean, (y - y_mean[None, :])


# computes t(x)%*%y
def crossprod(x, y=None, backend="cpu"):
    # assert on dimensions
    sys_platform = platform.system()
    if backend in ("gpu", "tpu") and (sys_platform in ('Linux', 'Darwin')):
        x = device_put(x)
        if y is None:
            return x.T @ x
        y = device_put(y)
        return x.T @ y
    if y is None:
        return x.T @ x
    return x.T @ y


# Obtain this for JAX
# Obtain this for JAX
# Obtain this for JAX
# Obtain this for JAX
# Obtain this for JAX
# dropout
def dropout(x, drop_prob=0, seed=123):

    assert 0 <= drop_prob <= 1

    n, p = x.shape

    if drop_prob == 0:
        return x

    if drop_prob == 1:
        return np.zeros_like(x)

    np.random.seed(seed)
    dropped_indices = np.random.rand(n, p) > drop_prob

    return dropped_indices * x / (1 - drop_prob)


# one-hot encoding
def one_hot_encode(x_clusters, n_clusters):

    assert (
        max(x_clusters) <= n_clusters
    ), "you must have max(x_clusters) <= n_clusters"

    n_obs = len(x_clusters)
    res = np.zeros((n_obs, n_clusters))

    for i in range(n_obs):
        res[i, x_clusters[i]] = 1

    return res


# one-hot encoding
def one_hot_encode2(y, n_classes):

    n_obs = len(y)
    res = np.zeros((n_obs, n_classes))

    for i in range(n_obs):
        res[i, y[i]] = 1

    return res


# row bind
def rbind(x, y, backend="cpu"):
    # if len(x.shape) == 1 or len(y.shape) == 1:
    sys_platform = platform.system()
    if backend in ("gpu", "tpu") and (sys_platform in ('Linux', 'Darwin')):
        return jnp.row_stack((x, y))
    return np.row_stack((x, y))


# adapted from sklearn.utils.exmath
def safe_sparse_dot(a, b, backend="cpu", dense_output=False):
    """Dot product that handle the sparse matrix case correctly

    Parameters
    ----------
    a : array or sparse matrix
    b : array or sparse matrix
    dense_output : boolean, (default=False)
        When False, ``a`` and ``b`` both being sparse will yield sparse output.
        When True, output will always be a dense array.

    Returns
    -------
    dot_product : array or sparse matrix
        sparse if ``a`` and ``b`` are sparse and ``dense_output=False``.
    """
    sys_platform = platform.system()

    if backend in ("gpu", "tpu") and (sys_platform in ('Linux', 'Darwin')):
        # modif when jax.scipy.sparse available
        return safe_sparse_dot(device_put(a), device_put(b)).block_until_ready()

    #    if backend == "cpu":
    if a.ndim > 2 or b.ndim > 2:
        if sparse.issparse(a):
            # sparse is always 2D. Implies b is 3D+
            # [i, j] @ [k, ..., l, m, n] -> [i, k, ..., l, n]
            b_ = np.rollaxis(b, -2)
            b_2d = b_.reshape((b.shape[-2], -1))
            ret = a @ b_2d
            ret = ret.reshape(a.shape[0], *b_.shape[1:])
        elif sparse.issparse(b):
            # sparse is always 2D. Implies a is 3D+
            # [k, ..., l, m] @ [i, j] -> [k, ..., l, j]
            a_2d = a.reshape(-1, a.shape[-1])
            ret = a_2d @ b
            ret = ret.reshape(*a.shape[:-1], b.shape[1])
        else:
            ret = safe_sparse_dot(a, b)
    else:
        ret = a @ b

    if (
        sparse.issparse(a)
        and sparse.issparse(b)
        and dense_output
        and hasattr(ret, "toarray")
    ):
        return ret.toarray()

    return ret

# Obtain this for JAX
# Obtain this for JAX
# Obtain this for JAX
# Obtain this for JAX
# Obtain this for JAX
# scale... covariates
def scale_covariates(X, choice="std", training=True, scaler=None):

    scaling_options = {
        "std": StandardScaler(copy=True, with_mean=True, with_std=True),
        "minmax": MinMaxScaler(),
    }

    if training == True:
        # scaler must be not None
        scaler = scaling_options[choice]
        scaled_X = scaler.fit_transform(X)
        return scaler, scaled_X

    # training == False:
    # scaler must be not None
    return scaler.transform(X)


# from sklearn.utils.exmath
def squared_norm(x, backend="cpu"):
    """Squared Euclidean or Frobenius norm of x.

    Faster than norm(x) ** 2.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    float
        The Euclidean norm when x is a vector, the Frobenius norm when x
        is a matrix (2-d array).
    """
    sys_platform = platform.system()
    if backend in ("gpu", "tpu") and (sys_platform in ('Linux', 'Darwin')):
        x = np.ravel(x, order="K")
        x = device_put(x)
        return safe_sparse_dot(x, x).block_until_ready()

    x = np.ravel(x, order="K")
    return safe_sparse_dot(x, x)


# computes x%*%t(y)
def tcrossprod(x, y=None, backend="cpu"):
    # assert on dimensions
    sys_platform = platform.system()
    if backend in ("gpu", "tpu") and (sys_platform in ('Linux', 'Darwin')):
        x = device_put(x)
        if y is None:
            return safe_sparse_dot(x, x.T).block_until_ready()
        y = device_put(y)
        return safe_sparse_dot(x, y.T).block_until_ready()
    if y is None:
        return safe_sparse_dot(x, x.transpose())
    return safe_sparse_dot(x, y.transpose())


# convert vector to numpy array
def to_np_array(X):
    return np.array(X.copy(), ndmin=2)
