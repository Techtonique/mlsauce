try:
    from ._lasso import LassoRegressor
except ImportError:
    pass

__all__ = ["LassoRegressor"]
