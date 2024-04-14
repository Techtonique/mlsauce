try:
    from ._ridge import RidgeRegressor
except ImportError:
    pass

__all__ = ["RidgeRegressor"]
