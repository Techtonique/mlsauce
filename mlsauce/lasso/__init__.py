try:
    from ._lasso import LassoRegressor
except ImportError as e:
    print(f"Could not import LassoRegressor: {e}")

__all__ = ["LassoRegressor"]
