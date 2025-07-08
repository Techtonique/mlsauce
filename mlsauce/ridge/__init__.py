try:
    from ._ridge import RidgeRegressor
except ImportError as e:
    print(f"Could not import RidgeRegressor: {e}")

__all__ = ["RidgeRegressor"]
