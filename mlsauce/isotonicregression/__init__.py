try:
    from .isotonicregressor import IsotonicRegressor
except ImportError as e:
    print(f"Could not import IsotonicRegressor: {e}")

__all__ = ["IsotonicRegressor"]
