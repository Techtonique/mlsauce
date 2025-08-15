try:
    from .fpca import GenericFunctionalForecaster
except ImportError as e:
    print(f"Could not import GenericFunctionalForecaster: {e}")

__all__ = ["GenericFunctionalForecaster"]
