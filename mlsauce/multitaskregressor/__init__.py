try: 
    from .mtaskregr import MultiTaskRegressor
except ImportError as e:
    print(f"Could not import MultiTaskRegressor: {e}")

__all__ = ["MultiTaskRegressor"]
