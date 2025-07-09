try:
    from ._stump_classifier import StumpClassifier
except ImportError as e: 
    print(f"Could not import StumpClassifier: {e}")

__all__ = ["StumpClassifier"]
