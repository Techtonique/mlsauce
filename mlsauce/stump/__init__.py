try:
    from ._stump_classifier import StumpClassifier
except ImportError:
    pass

__all__ = ["StumpClassifier"]
