try:
    from ._adaopt import AdaOpt
except ImportError as e:
    print(f"Could not import AdaOpt: {e}")

__all__ = ["AdaOpt"]
