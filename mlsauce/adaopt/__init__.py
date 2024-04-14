try:
    from ._adaopt import AdaOpt
except ImportError:
    pass

__all__ = ["AdaOpt"]
