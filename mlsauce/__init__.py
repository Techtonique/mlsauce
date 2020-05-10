"""Top-level package for mlsauce."""

__author__ = """T. Moudiki"""
__email__ = "thierry.moudiki@gmail.com"
__version__ = "__version__ = '0.2.3'"


from .adaopt_cython.adaoptc import fit_adaopt, predict_proba_adaopt
from .adaopt.adaopt import AdaOpt


__all__ = ["AdaOpt", "fit_adaopt", "predict_proba_adaopt"]
