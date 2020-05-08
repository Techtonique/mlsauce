"""Top-level package for mlsauce."""

__author__ = """T. Moudiki"""
__email__ = "thierry.moudiki@gmail.com"
__version__ = "__version__ = '0.2.2'"

from .adaopt.adaopt import AdaOpt
from .adaopt.adaoptc import fit_adaopt, predict_proba_adaopt

__all__ = ["AdaOpt", 
           "fit_adaopt", "predict_proba_adaopt"]
