"""Top-level package for mlsauce."""

__author__ = """T. Moudiki"""
__email__ = "thierry.moudiki@gmail.com"
__version__ = "__version__ = '0.3.0'"


from .adaopt_cython.adaoptc import fit_adaopt, predict_proba_adaopt
from .adaopt.adaopt import AdaOpt
from .booster_cython.boosterc import fit_booster_classifier, predict_proba_booster_classifier
from .booster.booster_classifier import LSBoostClassifier
from .stump_cython.stumpc import fit_stump_classifier, predict_proba_stump_classifier
from .stump.stump_classifier import StumpClassifier

__all__ = ["AdaOpt", "fit_adaopt", "predict_proba_adaopt", 
           "LSBoostClassifier", "fit_booster_classifier", "predict_proba_booster_classifier", 
           "StumpClassifier", "fit_stump_classifier", "predict_proba_stump_classifier", ]
