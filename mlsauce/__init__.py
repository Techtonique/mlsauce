try: 
    from .adaopt import AdaOpt
    from .booster import (
        LSBoostClassifier,
        LSBoostRegressor,
        GenericBoostingClassifier,
        GenericBoostingRegressor,
    )
    from .lazybooster import LazyBoostingClassifier, LazyBoostingRegressor, LazyBoostingMTS
    from .multitaskregressor import MultiTaskRegressor
    from .datasets import download
    from .elasticnet import ElasticNetRegressor
    from .kernelridge import KRLSRegressor
    from .lasso import LassoRegressor
    from .ridge import RidgeRegressor
    from .stump import StumpClassifier
except ImportError as e:
    print(f"Could not import some modules: {e}")

# from .encoders import corrtarget_encoder

__all__ = [
    "AdaOpt",
    "LSBoostClassifier",
    "GenericBoostingClassifier",
    "GenericBoostingRegressor",
    "StumpClassifier",
    "ElasticNetRegressor",
    "KRLSRegressor",
    "LassoRegressor",
    "LSBoostRegressor",
    "RidgeRegressor",
    "LazyBoostingClassifier",
    "LazyBoostingMTS",
    "LazyBoostingRegressor",
    "MultiTaskRegressor",
    # Other imports
    # "corrtarget_encoder",
    "download",
    # Non-modules:
    "get_config",
    "set_config",
    "config_context",
]

