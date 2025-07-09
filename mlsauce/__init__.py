try: 
    from .adaopt import AdaOpt
except ImportError as e:
    print(f"Could not import some modules: {e}")

try:
    from .booster import (
        LSBoostClassifier,
        LSBoostRegressor,
        GenericBoostingClassifier,
        GenericBoostingRegressor,
    )
except ImportError as e:
    print(f"Could not import some modules: {e}")

try:
    from .lazybooster import LazyBoostingClassifier, LazyBoostingRegressor, LazyBoostingMTS
except ImportError as e:
    print(f"Could not import some modules: {e}")

try:
    from .multitaskregressor import MultiTaskRegressor
except ImportError as e:
    print(f"Could not import some modules: {e}")

try:        
    from .datasets import download
except ImportError as e:
    print(f"Could not import some modules: {e}")

try:
    from .elasticnet import ElasticNetRegressor
except ImportError as e:
    print(f"Could not import some modules: {e}")

try: 
    from .kernelridge import KRLSRegressor
except ImportError as e:
    print(f"Could not import some modules: {e}")

try:
    from .lasso import LassoRegressor
except ImportError as e:
    print(f"Could not import some modules: {e}")

try:
    from .ridge import RidgeRegressor
except ImportError as e:
    print(f"Could not import some modules: {e}")

try:
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

