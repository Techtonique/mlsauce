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
    from .lazybooster import (
        LazyBoostingClassifier,
        LazyBoostingRegressor,
        LazyBoostingMTS,
    )
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

try:
    from .isotonicregression import IsotonicRegressor
except ImportError as e:
    print(f"Could not import some modules: {e}")

try:
    from .fpca import GenericFunctionalForecaster
except ImportError as e:
    print(f"Could not import some modules: {e}")

try: 
    from .generators import make_diverse_classification,\
    HealthcareTimeSeriesGenerator,\
    generate_synthetic_returns,\
    plot_synthetic_returns
except ImportError as e:
    print(f"Could not import generators: {e}")

try:
    from .catencoder import RankTargetEncoder
except ImportError as e:
    print(f"Could not import RankTargetEncoder: {e}")

try:
    from .rollingoriginregression import RollingOriginForecaster
except ImportError as e:
    print(f"Could not import RollingOriginForecaster: {e}")
# from .encoders import corrtarget_encoder

try:
    from .penalizedcv import penalized_cross_val_score
except ImportError as e:
    print(f"Could not import penalized_cross_val_score: {e}")

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
    "LSTMRegressor",
    "RidgeRegressor",
    "LazyBoostingClassifier",
    "LazyBoostingMTS",
    "LazyBoostingRegressor",
    "MultiTaskRegressor",
    "IsotonicRegressor",
    "GenericFunctionalForecaster",
    "RankTargetEncoder",
    "RollingOriginForecaster",
    # Other imports
    # "corrtarget_encoder",
    "download",
    # Non-modules:
    "get_config",
    "set_config",
    "config_context",
    "penalized_cross_val_score",
    "make_diverse_classification",
    "HealthcareTimeSeriesGenerator",
    "generate_synthetic_returns",
    "plot_synthetic_returns"
]
