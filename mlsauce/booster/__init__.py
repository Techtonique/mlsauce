try: 
    from ._booster_regressor import LSBoostRegressor
    from ._booster_regressor import GenericBoostingRegressor
    from ._booster_classifier import LSBoostClassifier
    from ._booster_classifier import GenericBoostingClassifier
except ImportError as e:
    print(f"Could not import booster modules: {e}")

__all__ = [
    "LSBoostClassifier",
    "LSBoostRegressor",
    "GenericBoostingClassifier",
    "GenericBoostingRegressor",
]
