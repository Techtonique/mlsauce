try: 
    from .lazyboosterclassif import LazyBoostingClassifier
    from .lazyboosterregression import LazyBoostingRegressor
    from .lazyboostermts import LazyBoostingMTS
except ImportError as e:
    print(f"Could not import lazybooster modules: {e}")

__all__ = ["LazyBoostingClassifier", "LazyBoostingRegressor", "LazyBoostingMTS"]
