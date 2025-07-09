try: 
    from .predictioninterval import PredictionInterval
except ImportError as e:
    print(f"Could not import PredictionInterval: {e}")

__all__ = ["PredictionInterval"]
