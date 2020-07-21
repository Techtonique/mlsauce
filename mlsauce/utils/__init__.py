from .sampling.rowsubsampling import subsample
from .misc.misc import merge_two_dicts, flatten, is_float, is_factor
from .matrixops._ridge import Ridge

__all__ = ["subsample", 
           "merge_two_dicts", "flatten", "is_float", "is_factor", 
           "Ridge"]
