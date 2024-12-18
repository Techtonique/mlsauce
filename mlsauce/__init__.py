import sys
import logging
import os

from ._config import get_config, set_config, config_context

logger = logging.getLogger(__name__)


# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
#
# __version__ = "0.10.0"


# On OSX, we can get a runtime error due to multiple OpenMP libraries loaded
# simultaneously. This can happen for instance when calling BLAS inside a
# prange. Setting the following environment variable allows multiple OpenMP
# libraries to be loaded. It should not degrade performances since we manually
# take care of potential over-subcription performance issues, in sections of
# the code where nested OpenMP loops can happen, by dynamically reconfiguring
# the inner OpenMP runtime to temporarily disable it while under the scope of
# the outer OpenMP parallel section.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

# Workaround issue discovered in intel-openmp 2019.5:
# https://github.com/ContinuumIO/anaconda-issues/issues/11294
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

try:
    # This variable is injected in the __builtins__ by the build
    # process. It is used to enable importing subpackages of mlsauce when
    # the binaries are not built
    # mypy error: Cannot determine type of '__MLSAUCE_SETUP__'
    __MLSAUCE_SETUP__  # type: ignore
except NameError:
    __MLSAUCE_SETUP__ = False

if __MLSAUCE_SETUP__:
    sys.stderr.write("Partial import of mlsauce during the build process.\n")
    # We are not importing the rest of scikit-learn during the build
    # process, as it may not be compiled yet

else:
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


def setup_module(module):
    """Fixture for the tests to assure globally controllable seeding of RNGs"""
    import os
    import numpy as np
    import random

    # Check if a random seed exists in the environment, if not create one.
    _random_seed = os.environ.get("MLSAUCE_SEED", None)
    if _random_seed is None:
        _random_seed = np.random.uniform() * np.iinfo(np.int32).max
    _random_seed = int(_random_seed)
    print("I: Seeding RNGs with %r" % _random_seed)
    np.random.seed(_random_seed)
    random.seed(_random_seed)
