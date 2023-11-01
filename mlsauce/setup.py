# adapted from sklearn

import sys
import os

from mlsauce._build_utils import cythonize_extensions


def configuration(parent_package="", top_path=None):
    from numpy.distutils.misc_util import Configuration
    import numpy

    libraries = []
    if os.name == "posix":
        libraries.append("m")

    config = Configuration("mlsauce", parent_package, top_path)

    # submodules with build utilities
    config.add_subpackage("__check_build")
    config.add_subpackage("_build_utils")

    # submodules which do not have their own setup.py
    # we must manually add sub-submodules & tests
    # config.add_subpackage("demo")
    #config.add_subpackage("encoders")
    config.add_subpackage("utils")
    config.add_subpackage("utils/memoryuse")
    config.add_subpackage("utils/misc")
    config.add_subpackage("utils/sampling")

    # submodules which have their own setup.py
    config.add_subpackage("adaopt")
    config.add_subpackage("booster")
    config.add_subpackage("lasso")
    config.add_subpackage("ridge")
    config.add_subpackage("stump")

    # add the test directory
    config.add_subpackage("tests")

    # Skip cythonization as we do not want to include the generated
    # C/C++ files in the release tarballs as they are not necessarily
    # forward compatible with future versions of Python for instance.
    if "sdist" not in sys.argv:
        cythonize_extensions(top_path, config)

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(**configuration(top_path="").todict())
