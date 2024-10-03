#!/usr/bin/env python

import os
import platform
import sys
import subprocess
from setuptools import Extension, find_packages, setup

subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "cython"])

from Cython.Build import cythonize
import numpy

DISTNAME = 'mlsauce'
DESCRIPTION = 'Miscellaneous Statistical/Machine Learning tools'
LONG_DESCRIPTION = 'Miscellaneous Statistical/Machine Learning tools'
MAINTAINER = 'T. Moudiki'
MAINTAINER_EMAIL = 'thierry.moudiki@gmail.com'
LICENSE = 'BSD3 Clause Clear'
VERSION = '0.20.0'

# Minimum versions for dependencies
if platform.python_implementation() == 'PyPy':
    SCIPY_MIN_VERSION = '1.1.0'
    NUMPY_MIN_VERSION = '1.14.0'
else:
    SCIPY_MIN_VERSION = '0.19.0'
    NUMPY_MIN_VERSION = '1.13.0'

install_requires = [
    f"numpy >= {NUMPY_MIN_VERSION}",
    f"scipy >= {SCIPY_MIN_VERSION}",
    "joblib >= 1.2.0",
    "scikit-learn >= 0.18.0",
    "threadpoolctl >= 2.0.0",
    "pandas >= 0.25.3",
    "requests >= 2.31.0"
]

# Include JAX only on Linux and macOS
if platform.system() in ('Linux', 'Darwin'):
    install_requires += ["jax >= 0.1.72", "jaxlib >= 0.1.51"]

# Define extensions
ext_modules = [
    Extension(
        "mlsauce.adaopt._adaoptc",
        sources=["mlsauce/adaopt/_adaoptc.pyx"],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        "mlsauce.booster._boosterc",
        sources=["mlsauce/booster/_boosterc.pyx"],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        "mlsauce.lasso._lassoc",
        sources=["mlsauce/lasso/_lassoc.pyx"],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        "mlsauce.ridge._ridgec",
        sources=["mlsauce/ridge/_ridgec.pyx"],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        "mlsauce.stump._stumpc",
        sources=["mlsauce/stump/_stumpc.pyx"],
        include_dirs=[numpy.get_include()]
    ),
]

def setup_package():
    metadata = dict(
        name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        version=VERSION,
        long_description=LONG_DESCRIPTION,
        classifiers=[
            'Development Status :: 2 - Pre-Alpha',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: BSD License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
        platforms=["linux", "macosx", "windows"],
        python_requires=">=3.5",
        install_requires=install_requires,
        setup_requires=["numpy>=1.13.0"],
        packages=find_packages(),
        ext_modules=cythonize(ext_modules),
        zip_safe=False,  # the package can run out of an .egg file
        include_package_data=True,
    )
    
    setup(**metadata)

if __name__ == "__main__":
    setup_package()
