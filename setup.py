#! /usr/bin/env python

import os
import sys
import subprocess
from pathlib import Path
from setuptools import Extension, find_packages, setup
from Cython.Build import cythonize

try:
    import builtins
except ImportError:
    import __builtin__ as builtins

builtins.__MLSAUCE_SETUP__ = True

DISTNAME = 'mlsauce'
DESCRIPTION = 'Miscellaneous Statistical/Machine Learning tools'
LONG_DESCRIPTION = 'Miscellaneous Statistical/Machine Learning tools'
MAINTAINER = 'T. Moudiki'
MAINTAINER_EMAIL = 'thierry.moudiki@gmail.com'
LICENSE = 'BSD-3-Clause'
VERSION = '0.25.0'

install_requires = [
    "numpy<2.0.0",
    "Cython",
    "joblib",
    "matplotlib",
    "nnetsauce",
    "pandas",
    "requests",
    "scikit-learn",
    "scipy",
    "tqdm"
]

# Ensure Cython and NumPy are installed
try:
    subprocess.run(['uv', 'pip', 'install', 'numpy<2.0.0'], check=True)
    subprocess.run(['uv', 'pip', 'install', 'Cython'], check=True)
except Exception:
    subprocess.run(['pip', 'install', 'numpy<2.0.0'])
    subprocess.run(['pip', 'install', 'Cython'])

import numpy

script_dir = Path(__file__).parent.resolve()

ext_modules = [
    Extension("mlsauce.adaopt._adaoptc",
              [str(script_dir / "mlsauce/adaopt/_adaoptc.pyx")],
              include_dirs=[numpy.get_include()]),

    Extension("mlsauce.booster._boosterc",
              [str(script_dir / "mlsauce/booster/_boosterc.pyx")],
              include_dirs=[numpy.get_include()]),

    Extension("mlsauce.lasso._lassoc",
              [str(script_dir / "mlsauce/lasso/_lassoc.pyx")],
              include_dirs=[numpy.get_include()]),

    Extension("mlsauce.ridge._ridgec",
              [str(script_dir / "mlsauce/ridge/_ridgec.pyx")],
              include_dirs=[numpy.get_include()]),

    Extension("mlsauce.stump._stumpc",
              [str(script_dir / "mlsauce/stump/_stumpc.pyx")],
              include_dirs=[numpy.get_include()])
]

setup(
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
        'License :: OSI Approved :: BSD-3-Clause',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    platforms=["linux", "macosx", "windows"],
    python_requires=">=3.5",
    install_requires=install_requires,
    setup_requires=["numpy<2.0.0", "Cython"],
    packages=find_packages(),
    ext_modules=cythonize(ext_modules, annotate=True),
    package_data={'': ['*.pxd']}
)

