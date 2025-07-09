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
LICENSE = 'BSD'
VERSION = '0.25.1'

install_requires = [
    "numpy",
    "Cython",
    "jax",
    "jaxlib",
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
    subprocess.run(['uv', 'pip', 'install', 'numpy'], check=True)
    subprocess.run(['uv', 'pip', 'install', 'Cython'], check=True)
except Exception:
    subprocess.run(['pip', 'install', 'numpy'])
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


if __name__ == "__main__":
    setup(
        name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        version=VERSION,
        long_description=LONG_DESCRIPTION,
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Natural Language :: English',
            "License :: OSI Approved :: BSD License",
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
        platforms=["linux", "macosx", "windows"],
        python_requires=">=3.5",
        install_requires=install_requires,
        setup_requires=["numpy", "Cython"],
        packages=find_packages(),
        ext_modules=cythonize(ext_modules, annotate=True),
        package_data={'': ['*.pxd']}
    )

    dir_path = os.path.dirname(os.path.realpath(__file__))
    folders = ['adaopt', 'booster', 'lasso', 'ridge', 'stump']
    for folder in folders: 
        filename = os.path.join(dir_path, "mlsauce", folder, 'setup2.py')                
        try: 
            subprocess.run(['python3', filename, 'build_ext', '--inplace'])  
        except Exception as e:
            print(f"Error running setup for {folder}: {e}")
            subprocess.run(['python', filename, 'build_ext', '--inplace'])  
