#! /usr/bin/env python
#
# Copyright (C) 2020-2024 T. Moudiki <thierry.moudiki@gmail.com>
# License: 3-clause BSD

import os
import platform
import setuptools 
import shutil
import subprocess
import sys


from os import path
from pathlib import Path
from setuptools import Extension, find_packages, setup

try:
    import builtins
except ImportError:    
    import __builtin__ as builtins

subprocess.run(['pip', 'install', 'numpy'], check=False)
subprocess.run(['pip', 'install', 'scipy'], check=False)
subprocess.run(['pip', 'install', 'Cython'], check=False)
subprocess.run(['pip', 'install', 'requests'], check=False)

import numpy
from Cython.Build import cythonize

builtins.__MLSAUCE_SETUP__ = True

DISTNAME = 'mlsauce'
DESCRIPTION = 'Miscellaneous Statistical/Machine Learning tools'
LONG_DESCRIPTION = 'Miscellaneous Statistical/Machine Learning tools'
MAINTAINER = 'T. Moudiki'
MAINTAINER_EMAIL = 'thierry.moudiki@gmail.com'
LICENSE = 'BSD3 Clause Clear'

__version__ = '0.18.4'

VERSION = __version__

if platform.python_implementation() == 'PyPy':
    SCIPY_MIN_VERSION = '1.1.0'
    NUMPY_MIN_VERSION = '1.14.0'
else:
    SCIPY_MIN_VERSION = '0.19.0'
    NUMPY_MIN_VERSION = '1.13.0'

JOBLIB_MIN_VERSION = '1.2.0'
SKLEARN_MIN_VERSION = '0.18.0'
THREADPOOLCTL_MIN_VERSION = '2.0.0'
PANDAS_MIN_VERSION = '0.25.3'
QUERIER_MIN_VERSION = '0.4.0'
REQUESTS_MIN_VERSION = '2.31.0'
if platform.system() in ('Linux', 'Darwin'):
    JAX_MIN_VERSION = '0.1.72'
    JAXLIB_MIN_VERSION = '0.1.51'

SETUPTOOLS_COMMANDS = {
    'develop', 'release', 'bdist_egg', 'bdist_rpm',
    'bdist_wininst', 'install_egg_info', 'build_sphinx',
    'egg_info', 'easy_install', 'upload', 'bdist_wheel',
    '--single-version-externally-managed',
}

if SETUPTOOLS_COMMANDS.intersection(sys.argv):
    extra_setuptools_args = dict(
        zip_safe=False,  # the package can run out of an .egg file
        include_package_data=True,
        extras_require={
            'alldeps': (
                'numpy >= {}'.format(NUMPY_MIN_VERSION),
                'scipy >= {}'.format(SCIPY_MIN_VERSION),
            ),
        },
    )
else:
    extra_setuptools_args = dict()


WHEELHOUSE_UPLOADER_COMMANDS = {'fetch_artifacts', 'upload_all'}
if WHEELHOUSE_UPLOADER_COMMANDS.intersection(sys.argv):
    import wheelhouse_uploader.cmd
    cmdclass.update(vars(wheelhouse_uploader.cmd))

ext_modules =[
    Extension(name="mlsauce.adaopt._adaoptc", 
              library_dirs=["mlsauce/adaopt/"],
              runtime_library_dirs=["mlsauce/adaopt/"],
              sources=["mlsauce/adaopt/_adaoptc.pyx"],
              include_dirs=[numpy.get_include()]),    
    Extension(name="mlsauce.booster._boosterc", 
              library_dirs=["mlsauce/booster/"],
              runtime_library_dirs=["mlsauce/booster/"],
              sources=["mlsauce/booster/_boosterc.pyx"], 
              include_dirs=[numpy.get_include()]),    
    Extension(name="mlsauce.lasso._lassoc", 
              library_dirs=["mlsauce/lasso/"],
              runtime_library_dirs=["mlsauce/lasso/"],
              sources=["mlsauce/lasso/_lassoc.pyx"], 
              include_dirs=[numpy.get_include()]),    
    Extension(name="mlsauce.ridge._ridgec", 
              library_dirs=["mlsauce/ridge/"],
              runtime_library_dirs=["mlsauce/ridge/"],
              sources=["mlsauce/ridge/_ridgec.pyx"], 
              include_dirs=[numpy.get_include()]),    
    Extension(name="mlsauce.stump._stumpc", 
              library_dirs=["mlsauce/stump/"],
              runtime_library_dirs=["mlsauce/stump/"],
              sources=["mlsauce/stump/_stumpc.pyx"], 
              include_dirs=[numpy.get_include()]),    
]

# Get the absolute path to the directory containing the setup script
abs_path = os.path.dirname(__file__)
rel_path = os.path.relpath(abs_path)
script_dir = rel_path
# Get absolute paths to Cython source files
adaopt_cython_file = str(script_dir + 'mlsauce/adaopt/_adaoptc.pyx')
booster_cython_file = str(script_dir + 'mlsauce/booster/_boosterc.pyx')
lasso_cython_file = str(script_dir + 'mlsauce/lasso/_lassoc.pyx')
ridge_cython_file = str(script_dir + 'mlsauce/ridge/_ridgec.pyx')
stump_cython_file = str(script_dir + 'mlsauce/stump/_stumpc.pyx')
# Update Extension definitions with absolute paths
ext_modules2 = [
    Extension(name="mlsauce.adaopt._adaoptc", 
              sources=[adaopt_cython_file],
              include_dirs=[numpy.get_include()]),
    Extension(name="mlsauce.booster._boosterc", 
              sources=[booster_cython_file],
              include_dirs=[numpy.get_include()]),
    Extension(name="mlsauce.booster._lassoc", 
              sources=[booster_cython_file],
              include_dirs=[numpy.get_include()]),
    Extension(name="mlsauce.booster._ridgec", 
              sources=[booster_cython_file],
              include_dirs=[numpy.get_include()]),
    Extension(name="mlsauce.booster._stumpc", 
              sources=[booster_cython_file],
              include_dirs=[numpy.get_include()]),                            
]


def setup_package():

    install_all_requires = [
        "numpy",
        "Cython",
        "joblib",
        "matplotlib",
        "pandas",
        "requests",
        "scikit-learn",
        "scipy",
        "tqdm"
    ]

    if platform.system() in ('Linux', 'Darwin'):
        install_jax_requires = ['jax', 'jaxlib']  
    else:
        install_jax_requires = []

    install_requires = [item for sublist in [install_all_requires, install_jax_requires] for item in sublist]

    try: 
        cythonize_ext_modules = cythonize(ext_modules2) 
    except: 
        cythonize_ext_modules = cythonize(ext_modules) 
        
    metadata = dict(name=DISTNAME,
                    maintainer=MAINTAINER,
                    maintainer_email=MAINTAINER_EMAIL,
                    description=DESCRIPTION,
                    license=LICENSE,
                    version=VERSION,
                    long_description=LONG_DESCRIPTION,
                    classifiers=['Development Status :: 2 - Pre-Alpha',
                                 'Intended Audience :: Developers',
                                 'License :: OSI Approved :: BSD License',
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
                    setup_requires=["numpy>= 1.13.0"],
                    package_data={'': ['*.pxd']},
                    packages=find_packages(),                    
                    ext_modules=cythonize_ext_modules,
                    **extra_setuptools_args)    
    
    
    setup(**metadata)

if __name__ == "__main__":
    setup_package()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    folders = ['adaopt', 'booster', 'elasticnet', 'lasso', 'ridge', 'stump']
    for folder in folders: 
        filename = os.path.join(dir_path, "mlsauce", folder, 'setup2.py')                
        subprocess.run(['python3', filename, 'build_ext', '--inplace'])      