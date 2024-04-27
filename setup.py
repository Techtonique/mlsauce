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

from distutils.command.clean import clean as Clean
from distutils.core import Extension, setup
from os import path
from pathlib import Path
from setuptools import find_packages

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

__version__ = '0.16.0'

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

# Custom clean command to remove build artifacts
class CleanCommand(Clean):
    description = "Remove build artifacts from the source tree"
    def run(self):
        Clean.run(self)
        # Remove c files if we are not within a sdist package
        cwd = os.path.abspath(os.path.dirname(__file__))
        remove_c_files = not os.path.exists(os.path.join(cwd, 'PKG-INFO'))
        if remove_c_files:
            print('Will remove generated .c files')
        if os.path.exists('build'):
            shutil.rmtree('build')
        for dirpath, dirnames, filenames in os.walk('mlsauce'):
            for filename in filenames:
                if any(filename.endswith(suffix) for suffix in
                       (".so", ".pyd", ".dll", ".pyc")):
                    os.unlink(os.path.join(dirpath, filename))
                    continue
                extension = os.path.splitext(filename)[1]
                if remove_c_files and extension in ['.c', '.cpp']:
                    pyx_file = str.replace(filename, extension, '.pyx')
                    if os.path.exists(os.path.join(dirpath, pyx_file)):
                        os.unlink(os.path.join(dirpath, filename))
            for dirname in dirnames:
                if dirname == '__pycache__':
                    shutil.rmtree(os.path.join(dirpath, dirname))

cmdclass = {'clean': CleanCommand}


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
script_dir = os.path.basename(__file__)
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
                    cmdclass=cmdclass,                    
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