#! /usr/bin/env python
#
# Copyright (C) 2020 T. Moudiki <thierry.moudiki@gmail.com>
# License: 3-clause BSD

import os
import platform
import shutil
import subprocess
import sys

from distutils.command.clean import clean as Clean
from distutils.command.sdist import sdist
from distutils.core import Extension, setup

from setuptools import find_packages

try:
    import builtins
except ImportError:    
    import __builtin__ as builtins

subprocess.run(['pip', 'install', 'Cython>=0.29.21'], check=False)
subprocess.run(['pip', 'install', 'numpy>= 1.13.0'], check=False)
subprocess.run(['pip', 'install', 'scipy>= 0.19.0'], check=False)
subprocess.run(['pip', 'install', 'requests>=2.31.0'], check=False)

import numpy
from Cython.Build import cythonize

builtins.__MLSAUCE_SETUP__ = True

DISTNAME = 'mlsauce'
DESCRIPTION = 'Miscellaneous Statistical/Machine Learning tools'
LONG_DESCRIPTION = 'Miscellaneous Statistical/Machine Learning tools'
MAINTAINER = 'T. Moudiki'
MAINTAINER_EMAIL = 'thierry.moudiki@gmail.com'
LICENSE = 'BSD3 Clause Clear'

__version__ = '0.10.0'

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

def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')
    
    #from mlsauce._build_utils import _check_cython_version
    print(f"\n ----- parent_package: ----- \n {parent_package}")
    print(f"\n ----- top_path: ----- \n {top_path}")
    config = Extension(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)
    #_check_cython_version()
    config.add_subpackage('mlsauce')
    return config

ext_modules =[
    Extension("mlsauce/adaopt._adaoptc", 
              sources=["mlsauce/adaopt/_adaoptc.pyx"],
              include_dirs=[numpy.get_include()]),    
    Extension("mlsauce/booster._boosterc", 
              sources=["mlsauce/booster/_boosterc.pyx"], 
              include_dirs=[numpy.get_include()]),    
    Extension("mlsauce/lasso._lassoc", 
              sources=["mlsauce/lasso/_lassoc.pyx"], 
              include_dirs=[numpy.get_include()]),    
    Extension("mlsauce/ridge._ridgec", 
              sources=["mlsauce/ridge/_ridgec.pyx"], 
              include_dirs=[numpy.get_include()]),    
    Extension("mlsauce/stump._stumpc", 
              sources=["mlsauce/stump/_stumpc.pyx"], 
              include_dirs=[numpy.get_include()]),    
]

def setup_package():

    install_all_requires = [
                        'numpy>={}'.format(NUMPY_MIN_VERSION),
                        'scipy>={}'.format(SCIPY_MIN_VERSION),
                        'joblib>={}'.format(JOBLIB_MIN_VERSION),
                        'scikit-learn>={}'.format(SKLEARN_MIN_VERSION),
                        'pandas>={}'.format(PANDAS_MIN_VERSION),
                        'requests>={}'.format(REQUESTS_MIN_VERSION),
                    ]
    install_jax_requires = [
                            'jax',
                            'jaxlib'
                            ] if platform.system() in ('Linux', 'Darwin') else []
    other_requirements = ["tqdm"]

    install_requires = [item for sublist in [install_jax_requires, install_all_requires, other_requirements] for item in sublist]

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
                    python_requires=">=3.5",
                    install_requires=install_requires,
                    setup_requires=["numpy>= 1.13.0"],
                    package_data={'': ['*.pxd']},
                    packages=find_packages(),
                    #configuration=configuration,
                    ext_modules=cythonize(ext_modules),
                    **extra_setuptools_args)    

    setup(**metadata)

if __name__ == "__main__":
    setup_package()