#! /usr/bin/env python
#
# Copyright (C) 2020 T. Moudiki <thierry.moudiki@gmail.com>
# License: 3-clause BSD

import subprocess
import sys
import os
import platform
import shutil
from distutils.command.clean import clean as Clean
#from pkg_resources import parse_version
from distutils.command.sdist import sdist
from setuptools import find_packages
import traceback
import importlib
try:
    import builtins
except ImportError:
    # Python 2 compat: just to be able to declare that Python >=3.5 is needed.
    import __builtin__ as builtins

try: 
    subprocess.run(['pip', 'install', 'numpy>= 1.13.0'], check=False)
    subprocess.run(['pip', 'install', 'scipy>= 0.19.0'], check=False)
    subprocess.run(['pip', 'install', 'Cython>=0.29.21'], check=False)
except Exception:
    pass 

builtins.__MLSAUCE_SETUP__ = True


DISTNAME = 'mlsauce'
DESCRIPTION = 'Miscellaneous Statistical/Machine Learning tools'
LONG_DESCRIPTION = 'Miscellaneous Statistical/Machine Learning tools'
MAINTAINER = 'T. Moudiki'
MAINTAINER_EMAIL = 'thierry.moudiki@gmail.com'
LICENSE = 'BSD3 Clause Clear'

# We can actually import a restricted version of mlsauce that
# does not need the compiled code
import mlsauce

__version__ = '0.9.0'

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
if platform.system() in ('Linux', 'Darwin'):
    JAX_MIN_VERSION = '0.1.72'
    JAXLIB_MIN_VERSION = '0.1.51'

# Optional setuptools features
# We need to import setuptools early, if we want setuptools features,
# as it monkey-patches the 'setup' function
# For some commands, use setuptools
SETUPTOOLS_COMMANDS = {
    'develop', 'release', 'bdist_egg', 'bdist_rpm',
    'bdist_wininst', 'install_egg_info', 'build_sphinx',
    'egg_info', 'easy_install', 'upload', 'bdist_wheel',
    '--single-version-externally-managed',
}
if SETUPTOOLS_COMMANDS.intersection(sys.argv):

    import setuptools

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

# custom build_ext command to set OpenMP compile flags depending on os and
# compiler
# build_ext has to be imported after setuptools
try:
    from numpy.distutils.command.build_ext import build_ext  # noqa

    class build_ext_subclass(build_ext):
        def build_extensions(self):
            # from mlsauce._build_utils.openmp_helpers import get_openmp_flag

            # if mlsauce._OPENMP_SUPPORTED:
            #     openmp_flag = get_openmp_flag(self.compiler)

            #     for e in self.extensions:
            #         e.extra_compile_args += openmp_flag
            #         e.extra_link_args += openmp_flag

            build_ext.build_extensions(self)

    cmdclass['build_ext'] = build_ext_subclass
    cmdclass['sdist'] = sdist

except ImportError:
    # Numpy should not be a dependency just to be able to introspect
    # that python 3.6 is required.
    pass


# Optional wheelhouse-uploader features
# To automate release of binary packages for mlsauce we need a tool
# to download the packages generated by travis and appveyor workers (with
# version number matching the current release) and upload them all at once
# to PyPI at release time.
# The URL of the artifact repositories are configured in the setup.cfg file.

WHEELHOUSE_UPLOADER_COMMANDS = {'fetch_artifacts', 'upload_all'}
if WHEELHOUSE_UPLOADER_COMMANDS.intersection(sys.argv):
    import wheelhouse_uploader.cmd

    cmdclass.update(vars(wheelhouse_uploader.cmd))


def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    from mlsauce._build_utils import _check_cython_version

    config = Configuration(None, parent_package, top_path)

    # Avoid non-useful msg:
    # "Ignoring attempt to set 'name' (from ... "
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    # Cython is required by config.add_subpackage for templated extensions
    # that need the tempita sub-submodule. So check that we have the correct
    # version of Cython so as to be able to raise a more informative error
    # message from the start if it's not the case.
    _check_cython_version()

    config.add_subpackage('mlsauce')

    return config


# def check_package_status(package, min_version):
#     """
#     Returns a dictionary containing a boolean specifying whether given package
#     is up-to-date, along with the version string (empty string if
#     not installed).
#     """
#     package_status = {}
#     try:
#         module = importlib.import_module(package)
#         package_version = module.__version__
#         package_status['up_to_date'] = parse_version(
#             package_version) >= parse_version(min_version)
#         package_status['version'] = package_version
#     except ImportError:
#         traceback.print_exc()
#         package_status['up_to_date'] = False
#         package_status['version'] = ""

#     req_str = "mlsauce requires {} >= {}.\n".format(
#         package, min_version)

#     instructions = ("Installation instructions are available on the "
#                     "mlsauce GitHub repo: ")

#     if package_status['up_to_date'] is False:
#         if package_status['version']:
#             raise ImportError("Your installation of {} "
#                               "{} is out-of-date.\n{}{}"
#                               .format(package, package_status['version'],
#                                       req_str, instructions))
#         else:
#             raise ImportError("{} is not "
#                               "installed.\n{}{}"
#                               .format(package, req_str, instructions))


def setup_package():

    install_all_requires = [
                        'numpy>={}'.format(NUMPY_MIN_VERSION),
                        'scipy>={}'.format(SCIPY_MIN_VERSION),
                        'joblib>={}'.format(JOBLIB_MIN_VERSION),
                        'scikit-learn>={}'.format(SKLEARN_MIN_VERSION),
                        #'threadpoolctl>={}'.format(THREADPOOLCTL_MIN_VERSION),
                        'pandas>={}'.format(PANDAS_MIN_VERSION),
                        #'querier>={}'.format(QUERIER_MIN_VERSION)
                    ]

    install_jax_requires = [
                            'jax',
                            'jaxlib'
                            ] if platform.system() in ('Linux', 'Darwin') else []

    other_requirements = ["requests",
                          "tqdm", 
                          #"pymongo >= 3.10.1", 
                          #"SQLAlchemy >= 1.3.18"
                          ]

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
                    **extra_setuptools_args)

    if len(sys.argv) == 1 or (
            len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or
                                    sys.argv[1] in ('--help-commands',
                                                    'egg_info',
                                                    'dist_info',
                                                    '--version',
                                                    'clean'))):
        # For these actions, NumPy is not required
        #
        # They are required to succeed without Numpy for example when
        # pip is used to install mlsauce when Numpy is not yet present in
        # the system.
        try:
            from setuptools import setup
        except ImportError:
            from distutils.core import setup

        metadata['version'] = VERSION
    else:
        if sys.version_info < (3, 5):
            raise RuntimeError(
                "mlsauce requires Python 3.5 or later. The current"
                " Python version is %s installed in %s."
                % (platform.python_version(), sys.executable))

        #check_package_status('numpy', NUMPY_MIN_VERSION)

        #check_package_status('scipy', SCIPY_MIN_VERSION)

        from numpy.distutils.core import setup

        metadata['configuration'] = configuration

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
