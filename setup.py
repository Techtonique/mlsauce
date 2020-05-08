#!/usr/bin/env python

# Set this to True to enable building extensions using Cython.
# Set it to False to build extensions from the C file (that
# was previously created using Cython).
# Set it to 'auto' to build with Cython if available, otherwise
# from the C file.
USE_CYTHON = True

import sys
from distutils.extension import Extension
from setuptools import setup, find_packages

# cython

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()
    
requirements = ["numpy >= 1.13.0", "scipy >= 0.19.0", 
                "scikit-learn >= 0.18.0", "Cython >= 0.29.13"]

setup_requirements = [ ]

test_requirements = [ ]    


# python

if USE_CYTHON:
    try:
        from Cython.Distutils import build_ext
    except ImportError:
        if USE_CYTHON=='auto':
            USE_CYTHON=False
        else:
            raise 
            
cmdclass = { }
ext_modules = [ ]


if USE_CYTHON:
    ext_modules += [
        Extension("mlsauce.adaopt.adaoptc", [ "adaopt/adaoptc.pyx" ],
                libraries=["m"],
                extra_compile_args=["-ffast-math"]),
    ]
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules += [
        Extension("mlsauce.adaopt.adaoptc", [ "adaopt/adaoptc.c" ],
                libraries=["m"],
                extra_compile_args=["-ffast-math"]),
    ]


setup(
    author="T. Moudiki",
    author_email='thierry.moudiki@gmail.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Statistical/Machine Learning",
    install_requires=requirements,
    license="BSD3 Clear license",
    long_description="Miscellaneous Statistical/Machine Learning stuff",
    include_package_data=True,
    cmdclass = cmdclass, # cython
    ext_modules=ext_modules, # cython
    keywords='mlsauce',
    name='mlsauce',
    packages=find_packages(include=['mlsauce', 'mlsauce.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/thierrymoudiki/mlsauce',
    version='0.2.2',
    zip_safe=False,
)
