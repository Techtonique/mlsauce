#!/usr/bin/env python

import numpy
from setuptools import setup, find_packages
from setuptools.extension import Extension


with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

    
requirements = ["Cython >= 0.29.13", 
                "numpy >= 1.13.0", "scipy >= 0.19.0", 
                "scikit-learn >= 0.18.0", "tqdm>=4.46.0"]


setup_requirements = [ ]

test_requirements = [ ]    


extensions = [
    Extension(name="mlsauce.adaopt_cython.adaoptc", 
              sources=[ "mlsauce/adaopt_cython/adaoptc.c" ],
              include_dirs = [numpy.get_include()],
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
    description="Miscellaneous Statistical/Machine Learning tools",
    install_requires=requirements,
    license="BSD3 Clear license",
    long_description="Miscellaneous Statistical/Machine Learning stuff",
    include_package_data=True,
    ext_modules=extensions, # cython modules
    keywords='mlsauce',
    name='mlsauce',
    packages=find_packages(include=['mlsauce', 'mlsauce.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/thierrymoudiki/mlsauce',
    version='0.2.3',
    zip_safe=False,
)
