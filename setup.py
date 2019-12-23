#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ["numpy >= 1.13.0", "scipy >= 0.19.0", 
                "scikit-learn >= 0.18.0"]

setup_requirements = [ ]

test_requirements = [ ]

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
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='mlsauce',
    name='mlsauce',
    packages=find_packages(include=['mlsauce', 'mlsauce.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/thierrymoudiki/mlsauce',
    version='0.1.0',
    zip_safe=False,
)
