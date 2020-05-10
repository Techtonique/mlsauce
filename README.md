mlsauce
--------



![teller logo](the-mlsauce.png)

<hr>

Miscellaneous Statistical/Machine learning stuff.  

![PyPI](https://img.shields.io/pypi/v/mlsauce) [![PyPI - License](https://img.shields.io/pypi/l/mlsauce)](https://github.com/thierrymoudiki/mlsauce/blob/master/LICENSE) [![Downloads](https://pepy.tech/badge/mlsauce)](https://pepy.tech/project/mlsauce)
[![Documentation Status](https://readthedocs.org/projects/mlsauce/badge/?version=latest)](https://mlsauce.readthedocs.io/en/latest/?badge=latest)


Installation
-------

- From Pypi (stable):

```bash
pip install mlsauce
```

- From Github (development):

```bash
pip install git+https://github.com/thierrymoudiki/mlsauce.git
```

Quickstart
-------

loren ipsum

Contributing
-------

Your contributions are welcome, and valuable. Please, make sure to __read__ the [Code of Conduct](CONTRIBUTING.md) first.

In Pull Requests, let's strive to use [`black`](https://black.readthedocs.io/en/stable/) for formatting: 

```bash
pip install black
black --line-length=80 file_submitted_for_pr.py
```

Tests
-------

Tests for `mlsauce`'s features are located [here](https://github.com/thierrymoudiki/mlsauce/tree/master/tests). In order to run them and obtain tests' coverage (using [`nose2`](https://nose2.readthedocs.io/en/latest/)), do: 

- Install packages required for testing: 

```bash
pip install nose2
pip install coverage
```

- Run tests and print coverage:

```bash
git clone https://github.com/thierrymoudiki/mlsauce.git
cd mlsauce
nose2 --with-coverage
```

- Obtain coverage reports:

At the command line:

```bash
coverage report -m
```

or an html report:

```bash
coverage html
```



Dependencies 
-------

- Numpy
- Scipy
- Cython


License
-------

[BSD 3-Clause](LICENSE) © Thierry Moudiki, 2019. 



Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
