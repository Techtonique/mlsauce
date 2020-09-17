mlsauce
--------

![mlsauce logo](the-mlsauce.png)

<hr>

Miscellaneous Statistical/Machine learning stuff.  

![PyPI](https://img.shields.io/pypi/v/mlsauce) [![PyPI - License](https://img.shields.io/pypi/l/mlsauce)](https://github.com/thierrymoudiki/mlsauce/blob/master/LICENSE) [![Downloads](https://pepy.tech/badge/mlsauce)](https://pepy.tech/project/mlsauce)
[![Documentation](https://img.shields.io/badge/documentation-is_here-green)](https://techtonique.github.io/mlsauce/)


## Contents 
 [Installation for Python and R](#installation-for-Python-and-R) |
 [Package description](#package-description) |
 [Quick start](#quick-start) |
 [Contributing](#Contributing) |
 [Tests](#Tests) |
 [Dependencies](#dependencies) |
 [Citing `mlsauce`](#Citation) |
 [API Documentation](#api-documentation) |
 [References](#References) |
 [License](#License) 


## Installation (for Python and R)

### Python 

- __1st method__: by using `pip` at the command line for the stable version

```bash
pip install mlsauce
```


- __2nd method__: from Github, for the development version

```bash
pip install git+https://github.com/thierrymoudiki/mlsauce.git
```

### R 

- __1st method__: From Github, in R console:

```r
library(devtools)
devtools::install_github("thierrymoudiki/mlsauce/R-package")
library(mlsauce)
```

__General rule for using the package in R__:  object accesses with `.`'s are replaced by `$`'s. See also [Quick start](#quick-start).



## Package description

Miscellaneous Statistical/Machine learning stuff. See next section. 

## Quick start

- [AdaOpt: probabilistic classifier based on a mix of multivariable optimization and a nearest neighbors algorithm](https://thierrymoudiki.github.io/blog/#AdaOpt)
- [LSBoost: Gradient Boosted randomized networks using Least Squares](https://thierrymoudiki.github.io/blog/#LSBoost)


## Contributing

Your contributions are welcome, and valuable. Please, make sure to __read__ the [Code of Conduct](CONTRIBUTING.md) first. If you're not comfortable with Git/Version Control yet, please use [this form](https://forms.gle/tm7dxP1jSc75puAb9) to provide a feedback.

In Pull Requests, let's strive to use [`black`](https://black.readthedocs.io/en/stable/) for formatting files: 

```bash
pip install black
black --line-length=80 file_submitted_for_pr.py
```

A few things that we could explore are:

- Enrich the [tests](#Tests)
- Continue to make `mlsauce` available to `R` users --> [here](./R-package)
- Any benchmarking of `mlsauce` models can be stored in [demo](/mlsauce/demo) (notebooks) or [examples](./examples) (flat files), with the following naming convention:  `yourgithubname_ddmmyy_shortdescriptionofdemo.[py|ipynb|R|Rmd]`


## Tests

**Ultimately**, tests for `mlsauce`'s features **will** be located [here](mlsauce/tests). In order to run them and obtain tests' coverage (using [`nose2`](https://nose2.readthedocs.io/en/latest/)), you'll do: 

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

## API Documentation

- https://techtonique.github.io/mlsauce/

## Dependencies 

- Numpy
- Scipy
- scikit-learn
- querier

## Citation

```
@misc{moudiki2019mlsauce,
author={Moudiki, Thierry},
title={\code{mlsauce}, {M}iscellaneous {S}tatistical/{M}achine {L}earning stuff},
howpublished={\url{https://github.com/thierrymoudiki/mlsauce}},
note={BSD 3-Clause Clear License. Version 0.x.x.},
year={2019--2020}
}
```

## References

- Moudiki, T. (2020). AdaOpt: Multivariable optimization for classification. 
    Available at: 
    https://www.researchgate.net/publication/341409169_AdaOpt_Multivariable_optimization_for_classification

## License

[BSD 3-Clause](LICENSE) © Thierry Moudiki, 2019. 


## Credits

This package was created with [Cookiecutter](https://github.com/cookiecutter/cookiecutter)  and the [project template](https://github.com/audreyr/cookiecutter-pypackage).

