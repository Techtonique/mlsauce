# version 0.25.0

- Fix setup.py to build extensions in place (at last!)

# version 0.22.3

- Add `GenericGradientBooster` for regression and classification. See: https://thierrymoudiki.github.io/blog/2024/10/06/python/r/genericboosting
and `examples/genboost*`

# version 0.18.2

- Gaussian weights in `LSBoostRegressor` and `LSBoostClassifier` randomized hidden layer

# version 0.17.0

- add `ElasticNetRegressor` `solver` to `LSBoostRegressor` and `LSBoostClassifier`

# version 0.16.0

- add clustering to `LSBoostRegressor`, `LSBoostClassifier`, and `AdaOpt`
- add polynomial features to `LSBoostRegressor`, `LSBoostClassifier`

# version 0.12.3

- add prediction intervals to `LSBoostRegressor` (split conformal prediction, 
  split conformal prediction with Kernel Density Estimation, and split 
  conformal prediction bootstrap)
  see `examples/lsboost_regressor_pi.py` for examples 
- do not rescale columns with zero variance in `LSBoostRegressor` and `LSBoostClassifier`
- faster ridge regression for `LSBoostRegressor` and `LSBoostClassifier`

# version 0.9.0

- dowload data from R-universe

# version 0.8.11

- install `numpy` before `setup`
- stop using `np.int`
- update Makefile
- update examples 
- no more refs to openmp (for now)
- update and align with R version
- submit conda version

# version 0.8.8
- Avoid division by 0 in scaling

# version 0.8.7
- Fix `row_sample` in `LSBoostRegressor`

# version 0.3.0

- include cython parallel processing Pt.1
- include manhattan distance

# version 0.2.0

- adjust setup.py

# version 0.1.0

- Initial version