# version 0.13.0

- add clustering to `LSBoostRegressor`, `LSBoostClassifier`, and `AdaOpt`

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