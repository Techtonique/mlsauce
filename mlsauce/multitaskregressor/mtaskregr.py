import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.base import BaseEstimator, RegressorMixin
from ..utils import is_multitask_estimator


class MultiTaskRegressor(BaseEstimator, RegressorMixin):
    """
    A class for multi-task regression

    Parameters
    ----------
    regr: object
        A regressor object

    Attributes
    ----------
    objs: list
        A list containing the fitted regressor objects

    """

    def __init__(self, regr):
        assert (
            is_multitask_estimator(regr) == False
        ), "The regressor is already a multi-task regressor"
        self.regr = regr
        self.objs = []

    def fit(self, X, y):
        """
        Fit the regressor

        Parameters
        ----------
        X: array-like
            The input data
        y: array-like
            The target values

        """
        n_tasks = y.shape[1]
        assert n_tasks > 1, "The number of columns in y must be greater than 1"
        self.n_outputs_ = n_tasks
        try:
            for i in range(n_tasks):
                self.regr.fit(X, y.iloc[:, i].values)
                self.objs.append(deepcopy(self.regr))
        except Exception:
            for i in range(n_tasks):
                self.regr.fit(X, y[:, i])
                self.objs.append(deepcopy(self.regr))
        return self

    def predict(self, X):
        """
        Predict the target values

        Parameters
        ----------
        X: array-like
            The input data

        Returns
        -------
        y_pred: array-like
            The predicted target values

        """
        assert len(self.objs) > 0, "The regressor has not been fitted yet"
        y_pred = np.zeros((X.shape[0], self.n_outputs_))
        for i in range(self.n_outputs_):
            y_pred[:, i] = self.objs[i].predict(X)
        return y_pred
