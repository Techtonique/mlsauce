import numpy as np
import pandas as pd
import time

try:
    import xgboost as xgb
except ImportError:
    pass

from functools import partial
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from copy import deepcopy
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.utils import all_estimators
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import RegressorMixin
from sklearn.metrics import (
    r2_score
)
from .config import REGRESSORS
from ..booster import GenericBoostingRegressor

import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.precision", 2)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ]
)

try:
    categorical_transformer_low = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="missing"),
            ),
            (
                "encoding",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
except TypeError:
    categorical_transformer_low = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="missing"),
            ),
            (
                "encoding",
                OneHotEncoder(handle_unknown="ignore", sparse=False),
            ),
        ]
    )

categorical_transformer_high = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        # 'OrdinalEncoder' Raise a ValueError when encounters an unknown value. Check https://github.com/scikit-learn/scikit-learn/pull/13423
        ("encoding", OrdinalEncoder()),
    ]
)


# Helper functions


def get_card_split(df, cols, n=11):
    cond = df[cols].nunique() > n
    card_high = cols[cond]
    card_low = cols[~cond]
    return card_low, card_high


def adjusted_rsquared(r2, n, p):
    return 1 - (1 - r2) * ((n - 1) / (n - p - 1))

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


class LazyBoostingRegressor(RegressorMixin):
    """
        Fitting -- almost -- all the regression algorithms
        and returning their scores.

    Parameters:

        verbose: int, optional (default=0)
            Any positive number for verbosity.

        ignore_warnings: bool, optional (default=True)
            When set to True, the warning related to algorigms that are not able to run are ignored.

        custom_metric: function, optional (default=None)
            When function is provided, models are evaluated based on the custom evaluation metric provided.

        predictions: bool, optional (default=False)
            When set to True, the predictions of all the models models are returned as dataframe.

        sort_by: string, optional (default='RMSE')
            Sort models by a metric. Available options are 'R-Squared', 'Adjusted R-Squared', 'RMSE', 'Time Taken' and 'Custom Metric'.
            or a custom metric identified by its name and provided by custom_metric.

        random_state: int, optional (default=42)
            Reproducibiility seed.

        estimators: list, optional (default='all')
            list of Estimators names or just 'all' (default='all')

        preprocess: bool
            preprocessing is done when set to True

        n_jobs : int, when possible, run in parallel
            For now, only used by individual models that support it.

        n_layers: int, optional (default=3)
            Number of layers of CustomRegressors to be used.

        All the other parameters are the same as CustomRegressor's.

    Attributes:

        models_: dict-object
            Returns a dictionary with each model pipeline as value
            with key as name of models.

        best_model_: object
            Returns the best model pipeline based on the sort_by metric.

    Examples:

        ```python
        import os
        import mlsauce as ms
        from sklearn.datasets import load_diabetes
        from sklearn.model_selection import train_test_split

        data = load_diabetes()
        X = data.data
        y= data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 123)

        regr = ms.LazyBoostingRegressor(verbose=0, ignore_warnings=True,
                                        custom_metric=None, preprocess=True)
        models, predictioms = regr.fit(X_train, X_test, y_train, y_test)
        model_dictionary = regr.provide_models(X_train, X_test, y_train, y_test)
        print(models)
        ```

    """

    def __init__(
        self,
        verbose=0,
        ignore_warnings=True,
        custom_metric=None,
        predictions=False,
        sort_by="RMSE",
        random_state=42,
        estimators="all",
        preprocess=False,
        n_jobs=None,
    ):
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.custom_metric = custom_metric
        self.predictions = predictions
        self.sort_by = sort_by
        self.models_ = {}
        self.best_model_ = None
        self.random_state = random_state
        self.estimators = estimators
        self.preprocess = preprocess
        self.n_jobs = n_jobs

    def fit(self, X_train, X_test, y_train, y_test, hist=False, **kwargs):
        """Fit Regression algorithms to X_train and y_train, predict and score on X_test, y_test.

        Parameters:

            X_train : array-like,
                Training vectors, where rows is the number of samples
                and columns is the number of features.

            X_test : array-like,
                Testing vectors, where rows is the number of samples
                and columns is the number of features.

            y_train : array-like,
                Training vectors, where rows is the number of samples
                and columns is the number of features.

            y_test : array-like,
                Testing vectors, where rows is the number of samples
                and columns is the number of features.
            
            hist: bool, optional (default=False)
                When set to True, the model is a HistGenericBoostingRegressor.

            **kwargs: dict,
                Additional parameters to be passed to the GenericBoostingRegressor.

        Returns:
        -------
        scores:  Pandas DataFrame
            Returns metrics of all the models in a Pandas DataFrame.

        predictions : Pandas DataFrame
            Returns predictions of all the models in a Pandas DataFrame.

        """
        R2 = []
        ADJR2 = []
        RMSE = []
        # WIN = []
        names = []
        TIME = []
        predictions = {}

        if self.custom_metric:
            CUSTOM_METRIC = []

        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)

        numeric_features = X_train.select_dtypes(include=[np.number]).columns
        categorical_features = X_train.select_dtypes(include=["object"]).columns

        categorical_low, categorical_high = get_card_split(
            X_train, categorical_features
        )

        if self.preprocess is True:
            preprocessor = ColumnTransformer(
                transformers=[
                    ("numeric", numeric_transformer, numeric_features),
                    (
                        "categorical_low",
                        categorical_transformer_low,
                        categorical_low,
                    ),
                    (
                        "categorical_high",
                        categorical_transformer_high,
                        categorical_high,
                    ),
                ]
            )

        # base models
        try:
            baseline_names = [
                "RandomForestRegressor",
                "XGBRegressor",
                "GradientBoostingRegressor",
            ]
            baseline_models = [
                RandomForestRegressor(),
                xgb.XGBRegressor(),
                GradientBoostingRegressor(),
            ]
        except Exception as exception:
            baseline_names = [
                "RandomForestRegressor",
                "GradientBoostingRegressor",
            ]
            baseline_models = [
                RandomForestRegressor(),
                GradientBoostingRegressor(),
            ]

        if self.verbose > 0:
            print("\n Fitting baseline models...")
        for name, model in tqdm(zip(baseline_names, baseline_models)):
            start = time.time()
            try:
                model.fit(X_train, y_train.ravel())
                self.models_[name] = model
                y_pred = model.predict(X_test)
                r_squared = r2_score(y_test, y_pred)
                adj_rsquared = adjusted_rsquared(
                    r_squared, X_test.shape[0], X_test.shape[1]
                )
                rmse = root_mean_squared_error(y_test, y_pred)

                names.append(name)
                R2.append(r_squared)
                ADJR2.append(adj_rsquared)
                RMSE.append(rmse)
                TIME.append(time.time() - start)

                if self.custom_metric:
                    custom_metric = self.custom_metric(y_test, y_pred)
                    CUSTOM_METRIC.append(custom_metric)

                if self.verbose > 0:
                    scores_verbose = {
                        "Model": name,
                        "R-Squared": r_squared,
                        "Adjusted R-Squared": adj_rsquared,
                        "RMSE": rmse,
                        "Time taken": time.time() - start,
                    }

                    if self.custom_metric:
                        scores_verbose["Custom metric"] = custom_metric

                    print(scores_verbose)
                if self.predictions:
                    predictions[name] = y_pred
            except Exception as exception:
                if self.ignore_warnings is False:
                    print(name + " model failed to execute")
                    print(exception)

        if self.estimators == "all":
            self.regressors = REGRESSORS
        else:
            self.regressors = [
                ("GenericBooster(" + est[0] + ")", est[1](**kwargs))
                for est in all_estimators()
                if (
                    issubclass(est[1], RegressorMixin)
                    and (est[0] in self.estimators)
                )
            ]

        if self.preprocess is True:

            if self.n_jobs is None:

                for name, regr in tqdm(self.regressors):  # do parallel exec

                    start = time.time()

                    try:

                        if hist is False:

                            model = GenericBoostingRegressor(
                                base_model=regr(), verbose=self.verbose, **kwargs
                            )
                        
                        else:

                            model = HistGenericBoostingRegressor(
                                base_model=regr(), verbose=self.verbose, **kwargs
                            )

                        model.fit(X_train, y_train.ravel())

                        pipe = Pipeline(
                            steps=[
                                ("preprocessor", preprocessor),
                                ("regressor", model),
                            ]
                        )
                        if self.verbose > 0:
                            print("\n Fitting boosted " + name + " model...")
                        pipe.fit(X_train, y_train.ravel())

                        self.models_[name] = pipe
                        y_pred = pipe.predict(X_test)
                        r_squared = r2_score(y_test, y_pred)
                        adj_rsquared = adjusted_rsquared(
                            r_squared, X_test.shape[0], X_test.shape[1]
                        )
                        rmse = root_mean_squared_error(y_test, y_pred)

                        names.append(name)
                        R2.append(r_squared)
                        ADJR2.append(adj_rsquared)
                        RMSE.append(rmse)
                        TIME.append(time.time() - start)

                        if self.custom_metric:
                            custom_metric = self.custom_metric(y_test, y_pred)
                            CUSTOM_METRIC.append(custom_metric)

                        if self.verbose > 0:
                            scores_verbose = {
                                "Model": name,
                                "R-Squared": r_squared,
                                "Adjusted R-Squared": adj_rsquared,
                                "RMSE": rmse,
                                "Time taken": time.time() - start,
                            }

                            if self.custom_metric:
                                scores_verbose["Custom metric"] = custom_metric

                            print(scores_verbose)
                        if self.predictions:
                            predictions[name] = y_pred

                    except Exception as exception:

                        if self.ignore_warnings is False:
                            print(name + " model failed to execute")
                            print(exception)

            else:

                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(self.train_model)(
                        name,
                        model,
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        use_preprocessing=True,
                        preprocessor=preprocessor,
                        **kwargs
                    )
                    for name, model in tqdm(self.regressors)
                )
                R2 = [
                    result["r_squared"]
                    for result in results
                    if result is not None
                ]
                ADJR2 = [
                    result["adj_rsquared"]
                    for result in results
                    if result is not None
                ]
                RMSE = [
                    result["rmse"] for result in results if result is not None
                ]
                TIME = [
                    result["time"] for result in results if result is not None
                ]
                names = [
                    result["name"] for result in results if result is not None
                ]
                if self.custom_metric:
                    CUSTOM_METRIC = [
                        result["custom_metric"]
                        for result in results
                        if result is not None
                    ]
                if self.predictions:
                    predictions = {
                        result["name"]: result["predictions"]
                        for result in results
                        if result is not None
                    }

        else:  # self.preprocess is False; no preprocessing

            if self.n_jobs is None:

                for name, regr in tqdm(self.regressors):  # do parallel exec
                    start = time.time()
                    try:
                        
                        if hist is False:
                            model = GenericBoostingRegressor(
                                base_model=regr(), verbose=self.verbose, **kwargs
                            )
                        else:
                            model = HistGenericBoostingRegressor(
                                base_model=regr(), verbose=self.verbose, **kwargs
                            )

                        if self.verbose > 0:
                            print("\n Fitting boosted " + name + " model...")
                        model.fit(X_train, y_train.ravel())

                        self.models_[name] = model
                        y_pred = model.predict(X_test)

                        r_squared = r2_score(y_test, y_pred)
                        adj_rsquared = adjusted_rsquared(
                            r_squared, X_test.shape[0], X_test.shape[1]
                        )
                        rmse = root_mean_squared_error(y_test, y_pred)

                        names.append(name)
                        R2.append(r_squared)
                        ADJR2.append(adj_rsquared)
                        RMSE.append(rmse)
                        TIME.append(time.time() - start)

                        if self.custom_metric:
                            custom_metric = self.custom_metric(y_test, y_pred)
                            CUSTOM_METRIC.append(custom_metric)

                        if self.verbose > 0:
                            scores_verbose = {
                                "Model": name,
                                "R-Squared": r_squared,
                                "Adjusted R-Squared": adj_rsquared,
                                "RMSE": rmse,
                                "Time taken": time.time() - start,
                            }

                            if self.custom_metric:
                                scores_verbose["Custom metric"] = custom_metric

                            print(scores_verbose)
                        if self.predictions:
                            predictions[name] = y_pred
                    except Exception as exception:
                        if self.ignore_warnings is False:
                            print(name + " model failed to execute")
                            print(exception)

            else:

                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(self.train_model)(
                        name,
                        model,
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        use_preprocessing=False,
                        **kwargs
                    )
                    for name, model in tqdm(self.regressors)
                )
                R2 = [
                    result["r_squared"]
                    for result in results
                    if result is not None
                ]
                ADJR2 = [
                    result["adj_rsquared"]
                    for result in results
                    if result is not None
                ]
                RMSE = [
                    result["rmse"] for result in results if result is not None
                ]
                TIME = [
                    result["time"] for result in results if result is not None
                ]
                names = [
                    result["name"] for result in results if result is not None
                ]
                if self.custom_metric:
                    CUSTOM_METRIC = [
                        result["custom_metric"]
                        for result in results
                        if result is not None
                    ]
                if self.predictions:
                    predictions = {
                        result["name"]: result["predictions"]
                        for result in results
                        if result is not None
                    }

        scores = {
            "Model": names,
            "Adjusted R-Squared": ADJR2,
            "R-Squared": R2,
            "RMSE": RMSE,
            "Time Taken": TIME,
        }

        if self.custom_metric:
            scores["Custom metric"] = CUSTOM_METRIC

        scores = pd.DataFrame(scores)
        scores = scores.sort_values(by=self.sort_by, ascending=True).set_index(
            "Model"
        )

        self.best_model_ = self.models_[scores.index[0]]

        if self.predictions:
            predictions_df = pd.DataFrame.from_dict(predictions)
        return scores, predictions_df if self.predictions is True else scores

    def get_best_model(self):
        """
        This function returns the best model pipeline based on the sort_by metric.

        Returns:

            best_model: object,
                Returns the best model pipeline based on the sort_by metric.

        """
        return self.best_model_

    def provide_models(self, X_train, X_test, y_train, y_test):
        """
        This function returns all the model objects trained in fit function.
        If fit is not called already, then we call fit and then return the models.

        Parameters:

            X_train : array-like,
                Training vectors, where rows is the number of samples
                and columns is the number of features.

            X_test : array-like,
                Testing vectors, where rows is the number of samples
                and columns is the number of features.

            y_train : array-like,
                Training vectors, where rows is the number of samples
                and columns is the number of features.

            y_test : array-like,
                Testing vectors, where rows is the number of samples
                and columns is the number of features.

        Returns:

            models: dict-object,
                Returns a dictionary with each model pipeline as value
                with key as name of models.

        """
        if len(self.models_.keys()) == 0:
            self.fit(X_train, X_test, y_train.ravel(), y_test.values)

        return self.models_

    def train_model(
        self,
        name,
        regr,
        X_train,
        y_train,
        X_test,
        y_test,
        use_preprocessing=False,
        preprocessor=None,
        hist=False,
        **kwargs
    ):
        """
        Function to train a single regression model and return its results.
        """
        start = time.time()

        try:
            if hist is False:
                model = GenericBoostingRegressor(
                    base_model=regr(), verbose=self.verbose, **kwargs
                )
            else:
                model = HistGenericBoostingRegressor(
                    base_model=regr(), verbose=self.verbose, **kwargs
                )

            if use_preprocessing and preprocessor is not None:
                pipe = Pipeline(
                    steps=[
                        ("preprocessor", preprocessor),
                        ("regressor", model),
                    ]
                )
                if self.verbose > 0:
                    print(
                        "\n Fitting boosted "
                        + name
                        + " model with preprocessing..."
                    )
                pipe.fit(X_train, y_train.ravel())
                y_pred = pipe.predict(X_test)
                fitted_model = pipe
            else:
                # Case with no preprocessing
                if self.verbose > 0:
                    print(
                        "\n Fitting boosted "
                        + name
                        + " model without preprocessing..."
                    )
                model.fit(X_train, y_train.ravel())
                y_pred = model.predict(X_test)
                fitted_model = model

            r_squared = r2_score(y_test, y_pred)
            adj_rsquared = adjusted_rsquared(
                r_squared, X_test.shape[0], X_test.shape[1]
            )
            rmse = root_mean_squared_error(y_test, y_pred)

            custom_metric = None
            if self.custom_metric:
                custom_metric = self.custom_metric(y_test, y_pred)

            return {
                "name": name,
                "model": fitted_model,
                "r_squared": r_squared,
                "adj_rsquared": adj_rsquared,
                "rmse": rmse,
                "custom_metric": custom_metric,
                "time": time.time() - start,
                "predictions": y_pred,
            }

        except Exception as exception:
            if self.ignore_warnings is False:
                print(name + " model failed to execute")
                print(exception)
            return None
