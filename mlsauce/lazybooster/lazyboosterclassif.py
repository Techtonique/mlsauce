import nnetsauce as ns
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from copy import deepcopy
from functools import partial
from joblib import Parallel, delayed
from tqdm import tqdm
import time

try:
    import xgboost as xgb
except ImportError:
    pass

from sklearn.utils import all_estimators
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
)
from .config import REGRESSORS, MTASKREGRESSORS
from ..booster import GenericBoostingClassifier
from ..multitaskregressor import MultiTaskRegressor

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


def get_card_split(df, cols, n=11):
    cond = df[cols].nunique() > n
    card_high = cols[cond]
    card_low = cols[~cond]
    return card_low, card_high


class LazyBoostingClassifier(ClassifierMixin):
    """

    Fitting -- almost -- all the classification algorithms
    and returning their scores.

    Parameters:

        verbose: int, optional (default=0)
            Any positive number for verbosity.

        ignore_warnings: bool, optional (default=True)
            When set to True, the warning related to algorigms that are not
            able to run are ignored.

        custom_metric: function, optional (default=None)
            When function is provided, models are evaluated based on the custom
              evaluation metric provided.

        predictions: bool, optional (default=False)
            When set to True, the predictions of all the models models are
            returned as data frame.

        sort_by: string, optional (default='Accuracy')
            Sort models by a metric. Available options are 'Accuracy',
            'Balanced Accuracy', 'ROC AUC', 'F1 Score' or a custom metric
            identified by its name and provided by custom_metric.

        random_state: int, optional (default=42)
            Reproducibiility seed.

        estimators: list, optional (default='all')
            list of Estimators names or just 'all' for > 90 classifiers
            (default='all')

        preprocess: bool, preprocessing is done when set to True

        n_jobs: int, when possible, run in parallel
            For now, only used by individual models that support it.

        n_layers: int, optional (default=3)
            Number of layers of GenericBoostingClassifiers to be used.

        All the other parameters are the same as GenericBoostingClassifier's.

    Attributes:

        models_: dict-object
            Returns a dictionary with each model pipeline as value
            with key as name of models.

        best_model_: object
            Returns the best model pipeline.

    Examples

        ```python
        import os
        import mlsauce as ms
        from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
        from sklearn.model_selection import train_test_split
        from time import time

        load_models = [load_breast_cancer, load_iris, load_wine]

        for model in load_models:

            data = model()
            X = data.data
            y= data.target

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 13)

            clf = ms.LazyBoostingClassifier(verbose=1, ignore_warnings=False,
                                            custom_metric=None, preprocess=False)

            start = time()
            models, predictioms = clf.fit(X_train, X_test, y_train, y_test)
            print(f"\nElapsed: {time() - start} seconds\n")

            print(models)
        ```

    """

    def __init__(
        self,
        verbose=0,
        ignore_warnings=True,
        custom_metric=None,
        predictions=False,
        sort_by="Accuracy",
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

    def fit(self, X_train, X_test, y_train, y_test, hist=False,  **kwargs):
        """Fit classifiers to X_train and y_train, predict and score on X_test,
        y_test.

        Parameters:

            X_train: array-like,
                Training vectors, where rows is the number of samples
                and columns is the number of features.

            X_test: array-like,
                Testing vectors, where rows is the number of samples
                and columns is the number of features.

            y_train: array-like,
                Training vectors, where rows is the number of samples
                and columns is the number of features.

            y_test: array-like,
                Testing vectors, where rows is the number of samples
                and columns is the number of features.
            
            hist: bool, optional (default=False)
                When set to True, the model is a GenericBoostingClassifier.

            **kwargs: dict,
                Additional arguments to be passed to the fit GenericBoostingClassifier.

        Returns:

            scores: Pandas DataFrame
                Returns metrics of all the models in a Pandas DataFrame.

            predictions: Pandas DataFrame
                Returns predictions of all the models in a Pandas DataFrame.
        """
        Accuracy = []
        B_Accuracy = []
        ROC_AUC = []
        F1 = []
        names = []
        TIME = []
        predictions = {}

        if self.custom_metric is not None:
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

        # baseline models
        try:
            baseline_names = ["RandomForestClassifier", "XGBClassifier"]
            baseline_models = [RandomForestClassifier(), xgb.XGBClassifier()]
        except Exception as exception:
            baseline_names = ["RandomForestClassifier"]
            baseline_models = [RandomForestClassifier()]

        if self.verbose > 0:
            print("\n Fitting baseline models...")
        for name, model in tqdm(zip(baseline_names, baseline_models)):
            start = time.time()
            try:
                model.fit(X_train, y_train)
                self.models_[name] = model
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred, normalize=True)
                b_accuracy = balanced_accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                try:
                    roc_auc = roc_auc_score(y_test, y_pred)
                except Exception as exception:
                    roc_auc = None
                    if self.ignore_warnings is False:
                        print("ROC AUC couldn't be calculated for " + name)
                        print(exception)
                names.append(name)
                Accuracy.append(accuracy)
                B_Accuracy.append(b_accuracy)
                ROC_AUC.append(roc_auc)
                F1.append(f1)
                TIME.append(time.time() - start)
                if self.custom_metric is not None:
                    custom_metric = self.custom_metric(y_test, y_pred)
                    CUSTOM_METRIC.append(custom_metric)
                if self.verbose > 0:
                    if self.custom_metric is not None:
                        print(
                            {
                                "Model": name,
                                "Accuracy": accuracy,
                                "Balanced Accuracy": b_accuracy,
                                "ROC AUC": roc_auc,
                                "F1 Score": f1,
                                self.custom_metric.__name__: custom_metric,
                                "Time taken": time.time() - start,
                            }
                        )
                    else:
                        print(
                            {
                                "Model": name,
                                "Accuracy": accuracy,
                                "Balanced Accuracy": b_accuracy,
                                "ROC AUC": roc_auc,
                                "F1 Score": f1,
                                "Time taken": time.time() - start,
                            }
                        )
                if self.predictions:
                    predictions[name] = y_pred
            except Exception as exception:
                if self.ignore_warnings is False:
                    print(name + " model failed to execute")
                    print(exception)

        if self.estimators == "all":
            self.classifiers = REGRESSORS + MTASKREGRESSORS
        else:
            self.classifiers = [
                ("GBoostClassifier(" + est[0] + ")", est[1]())
                for est in all_estimators()
                if (
                    issubclass(est[1], RegressorMixin)
                    and (est[0] in self.estimators)
                )
            ] + [
                (
                    "GBoostClassifier(MultiTask(" + est[0] + "))",
                    partial(MultiTaskRegressor, regr=est[1]()),
                )
                for est in all_estimators()
                if (
                    issubclass(est[1], RegressorMixin)
                    and (est[0] in self.estimators)
                )
            ]

        if self.preprocess is True:

            if self.n_jobs is None:

                for name, model in tqdm(self.classifiers):  # do parallel exec

                    other_args = (
                        {}
                    )  # use this trick for `random_state` too --> refactor
                    try:
                        if (
                            "n_jobs" in model().get_params().keys()
                            and name.find("LogisticRegression") == -1
                        ):
                            other_args["n_jobs"] = self.n_jobs
                    except Exception:
                        pass

                    start = time.time()

                    try:
                        if "random_state" in model().get_params().keys():
                            if hist is False:
                                fitted_clf = GenericBoostingClassifier(
                                    {**other_args, **kwargs},
                                    verbose=self.verbose,
                                    base_model=model(
                                        random_state=self.random_state
                                    ),
                                )
                            else:
                                fitted_clf = GenericBoostingClassifier(
                                    {**other_args, **kwargs},
                                    verbose=self.verbose,
                                    base_model=model(
                                        random_state=self.random_state
                                    ),
                                    hist=True,
                                )

                        else:
                            if hist is False: 
                                fitted_clf = GenericBoostingClassifier(
                                    base_model=model(**kwargs),
                                    verbose=self.verbose,
                                )
                            else:
                                fitted_clf = GenericBoostingClassifier(
                                    base_model=model(**kwargs),
                                    verbose=self.verbose,
                                    hist=True,
                                )

                        if self.verbose > 0:
                            print("\n Fitting boosted " + name + " model...")
                        fitted_clf.fit(X_train, y_train)

                        pipe = Pipeline(
                            [
                                ("preprocessor", preprocessor),
                                ("classifier", fitted_clf),
                            ]
                        )

                        if self.verbose > 0:
                            print("\n Fitting boosted " + name + " model...")
                        pipe.fit(X_train, y_train)
                        self.models_[name] = pipe
                        y_pred = pipe.predict(X_test)
                        accuracy = accuracy_score(
                            y_test, y_pred, normalize=True
                        )
                        b_accuracy = balanced_accuracy_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred, average="weighted")
                        try:
                            roc_auc = roc_auc_score(y_test, y_pred)
                        except Exception as exception:
                            roc_auc = None
                            if self.ignore_warnings is False:
                                print(
                                    "ROC AUC couldn't be calculated for " + name
                                )
                                print(exception)
                        names.append(name)
                        Accuracy.append(accuracy)
                        B_Accuracy.append(b_accuracy)
                        ROC_AUC.append(roc_auc)
                        F1.append(f1)
                        TIME.append(time.time() - start)
                        if self.custom_metric is not None:
                            custom_metric = self.custom_metric(y_test, y_pred)
                            CUSTOM_METRIC.append(custom_metric)
                        if self.verbose > 0:
                            if self.custom_metric is not None:
                                print(
                                    {
                                        "Model": name,
                                        "Accuracy": accuracy,
                                        "Balanced Accuracy": b_accuracy,
                                        "ROC AUC": roc_auc,
                                        "F1 Score": f1,
                                        self.custom_metric.__name__: custom_metric,
                                        "Time taken": time.time() - start,
                                    }
                                )
                            else:
                                print(
                                    {
                                        "Model": name,
                                        "Accuracy": accuracy,
                                        "Balanced Accuracy": b_accuracy,
                                        "ROC AUC": roc_auc,
                                        "F1 Score": f1,
                                        "Time taken": time.time() - start,
                                    }
                                )
                        if self.predictions:
                            predictions[name] = y_pred
                    except Exception as exception:
                        if self.ignore_warnings is False:
                            print(name + " model failed to execute")
                            print(exception)

            else:

                # train_model(self, name, model, X_train, y_train, X_test, y_test,
                # use_preprocessing=False, preprocessor=None,
                #    **kwargs):
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
                    for name, model in tqdm(self.classifiers)
                )
                Accuracy = [res["accuracy"] for res in results]
                B_Accuracy = [res["balanced_accuracy"] for res in results]
                ROC_AUC = [res["roc_auc"] for res in results]
                F1 = [res["f1"] for res in results]
                names = [res["name"] for res in results]
                TIME = [res["time"] for res in results]
                if self.custom_metric is not None:
                    CUSTOM_METRIC = [res["custom_metric"] for res in results]
                if self.predictions:
                    predictions = {
                        res["name"]: res["predictions"] for res in results
                    }

        else:  # no preprocessing

            if self.n_jobs is None:

                for name, model in tqdm(self.classifiers):  # do parallel exec
                    start = time.time()
                    try:
                        if "random_state" in model().get_params().keys():
                            if hist is False:
                                fitted_clf = GenericBoostingClassifier(
                                    base_model=model(
                                        random_state=self.random_state
                                    ),
                                    verbose=self.verbose,
                                    **kwargs
                                )
                            else: 
                                fitted_clf = GenericBoostingClassifier(
                                    base_model=model(
                                        random_state=self.random_state
                                    ),
                                    verbose=self.verbose,
                                    hist=True,
                                    **kwargs
                                )

                        else:
                            if hist is False:
                                fitted_clf = GenericBoostingClassifier(
                                    base_model=model(),
                                    verbose=self.verbose,
                                    **kwargs
                                )
                            else:
                                fitted_clf = GenericBoostingClassifier(
                                    base_model=model(),
                                    verbose=self.verbose,
                                    hist=True,
                                    **kwargs
                                )

                        fitted_clf.fit(X_train, y_train)

                        self.models_[name] = fitted_clf
                        y_pred = fitted_clf.predict(X_test)
                        accuracy = accuracy_score(
                            y_test, y_pred, normalize=True
                        )
                        b_accuracy = balanced_accuracy_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred, average="weighted")
                        try:
                            roc_auc = roc_auc_score(y_test, y_pred)
                        except Exception as exception:
                            roc_auc = None
                            if self.ignore_warnings is False:
                                print(
                                    "ROC AUC couldn't be calculated for " + name
                                )
                                print(exception)
                        names.append(name)
                        Accuracy.append(accuracy)
                        B_Accuracy.append(b_accuracy)
                        ROC_AUC.append(roc_auc)
                        F1.append(f1)
                        TIME.append(time.time() - start)
                        if self.custom_metric is not None:
                            custom_metric = self.custom_metric(y_test, y_pred)
                            CUSTOM_METRIC.append(custom_metric)
                        if self.verbose > 0:
                            if self.custom_metric is not None:
                                print(
                                    {
                                        "Model": name,
                                        "Accuracy": accuracy,
                                        "Balanced Accuracy": b_accuracy,
                                        "ROC AUC": roc_auc,
                                        "F1 Score": f1,
                                        self.custom_metric.__name__: custom_metric,
                                        "Time taken": time.time() - start,
                                    }
                                )
                            else:
                                print(
                                    {
                                        "Model": name,
                                        "Accuracy": accuracy,
                                        "Balanced Accuracy": b_accuracy,
                                        "ROC AUC": roc_auc,
                                        "F1 Score": f1,
                                        "Time taken": time.time() - start,
                                    }
                                )
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
                    for name, model in tqdm(self.classifiers)
                )
                Accuracy = [res["accuracy"] for res in results]
                B_Accuracy = [res["balanced_accuracy"] for res in results]
                ROC_AUC = [res["roc_auc"] for res in results]
                F1 = [res["f1"] for res in results]
                names = [res["name"] for res in results]
                TIME = [res["time"] for res in results]
                if self.custom_metric is not None:
                    CUSTOM_METRIC = [res["custom_metric"] for res in results]
                if self.predictions:
                    predictions = {
                        res["name"]: res["predictions"] for res in results
                    }

        if self.custom_metric is None:
            scores = pd.DataFrame(
                {
                    "Model": names,
                    "Accuracy": Accuracy,
                    "Balanced Accuracy": B_Accuracy,
                    "ROC AUC": ROC_AUC,
                    "F1 Score": F1,
                    "Time Taken": TIME,
                }
            )
        else:
            scores = pd.DataFrame(
                {
                    "Model": names,
                    "Accuracy": Accuracy,
                    "Balanced Accuracy": B_Accuracy,
                    "ROC AUC": ROC_AUC,
                    "F1 Score": F1,
                    "Custom metric": CUSTOM_METRIC,
                    "Time Taken": TIME,
                }
            )
        scores = scores.sort_values(by=self.sort_by, ascending=False).set_index(
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
        """Returns all the model objects trained. If fit hasn't been called yet,
        then it's called to return the models.

        Parameters:

        X_train: array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.

        X_test: array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.

        y_train: array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.

        y_test: array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.

        Returns:

            models: dict-object,
                Returns a dictionary with each model's pipeline as value
                and key = name of the model.
        """
        if len(self.models_.keys()) == 0:
            self.fit(X_train, X_test, y_train, y_test)

        return self.models_

    def train_model(
        self,
        name,
        model,
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
        Function to train a single model and return its results.
        """
        other_args = {}

        # Handle n_jobs parameter
        try:
            if (
                "n_jobs" in model().get_params().keys()
                and "LogisticRegression" not in name
            ):
                other_args["n_jobs"] = self.n_jobs
        except Exception:
            pass

        start = time.time()

        try:
            # Handle random_state parameter
            if "random_state" in model().get_params().keys():
                if hist is False:
                    fitted_clf = GenericBoostingClassifier(
                        {**other_args, **kwargs},
                        verbose=self.verbose,
                        base_model=model(random_state=self.random_state),
                    )
                else:
                    fitted_clf = GenericBoostingClassifier(
                        {**other_args, **kwargs},
                        verbose=self.verbose,
                        base_model=model(random_state=self.random_state),
                        hist=True,
                    )
            else:
                if hist is False: 
                    fitted_clf = GenericBoostingClassifier(
                        base_model=model(**kwargs),
                        verbose=self.verbose,
                    )
                else:
                    fitted_clf = GenericBoostingClassifier(
                        base_model=model(**kwargs),
                        verbose=self.verbose,
                        hist=True,
                    )

            if self.verbose > 0:
                print("\n Fitting boosted " + name + " model...")

            fitted_clf.fit(X_train, y_train)

            if use_preprocessing and preprocessor is not None:
                pipe = Pipeline(
                    [
                        ("preprocessor", preprocessor),
                        ("classifier", fitted_clf),
                    ]
                )
                if self.verbose > 0:
                    print(
                        "\n Fitting pipeline with preprocessing for "
                        + name
                        + " model..."
                    )
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)
            else:
                # Case with no preprocessing
                if self.verbose > 0:
                    print(
                        "\n Fitting model without preprocessing for "
                        + name
                        + " model..."
                    )
                y_pred = fitted_clf.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred, normalize=True)
            b_accuracy = balanced_accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            roc_auc = None

            try:
                roc_auc = roc_auc_score(y_test, y_pred)
            except Exception as exception:
                if self.ignore_warnings is False:
                    print("ROC AUC couldn't be calculated for " + name)
                    print(exception)

            custom_metric = None
            if self.custom_metric is not None:
                custom_metric = self.custom_metric(y_test, y_pred)

            return {
                "name": name,
                "model": fitted_clf if not use_preprocessing else pipe,
                "accuracy": accuracy,
                "balanced_accuracy": b_accuracy,
                "roc_auc": roc_auc,
                "f1": f1,
                "custom_metric": custom_metric,
                "time": time.time() - start,
                "predictions": y_pred,
            }
        except Exception as exception:
            if self.ignore_warnings is False:
                print(name + " model failed to execute")
                print(exception)
            return None
