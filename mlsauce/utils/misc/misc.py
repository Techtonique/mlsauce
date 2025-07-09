# Authors: Thierry Moudiki
#
# License: BSD 3
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def cluster(
    X,
    n_clusters=None,
    method="kmeans",
    type_scaling="standard",
    training=True,
    scaler=None,
    label_encoder=None,
    clusterer=None,
    seed=123,
):

    assert method in ("kmeans", "gmm"), "method must be in ('kmeans', 'gmm')"
    assert type_scaling in (
        "standard",
        "minmax",
        "robust",
    ), "type_scaling must be in ('standard', 'minmax', 'robust')"

    if training:
        assert (
            n_clusters is not None
        ), "n_clusters must be provided at training time"
        if type_scaling == "standard":
            scaler = StandardScaler()
        elif type_scaling == "minmax":
            scaler = MinMaxScaler()
        elif type_scaling == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(
                "type_scaling must be in ('standard', 'minmax', 'robust')"
            )

        scaled_X = scaler.fit_transform(X)
        label_encoder = OneHotEncoder(handle_unknown="ignore")

        if method == "kmeans":
            clusterer = KMeans(
                n_clusters=n_clusters, random_state=seed, n_init="auto"
            ).fit(scaled_X)
            res = label_encoder.fit_transform(
                clusterer.labels_.reshape(-1, 1)
            ).toarray()
        elif method == "gmm":
            clusterer = GaussianMixture(
                n_components=n_clusters, random_state=seed
            ).fit(scaled_X)
            res = label_encoder.fit_transform(
                clusterer.predict(scaled_X).reshape(-1, 1)
            ).toarray()
        else:
            raise ValueError("method must be in ('kmeans', 'gmm')")

        return res, scaler, label_encoder, clusterer

    else:  # @ inference time

        assert (
            scaler is not None
        ), "scaler must be provided at inferlabel_encodere time"
        assert (
            label_encoder is not None
        ), "label_encoder must be provided at inferlabel_encodere time"
        assert (
            clusterer is not None
        ), "clusterer must be provided at inferlabel_encodere time"
        scaled_X = scaler.transform(X)

        return label_encoder.transform(
            clusterer.predict(scaled_X).reshape(-1, 1)
        ).toarray()

def convert_df_to_numeric(df):
    """
    Convert all columns of DataFrame to numeric type using astype with loop.

    Parameters:
        df (pd.DataFrame): Input DataFrame with mixed data types.

    Returns:
        pd.DataFrame: DataFrame with all columns converted to numeric type.
    """
    if isinstance(df, pd.DataFrame):
        for column in df.columns:
            # Attempt to convert the column to numeric type using astype
            try:
                df[column] = df[column].astype(float)
            except ValueError:
                print(f"Column '{column}' contains non-numeric values.")
        return df


def dict_to_dataframe_series(data, series_names):
    df = pd.DataFrame(
        np.zeros((len(data["Model"]), 2)), columns=["Model", "Time Taken"]
    )
    for key, value in data.items():
        if all(hasattr(elt, "__len__") for elt in value) and key not in (
            "Model",
            "Time Taken",
        ):
            for i, elt1 in enumerate(value):
                for j, elt2 in enumerate(elt1):
                    df.loc[i, f"{key}_{series_names[j]}"] = elt2
        else:
            df[key] = value
    return df


# merge two dictionaries
def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


# check if x is int
def is_int(x):
    try:
        return int(x) == x
    except:
        return False


# check if x is float
def is_float(x):
    return isinstance(x, float)


# check if the response contains only integers
def is_factor(y):
    n = len(y)
    ans = True
    idx = 0

    while idx < n:
        if is_int(y[idx]) & (is_float(y[idx]) == False):
            idx += 1
        else:
            ans = False
            break

    return ans


# flatten list of lists
def flatten(l):
    return [item for sublist in l for item in sublist]


import importlib
import subprocess
import sys


def install_package(package_name):
    """Install a package dynamically using pip."""
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", package_name]
    )


def is_multitask_estimator(estimator):
    if hasattr(estimator, "_get_tags"):
        return estimator._get_tags().get("multioutput", False)
    return False

def check_and_install(package_name):
    """Check if a package is installed; if not, install it."""
    try:
        # Check if the package is already installed by importing it
        importlib.import_module(package_name)
        #print(f"'{package_name}' is already installed.")
    except ImportError:
        #print(f"'{package_name}' not found. Installing...")
        install_package(package_name)
        # Retry importing the package after installation
        importlib.import_module(package_name)
        #print(f"'{package_name}' has been installed successfully.")
