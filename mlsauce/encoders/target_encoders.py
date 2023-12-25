import numpy as np
import pandas as pd
import pickle
import querier as qr
from tqdm import tqdm


def corrtarget_encoder(df, target, rho=0.4, verbose=1, seed=123):
    """Encode non-numerical columns using correlations.

    Parameters
    ----------
    df: a data frame
        a data frame

    target: str
        target column a.k.a response

    rho: float
        correlation between pseudo target (used for averaging) and target

    verbose: int
        currently 0 = nothing printed; 1 = progress bar printed

    seed: int
        reproducibility seed

    Returns
    --------
    a tuple: numerical data frame and achieved correlation

    """

    target_ = df[target].values
    target_mean = target_.mean()
    target_std = target_.std()
    n_target = len(target_)

    C = np.eye(2)
    C[0, 1] = rho
    C[1, 0] = rho
    C_ = np.linalg.cholesky(C).T

    np.random.seed(seed)
    temp = np.vstack((target_, np.random.normal(size=n_target))).T
    df_ = pickle.loads(pickle.dumps(df, -1))
    df_["pseudo_target"] = target_mean + np.dot(temp, C_)[:, 1] * target_std

    covariates_names = df.columns.values[df.columns.values != target].tolist()
    X = qr.select(df, ", ".join(covariates_names))
    X_dtypes = X.dtypes
    X_numeric = pickle.loads(pickle.dumps(df, -1))

    col_iterator = covariates_names if verbose == 0 else tqdm(covariates_names)

    for col in col_iterator:
        if X_dtypes[col] == np.object:  # something like a character string
            X_temp = qr.summarize(
                df_, req=col + ", avg(pseudo_target)", group_by=col
            )
            levels = np.unique(qr.select(X, col).values)

            for l in levels:
                qrobj = qr.Querier(X_temp)

                val = np.float(
                    qrobj.filtr(col + '== "' + l + '"')
                    .select("avg_pseudo_target")
                    .df.values
                )

                X_numeric = qr.setwhere(
                    X_numeric, col=col, val=l, replace=val, copy=False
                )

        else:  # a numeric column
            X_numeric[col] = X[col]

    return (
        X_numeric,
        np.corrcoef(df[target].values, df_["pseudo_target"].values)[0, 1],
    )
