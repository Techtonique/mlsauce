import pandas as pd
import requests


def download(
    pkgname="MASS",
    dataset="Boston",
    source="https://cran.r-universe.dev/",
    **kwargs
):
    URL = source + pkgname + "/data/" + dataset + "/json"
    res = requests.get(URL)
    return pd.DataFrame(res.json(), **kwargs)
