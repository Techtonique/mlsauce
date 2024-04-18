# Authors: Thierry Moudiki
#
# License: BSD 3
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def cluster(X, n_clusters=None, 
            method="kmeans", 
            type_scaling = "standard",
            training=True, 
            scaler=None,
            label_encoder=None,
            clusterer=None,
            seed=123):
    
    assert method in ("kmeans", "gmm"), "method must be in ('kmeans', 'gmm')"
    assert type_scaling in ("standard", "minmax", "robust"), "type_scaling must be in ('standard', 'minmax', 'robust')"

    if training: 
        assert n_clusters is not None, "n_clusters must be provided at training time"
        if type_scaling == "standard":
            scaler = StandardScaler()            
        elif type_scaling == "minmax":
            scaler = MinMaxScaler()
        elif type_scaling == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError("type_scaling must be in ('standard', 'minmax', 'robust')")
        
        scaled_X = scaler.fit_transform(X)
        label_encoder = OneHotEncoder(handle_unknown='ignore')
        
        if method == "kmeans":            
            clusterer = KMeans(n_clusters=n_clusters, 
                               random_state=seed, 
                               n_init="auto").fit(scaled_X)        
            res = label_encoder.fit_transform(clusterer.labels_.reshape(-1, 1)).toarray()                
        elif method == "gmm":            
            clusterer = GaussianMixture(n_components=n_clusters, 
                                        random_state=seed).fit(scaled_X)            
            res = label_encoder.fit_transform(clusterer.predict(scaled_X).reshape(-1, 1)).toarray()
        else:
            raise ValueError("method must be in ('kmeans', 'gmm')")            
        
        return res, scaler, label_encoder, clusterer
        
    else: # @ inference time

        assert scaler is not None, "scaler must be provided at inferlabel_encodere time"
        assert label_encoder is not None, "label_encoder must be provided at inferlabel_encodere time"
        assert clusterer is not None, "clusterer must be provided at inferlabel_encodere time"
        scaled_X = scaler.transform(X)        
        
        return label_encoder.transform(clusterer.predict(scaled_X).reshape(-1, 1)).toarray() 

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
