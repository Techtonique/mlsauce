import os 
import mlsauce as ms
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

data = load_diabetes()
X = data.data
y= data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 123)

regr = ms.LazyBoostingRegressor(verbose=0, ignore_warnings=True, #n_jobs=2,
                                custom_metric=None, preprocess=True)
models, predictioms = regr.fit(X_train, X_test, y_train, y_test)
model_dictionary = regr.provide_models(X_train, X_test, y_train, y_test)
print(models)

data = fetch_california_housing()
X = data.data[0:1000,:]
y= data.target[0:1000]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 123)

regr = ms.LazyBoostingRegressor(verbose=0, ignore_warnings=True, 
                                custom_metric=None, preprocess=True)
models, predictioms = regr.fit(X_train, X_test, y_train, y_test, hist=True)
model_dictionary = regr.provide_models(X_train, X_test, y_train, y_test)
print(models)


from sklearn.datasets import fetch_openml

# Load the dataset from OpenML
boston = fetch_openml(name='boston', version=1, as_frame=True)

# Get the features and target
X = boston.data
y = boston.target

# Display the first few rows
print(X.head())
print(y.head())

np.random.seed(1509)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2)

X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)
y_train = y_train.astype(np.float64)
y_test = y_test.astype(np.float64)

regr = ms.LazyBoostingRegressor(verbose=0, ignore_warnings=True, #n_jobs=2,
                                custom_metric=None, preprocess=True)
models, predictioms = regr.fit(X_train, X_test, y_train, y_test, hist=True)
model_dictionary = regr.provide_models(X_train, X_test, y_train, y_test)
print(models)
