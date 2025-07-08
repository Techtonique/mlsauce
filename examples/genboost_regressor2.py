import subprocess
import sys
import os 

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

import mlsauce as ms
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from time import time
from os import chdir
from sklearn import metrics
from sklearn.datasets import fetch_openml

# Load the dataset from OpenML
boston = fetch_openml(name='boston', version=1, as_frame=True)

# Get the features and target
X = boston.data
y = boston.target

# Display the first few rows
print(X.head())
print(y.head())

np.random.seed(15029)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2)

X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)
y_train = y_train.astype(np.float64)
y_test = y_test.astype(np.float64)

obj = ms.GenericBoostingRegressor(col_sample=0.9, row_sample=0.9)
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
print(np.sqrt(np.mean(np.square(obj.predict(X_test) - y_test))))
print(time()-start)
print(obj.obj['loss'])
