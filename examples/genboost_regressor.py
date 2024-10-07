import subprocess
import sys
import os 

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")


subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])

import mlsauce as ms
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from time import time
from os import chdir
from sklearn import metrics


print("\n")
print("diabetes data -----")

regr = LinearRegression()

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
# split data into training test and test set
np.random.seed(15029)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2)

print("\n Example 1 --------------------------- \n")

obj = ms.GenericBoostingRegressor(regr, col_sample=0.9, row_sample=0.9)
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
print(np.sqrt(np.mean(np.square(obj.predict(X_test) - y_test))))
print(time()-start)
print(obj.obj['loss'])

print("\n Example 2 --------------------------- \n")

obj = ms.GenericBoostingRegressor(regr, col_sample=0.9, row_sample=0.9, n_clusters=2)
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
print(np.sqrt(np.mean(np.square(obj.predict(X_test) - y_test))))
print(time()-start)
print(obj.obj['loss'])
print(obj.obj['fit_obj_i'])

print("\n Example 3 --------------------------- \n")

housing = fetch_california_housing()
n_samples = 500
X = housing.data[:n_samples]
y = housing.target[:n_samples]
# split data into training test and test set
np.random.seed(15029)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2)
obj = ms.GenericBoostingRegressor(regr, n_hidden_features=2, n_estimators=10)


obj.fit(X_train, y_train)
print(f"obj.obj['xm']: {obj.obj['xm']}")
print(f"obj.obj['xsd']: {obj.obj['xsd']}")
print(f"obj.obj['fit_obj_i'][0]: {obj.obj['fit_obj_i'][0].coef_}")
print(f"score: {np.sqrt(np.mean(np.square(obj.predict(X_test[3:10,:]) - y_test[3:10])))}")

obj = obj.update(X_test[0,:], y_test[0])
obj = obj.update(X_test[1,:], y_test[1])
obj = obj.update(X_test[2,:], y_test[2])
print(f"obj.obj['xm']: {obj.obj['xm']}")
print(f"obj.obj['xsd']: {obj.obj['xsd']}")
print(f"obj.obj['fit_obj_i'][0]: {obj.obj['fit_obj_i'][0].coef_}")
print(f"obj: {obj}")
print(f"preds: {obj.predict(X_test[3:10,:])}")
print(f"score: {np.sqrt(np.mean(np.square(obj.predict(X_test[3:10,:]) - y_test[3:10])))}")