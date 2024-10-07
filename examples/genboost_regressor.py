import subprocess
import sys
import os 

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")


subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])

import mlsauce as ms
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from time import time
from os import chdir
from sklearn import metrics


print("\n")
print("diabetes data -----")

regr = Ridge()

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

obj = ms.GenericBoostingRegressor(regr)
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
print(np.sqrt(np.mean(np.square(obj.predict(X_test) - y_test))))
print(time()-start)
print(obj.obj['loss'])
coeffs = np.asarray([obj.obj['fit_obj_i'][i].coef_ for i in range(len(obj.obj['fit_obj_i']))])
print(coeffs.shape)
print(coeffs)
obj.update(X_test[0,:], y_test[0])
print(coeffs.shape)
print(coeffs)