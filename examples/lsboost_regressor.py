import subprocess
import sys
import os 

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")


subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])

import mlsauce as ms
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from time import time
from os import chdir
from sklearn import metrics

# ridge

print("\n")
print("ridge -----")
print("\n")

# data 2

print("\n")
print("diabetes data -----")

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
# split data into training test and test set
np.random.seed(15029)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2)

obj = ms.LSBoostRegressor(col_sample=0.9, row_sample=0.9)
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
print(np.sqrt(np.mean(np.square(obj.predict(X_test) - y_test))))
print(time()-start)

print(obj.obj['loss'])

obj = ms.LSBoostRegressor(col_sample=0.9, row_sample=0.9, n_clusters=2)
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
print(np.sqrt(np.mean(np.square(obj.predict(X_test) - y_test))))
print(time()-start)

print(obj.obj['loss'])

# MORE DATA NEEDED # MORE DATA NEEDED # MORE DATA NEEDED
obj = ms.LSBoostRegressor(backend="gpu")
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
print(np.sqrt(np.mean(np.square(obj.predict(X_test) - y_test))))
print(time()-start)



# lasso

print("\n")
print("lasso -----")
print("\n")

# data 2
print("\n")
print("diabetes data -----")

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
# split data into training test and test set
np.random.seed(15029)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2)

obj = ms.LSBoostRegressor(solver="lasso")
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
print(np.sqrt(np.mean(np.square(obj.predict(X_test) - y_test))))
print(time()-start)

print(obj.obj['loss'])

obj = ms.LSBoostRegressor(solver="lasso", n_clusters=2)
print(obj.get_params())
obj.fit(X_train, y_train)
print(obj.obj['loss'])

obj = ms.LSBoostRegressor(solver="lasso", n_clusters=2, 
                          degree=2)
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(obj.obj['loss'])

# MORE DATA NEEDED # MORE DATA NEEDED # MORE DATA NEEDED
# obj = ms.LSBoostRegressor(backend="gpu", solver="lasso")
# print(obj.get_params())
# start = time()
# obj.fit(X_train, y_train)
# print(time()-start)
# start = time()
# print(np.sqrt(np.mean(np.square(obj.predict(X_test) - y_test))))
# print(time()-start)






