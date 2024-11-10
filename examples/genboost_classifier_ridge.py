import numpy as np 
from sklearn.datasets import load_digits, load_breast_cancer, load_wine, load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from time import time
from os import chdir
from sklearn import metrics
import os 

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

import mlsauce as ms

#ridge

print("\n")
print("GenericBoosting Ridge -----")
print("\n")

print("\n")
print("breast_cancer data -----")

# data 1
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
# split data into training test and test set
np.random.seed(15029)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2)

clf = ms.RidgeRegressor(reg_lambda=0.05)
clf2 = ms.RidgeRegressor(reg_lambda=0.05, backend="gpu")

obj = ms.GenericBoostingClassifier(clf)
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
print(obj.score(X_test, y_test))
print(time()-start)

print(obj.obj['loss'])

print(obj.obj['fit_obj_i'])

# needs more data 
obj = ms.GenericBoostingClassifier(clf2, backend="gpu")
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
print(obj.score(X_test, y_test))
print(time()-start)

print(obj.obj['loss'])

print(obj.obj['fit_obj_i'])

