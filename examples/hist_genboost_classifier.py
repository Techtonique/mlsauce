import numpy as np 
from sklearn.datasets import load_digits, load_breast_cancer, load_wine, load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from time import time
from os import chdir
from sklearn import metrics
import os 

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

print(os.path.relpath(os.path.dirname(__file__)))

#wd="/workspace/mlsauce/mlsauce/examples"
#
#chdir(wd)

import mlsauce as ms

#ridge

print("\n")
print("GenericBoosting Decision tree -----")
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

clf = ExtraTreeRegressor()
clf2 = LinearRegression()

obj = ms.GenericBoostingClassifier(clf, hist=True)
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
print(obj.score(X_test, y_test))
print(time()-start)

print(obj.obj['loss'])

print(obj.obj['fit_obj_i'])

