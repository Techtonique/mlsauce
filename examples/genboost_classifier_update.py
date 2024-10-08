import numpy as np 
from sklearn.datasets import load_digits, load_breast_cancer, load_wine, load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge, LinearRegression, RidgeCV, ElasticNetCV, LassoCV, Lasso
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

eta = 0.9


# data 1
regr1 = Ridge()


print("\n Example 1 --------------------------- \n")
wine = load_wine()
X = wine.data
y = wine.target
# split data into training test and test set
np.random.seed(15029)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2)

obj = ms.GenericBoostingClassifier(regr1)
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
print(f"obj.obj['fit_obj_i'][0]: {obj.obj['fit_obj_i'][0].coef_}")
start = time()
print(f"score init. {obj.score(X_test[3:,], 
                y_test[3:])}")
print(time()-start)
obj = obj.update(X_test[0,:], y_test[0], eta=eta)
obj = obj.update(X_test[1,:], y_test[1], eta=eta)
obj = obj.update(X_test[2,:], y_test[2], eta=eta)
print(f"score updated: {obj.score(X_test[3:,], 
                y_test[3:])}")
print(f"obj.obj['fit_obj_i'][0].coef_: {obj.obj['fit_obj_i'][0].coef_}")

print("\n Example 2 --------------------------- \n")

eta = 0.3

regr2 = Ridge()

dataset = load_iris()
X = dataset.data
y = dataset.target
# split data into training test and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=13)

obj = ms.GenericBoostingClassifier(regr2)
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
print(f"obj.obj['fit_obj_i'][0]: {obj.obj['fit_obj_i'][0].coef_}")
print(f"obj.obj['Ym']: {obj.obj['Ym']}")
start = time()
print(f"score init. {obj.score(X_test[5:,], 
                y_test[5:])}")
print(time()-start)
obj = obj.update(X_test[0,:], y_test[0], eta=eta)
obj = obj.update(X_test[1,:], y_test[1], eta=eta)
obj = obj.update(X_test[2,:], y_test[2], eta=eta)
obj = obj.update(X_test[3,:], y_test[3], eta=eta)
obj = obj.update(X_test[4,:], y_test[4], eta=eta)
print(f"score updated: {obj.score(X_test[5:,], 
                y_test[5:])}")
print(f"obj.obj['fit_obj_i'][0].coef_: {obj.obj['fit_obj_i'][0].coef_}")
print(f"obj.obj['Ym']: {obj.obj['Ym']}")
