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

clf = DecisionTreeRegressor()
clf2 = KernelRidge()

obj = ms.GenericBoostingClassifier(clf, tolerance=1e-2)
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
print(obj.score(X_test, y_test))
print(time()-start)

print(obj.obj['loss'])

obj = ms.GenericBoostingClassifier(clf, tolerance=1e-2, n_clusters=2)
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
print(obj.score(X_test, y_test))
print(time()-start)

print(obj.obj['loss'])


# data 2
print("\n")
print("wine data -----")

wine = load_wine()
Z = wine.data
t = wine.target
np.random.seed(879423)
X_train, X_test, y_train, y_test = train_test_split(Z, t, 
                                                    test_size=0.2)

obj = ms.GenericBoostingClassifier(clf)
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
print(obj.score(X_test, y_test))
print(time()-start)

print(obj.obj['loss'])

obj = ms.GenericBoostingClassifier(clf, n_clusters=3)
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
print(obj.score(X_test, y_test))
print(time()-start)

print(obj.obj['loss'])

# data 3
print("\n")
print("iris data -----")

iris = load_iris()
Z = iris.data
t = iris.target
np.random.seed(734563)
X_train, X_test, y_train, y_test = train_test_split(Z, t, 
                                                    test_size=0.2)


obj = ms.GenericBoostingClassifier(clf)
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
print(obj.score(X_test, y_test))
print(time()-start)

print(obj.obj['loss'])


print("\n")
print("GenericBoosting  KRR -----")
print("\n")

obj = ms.GenericBoostingClassifier(clf2, tolerance=1e-2)
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
print(obj.score(X_test, y_test))
print(time()-start)

print(obj.obj['loss'])

obj = ms.GenericBoostingClassifier(clf2, tolerance=1e-2, n_clusters=2)
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
print(obj.score(X_test, y_test))
print(time()-start)

print(obj.obj['loss'])


# data 2
print("\n")
print("wine data -----")

wine = load_wine()
Z = wine.data
t = wine.target
np.random.seed(879423)
X_train, X_test, y_train, y_test = train_test_split(Z, t, 
                                                    test_size=0.2)

obj = ms.GenericBoostingClassifier(clf2)
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
print(obj.score(X_test, y_test))
print(time()-start)

print(obj.obj['loss'])

obj = ms.GenericBoostingClassifier(clf2, n_clusters=3)
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
print(obj.score(X_test, y_test))
print(time()-start)

print(obj.obj['loss'])

# data 3
print("\n")
print("iris data -----")

iris = load_iris()
Z = iris.data
t = iris.target
np.random.seed(734563)
X_train, X_test, y_train, y_test = train_test_split(Z, t, 
                                                    test_size=0.2)


obj = ms.GenericBoostingClassifier(clf2)
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
print(obj.score(X_test, y_test))
print(time()-start)

print(obj.obj['loss'])

