from sklearn.datasets import load_digits, load_breast_cancer, load_wine, load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import numpy as np 
from time import time
from os import chdir

wd="/Users/moudiki/Documents/Python_Packages/mlsauce"

chdir(wd)

import mlsauce as ms


# data 1
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
# split data into training test and test set
np.random.seed(15029)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2)

print(X_train.shape)
print(X_test.shape)


obj = ms.AdaOpt()
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
print(obj.score(X_test, y_test))
print(time()-start)


# data 2
wine = load_wine()
Z = wine.data
t = wine.target
np.random.seed(879423)
X_train, X_test, y_train, y_test = train_test_split(Z, t, 
                                                    test_size=0.2)

print(X_train.shape)
print(X_test.shape)


obj = ms.AdaOpt()
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
print(obj.score(X_test, y_test))
print(time()-start)


# data 3
iris = load_iris()
Z = iris.data
t = iris.target
np.random.seed(734563)
X_train, X_test, y_train, y_test = train_test_split(Z, t, 
                                                    test_size=0.2)

print(X_train.shape)
print(X_test.shape)


obj = ms.AdaOpt()
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
print(obj.score(X_test, y_test))
print(time()-start)


# data 4
digits = load_digits()
Z = digits.data
t = digits.target
np.random.seed(13239)
X_train, X_test, y_train, y_test = train_test_split(Z, t, 
                                                    test_size=0.2)

print(X_train.shape)
print(X_test.shape)

obj = ms.AdaOpt(n_iterations=50,
           learning_rate=0.3,
           reg_lambda=0.1,            
           reg_alpha=0.5,
           eta=0.01,
           gamma=0.01, 
           tolerance=1e-4)
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
print(obj.score(X_test, y_test))
print(time()-start)

# with clustering
obj = ms.AdaOpt(n_clusters=25, k=1)
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
print(obj.score(X_test, y_test))
print(time()-start)