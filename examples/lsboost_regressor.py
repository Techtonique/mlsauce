import numpy as np 
from sklearn.datasets import load_boston, load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from time import time
from os import chdir
from sklearn import metrics


#wd="/workspace/mlsauce/mlsauce/examples"
#
#chdir(wd)

import mlsauce as ms

# ridge

print("\n")
print("ridge -----")
print("\n")

# data 1
boston = load_boston()
X = boston.data
y = boston.target
# split data into training test and test set
np.random.seed(15029)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2)

obj = ms.LSBoostRegressor()
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
print(obj)
start = time()
print(np.sqrt(np.mean(np.square(obj.predict(X_test) - y_test))))
print(time()-start)

# MORE DATA NEEDED # MORE DATA NEEDED # MORE DATA NEEDED
obj = ms.LSBoostRegressor(backend="gpu")
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
print(obj)
start = time()
print(np.sqrt(np.mean(np.square(obj.predict(X_test) - y_test))))
print(time()-start)

# data 2
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
# split data into training test and test set
np.random.seed(15029)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2)

obj = ms.LSBoostRegressor()
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
print(np.sqrt(np.mean(np.square(obj.predict(X_test) - y_test))))
print(time()-start)

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

# data 1
boston = load_boston()
X = boston.data
y = boston.target
# split data into training test and test set
np.random.seed(15029)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2)

obj = ms.LSBoostRegressor(solver="lasso")
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
print(obj)
start = time()
print(np.sqrt(np.mean(np.square(obj.predict(X_test) - y_test))))
print(time()-start)

# MORE DATA NEEDED # MORE DATA NEEDED # MORE DATA NEEDED
# obj = ms.LSBoostRegressor(backend="gpu", solver="lasso")
# print(obj.get_params())
# start = time()
# obj.fit(X_train, y_train)
# print(time()-start)
# print(obj)
# start = time()
# print(np.sqrt(np.mean(np.square(obj.predict(X_test) - y_test))))
# print(time()-start)

# data 2
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

# MORE DATA NEEDED # MORE DATA NEEDED # MORE DATA NEEDED
# obj = ms.LSBoostRegressor(backend="gpu", solver="lasso")
# print(obj.get_params())
# start = time()
# obj.fit(X_train, y_train)
# print(time()-start)
# start = time()
# print(np.sqrt(np.mean(np.square(obj.predict(X_test) - y_test))))
# print(time()-start)






