import mlsauce as ms
import numpy as np 
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from time import time
from os import chdir
from sklearn import metrics

# data 2
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
# split data into training test and test set
np.random.seed(15029)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2)

obj = ms.LassoRegressor(backend="cpu")
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
print(np.sqrt(np.mean(np.square(obj.predict(X_test) - y_test))))
print(time()-start)
print(obj.beta)

# obj = ms.LassoRegressor(backend="gpu")
# print(obj.get_params())
# start = time()
# obj.fit(X_train, y_train)
# print(time()-start)
# start = time()
# print(np.sqrt(np.mean(np.square(obj.predict(X_test) - y_test))))
# print(time()-start)

#lam=0.1
#[  0.27217132 -10.9310058   25.15384023  15.9590922  -40.48397308
#  20.27425445  12.63036315  18.86586336  33.4090342    3.43798249]
#lam=10
#[  0.26883716 -10.88809883  25.16445231  15.93114765 -38.24113256
#  18.54837126  11.54861804  18.491766    32.58150795   3.43506438]
#lam=100
#[  0.23852655 -10.4980356   25.2609257   15.67710642 -17.85168498
#   2.85853324   1.71457677  15.09088226  25.05854651   3.40853606]