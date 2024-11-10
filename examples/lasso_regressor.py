import mlsauce as ms
import numpy as np 
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from time import time
from os import chdir
from sklearn import metrics
import os 

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")


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
print("Elapsed: ", time()-start)
start = time()
print(np.sqrt(np.mean(np.square(obj.predict(X_test) - y_test))))
print("Elapsed: ", time()-start)
print(obj.beta)

obj = ms.LassoRegressor(backend="cpu")
start = time()
obj.fit(X_train, y_train)
print("Elapsed: ", time()-start)
start = time()
print(np.sqrt(np.mean(np.square(obj.predict(X_test) - y_test))))
print("Elapsed: ", time()-start)
print(obj.beta)

