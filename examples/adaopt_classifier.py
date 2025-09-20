import numpy as np 
from sklearn.datasets import load_digits, load_breast_cancer, load_wine, load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from time import time
from os import chdir
from sklearn import metrics


#wd="/workspace/mlsauce/mlsauce/examples"
#
#chdir(wd)

import mlsauce as ms
import os 

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

print("\n breast cancer ---------- \n")

# data 1
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
# split data into training test and test set
np.random.seed(15029)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2)

#obj = ms.AdaOpt(n_jobs=4, type_dist="euclidean", verbose=1)
obj = ms.AdaOpt()
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
print(obj.score(X_test, y_test))
print(time()-start)

#obj = ms.AdaOpt(n_jobs=4, type_dist="euclidean", verbose=1, 
#                n_clusters_input=2)
obj = ms.AdaOpt()
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
print(obj.score(X_test, y_test))
print(time()-start)

print("\n wine ---------- \n")

# data 2
wine = load_wine()
Z = wine.data
t = wine.target
np.random.seed(879423)
X_train, X_test, y_train, y_test = train_test_split(Z, t, 
                                                    test_size=0.2)

obj = ms.AdaOpt()
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
print(obj.score(X_test, y_test))
print(time()-start)

obj = ms.AdaOpt(n_clusters_input=3)
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
print(obj.score(X_test, y_test))
print(time()-start)

print("\n iris ---------- \n")

# data 3
iris = load_iris()
Z = iris.data
t = iris.target
np.random.seed(734563)
X_train, X_test, y_train, y_test = train_test_split(Z, t, 
                                                    test_size=0.2)


obj = ms.AdaOpt()
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
print(obj.score(X_test, y_test))
print(time()-start)

obj = ms.AdaOpt(n_clusters_input=3)
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
print(obj.score(X_test, y_test))
print(time()-start)


print("\n digits ---------- \n")

# data 4
digits = load_digits()
Z = digits.data
t = digits.target
np.random.seed(13239)
X_train, X_test, y_train, y_test = train_test_split(Z, t, 
                                                    test_size=0.2)

obj = ms.AdaOpt(n_iterations=50,
           learning_rate=0.3,
           reg_lambda=0.1,            
           reg_alpha=0.5,
           eta=0.01,
           gamma=0.01, 
           tolerance=1e-4,
           row_sample=1, 
           k=1, 
           n_jobs=3, type_dist="euclidean", verbose=1)
start = time()
obj.fit(X_train, y_train)
print("Elapsed: ", time()-start)
start = time()
print(obj.score(X_test, y_test))
print("Elapsed: ", time()-start)

# ------ 

obj = ms.AdaOpt(n_iterations=50,
           learning_rate=0.3,
           reg_lambda=0.1,            
           reg_alpha=0.5,
           eta=0.01,
           gamma=0.01, 
           tolerance=1e-4,
           row_sample=1, 
           k=1, backend="cpu",
           n_jobs=3, type_dist="euclidean", verbose=1)
start = time()
obj.fit(X_train, y_train)
print("Elapsed: ", time()-start)
start = time()
print(obj.score(X_test, y_test))
print("Elapsed: ", time()-start)

# ------ 

obj = ms.AdaOpt(n_iterations=50,
           learning_rate=0.3,
           reg_lambda=0.1,            
           reg_alpha=0.5,
           eta=0.01,
           gamma=0.01, 
           tolerance=1e-4,
           row_sample=1, 
           k=1, backend="cpu",
           n_jobs=3, type_dist="cosine", verbose=1)
start = time()
obj.fit(X_train, y_train)
print("Elapsed: ", time()-start)
start = time()
print(obj.score(X_test, y_test))
print("Elapsed: ", time()-start)