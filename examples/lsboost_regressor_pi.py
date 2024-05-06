import subprocess
import sys
import os 

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")


subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])

import mlsauce as ms
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from time import time
from os import chdir
from sklearn import metrics

# ridge

print("\n")
print("ridge -----")
print("\n")


dataset = fetch_california_housing()
X = dataset.data
y = dataset.target
# split data into training test and test set
np.random.seed(15029)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2)

obj = ms.LSBoostRegressor(col_sample=0.9, row_sample=0.9)
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(f"Elapsed: {time()-start}")
start = time()
preds = obj.predict(X_test, return_pi=True, method="splitconformal")
print(time()-start)
print(f"splitconformal coverage 1: {np.mean((preds.upper >= y_test)*(preds.lower <= y_test))}")   


obj = ms.LSBoostRegressor(col_sample=0.9, row_sample=0.9,
                          replications=50, 
                          type_pi="bootstrap")
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(f"Elapsed: {time()-start}")
start = time()
preds = obj.predict(X_test, return_pi=True, 
                    method="splitconformal")
print(time()-start)
print(f"splitconformal bootstrap coverage 1: {np.mean((preds.upper >= y_test)*(preds.lower <= y_test))}")   


obj = ms.LSBoostRegressor(col_sample=0.9, row_sample=0.9,
                          replications=50, 
                          type_pi="kde")
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(f"Elapsed: {time()-start}")
start = time()
preds = obj.predict(X_test, return_pi=True, 
                    method="splitconformal")
print(time()-start)
print(f"splitconformal kde coverage 1: {np.mean((preds.upper >= y_test)*(preds.lower <= y_test))}")


dataset = load_diabetes()
X = dataset.data
y = dataset.target
# split data into training test and test set
np.random.seed(15029)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2)

obj = ms.LSBoostRegressor(col_sample=0.9, row_sample=0.9)
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(f"Elapsed: {time()-start}")
start = time()
preds = obj.predict(X_test, return_pi=True, method="splitconformal")
print(time()-start)
print(f"splitconformal coverage 2: {np.mean((preds.upper >= y_test)*(preds.lower <= y_test))}")   


obj = ms.LSBoostRegressor(col_sample=0.9, row_sample=0.9,
                          replications=50, 
                          type_pi="bootstrap")
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(f"Elapsed: {time()-start}")
start = time()
preds = obj.predict(X_test, return_pi=True, 
                    method="splitconformal")
print(time()-start)
print(f"splitconformal bootstrap coverage 2: {np.mean((preds.upper >= y_test)*(preds.lower <= y_test))}")   


obj = ms.LSBoostRegressor(col_sample=0.9, row_sample=0.9,
                          replications=50, 
                          type_pi="kde")
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(f"Elapsed: {time()-start}")
start = time()
preds = obj.predict(X_test, return_pi=True, 
                    method="splitconformal")
print(time()-start)
print(f"splitconformal kde coverage 2: {np.mean((preds.upper >= y_test)*(preds.lower <= y_test))}")



# lasso

print("\n")
print("lasso -----")
print("\n")

print("\n")
print("california housing -----")
print("\n")

dataset = fetch_california_housing()
X = dataset.data
y = dataset.target
# split data into training test and test set
np.random.seed(15029)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2)

obj = ms.LSBoostRegressor(n_estimators=50, solver="lasso", col_sample=0.9, row_sample=0.9)
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(f"Elapsed: {time()-start}")
start = time()
preds = obj.predict(X_test, return_pi=True, method="splitconformal")
print(time()-start)
print(f"splitconformal coverage 3: {np.mean((preds.upper >= y_test)*(preds.lower <= y_test))}")   


obj = ms.LSBoostRegressor(n_estimators=50, solver="lasso", col_sample=0.9, row_sample=0.9,
                          replications=50, 
                          type_pi="bootstrap")
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
preds = obj.predict(X_test, return_pi=True, 
                    method="splitconformal")
print(f"Elapsed: {time()-start}")
print(f"splitconformal bootstrap coverage 3: {np.mean((preds.upper >= y_test)*(preds.lower <= y_test))}")   


obj = ms.LSBoostRegressor(n_estimators=50, solver="lasso", col_sample=0.9, row_sample=0.9,
                          replications=50, 
                          type_pi="kde")
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(f"Elapsed: {time()-start}")
start = time()
preds = obj.predict(X_test, return_pi=True, 
                    method="splitconformal")
print(time()-start)
print(f"splitconformal kde coverage 3: {np.mean((preds.upper >= y_test)*(preds.lower <= y_test))}")


dataset = load_diabetes()
X = dataset.data
y = dataset.target
# split data into training test and test set
np.random.seed(15029)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2)

obj = ms.LSBoostRegressor(n_estimators=50, solver="lasso", reg_lambda=0.002, 
                          col_sample=0.9, row_sample=0.9)
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(f"Elapsed: {time()-start}")
start = time()
preds = obj.predict(X_test, return_pi=True, method="splitconformal")
print(time()-start)
print(f"splitconformal coverage 4: {np.mean((preds.upper >= y_test)*(preds.lower <= y_test))}")   


obj = ms.LSBoostRegressor(n_estimators=10, solver="lasso", col_sample=0.9, row_sample=0.9,
                          replications=50, reg_lambda=0.003, dropout=0.4,
                          type_pi="bootstrap")
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(time()-start)
start = time()
preds = obj.predict(X_test, return_pi=True, 
                    method="splitconformal")
print(time()-start)
print(f"splitconformal bootstrap coverage 4: {np.mean((preds.upper >= y_test)*(preds.lower <= y_test))}")   


obj = ms.LSBoostRegressor(n_estimators=10, solver="lasso", col_sample=0.9, row_sample=0.9,
                          replications=50, reg_lambda=0.001, dropout=0.4,
                          type_pi="kde")
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print(f"Elapsed: {time()-start}")
start = time()
preds = obj.predict(X_test, return_pi=True, 
                    method="splitconformal")
print(time()-start)
print(f"splitconformal kde coverage 4: {np.mean((preds.upper >= y_test)*(preds.lower <= y_test))}")




#Load Boston housing dataset
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)


obj = ms.LSBoostRegressor(n_estimators=50, solver="ridge", reg_lambda=0.002, 
                          col_sample=0.8, row_sample=0.8)
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
preds = obj.predict(X_test, return_pi=True, 
                    level = 90,
                    method="splitconformal")
print(f"Elapsed: {time()-start}")
print(f"splitconformal coverage 5: {np.mean((preds.upper >= y_test)*(preds.lower <= y_test))}")   


obj = ms.LSBoostRegressor(n_estimators=50, solver="lasso", reg_lambda=0.002, 
                          col_sample=0.8, row_sample=0.8)
print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
preds = obj.predict(X_test, return_pi=True, 
                    level = 90, 
                    method="splitconformal")
print(f"Elapsed: {time()-start}")
print(f"splitconformal bootstrap coverage 5: {np.mean((preds.upper >= y_test)*(preds.lower <= y_test))}")   


obj = ms.LSBoostRegressor(n_estimators=50, solver="enet", reg_lambda=0.002, 
                          col_sample=0.8, row_sample=0.8)
obj.fit(X_train, y_train)
preds = obj.predict(X_test, return_pi=True, 
                    level = 90, 
                    method="splitconformal")
print(f"Elapsed: {time()-start}")
print(f"splitconformal bootstrap coverage 6: {np.mean((preds.upper >= y_test)*(preds.lower <= y_test))}")   
