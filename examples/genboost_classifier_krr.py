import numpy as np 
from sklearn.datasets import load_digits, load_breast_cancer, load_wine, load_iris, load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from time import time
from os import chdir
from sklearn import metrics
import os 

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

import mlsauce as ms

#ridge

print("\n")
print("GenericBoosting KernelRidge -----")
print("\n")

# data 1
breast_cancer = load_wine() #load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
# split data into training test and test set
np.random.seed(15029)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2)

clf = ms.KRLSRegressor()
obj = ms.GenericBoostingClassifier(clf)

print(obj.get_params())
start = time()
obj.fit(X_train, y_train)
print("Elapsed", time()-start)

pred = obj.predict(X_test)
print("Accuracy", metrics.accuracy_score(y_test, pred))