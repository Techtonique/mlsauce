import os 
import mlsauce as ms 
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from time import time

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

#load_models = [load_breast_cancer, load_iris, load_wine, load_digits]
load_models = [load_breast_cancer, load_iris, load_wine]
#load_models = [load_digits]

for model in load_models: 

    data = model()
    X = data.data
    y= data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 13)

    clf = ms.LazyBoostingClassifier(verbose=0, ignore_warnings=True, #n_jobs=2,
                                    custom_metric=None, preprocess=False)

    start = time()
    models, predictioms = clf.fit(X_train, X_test, y_train, y_test)
    print(f"\nElapsed: {time() - start} seconds\n")

    print(models)

