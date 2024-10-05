import os 
import mlsauce as ms
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

data = load_diabetes()
X = data.data
y= data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 123)

regr = ms.LazyBoostingRegressor(verbose=0, ignore_warnings=True, 
                                custom_metric=None, preprocess=True)
models, predictioms = regr.fit(X_train, X_test, y_train, y_test)
model_dictionary = regr.provide_models(X_train, X_test, y_train, y_test)
print(models)