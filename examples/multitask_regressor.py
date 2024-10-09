import mlsauce as ms
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import HuberRegressor, LinearRegression

data = load_iris()
X = data.data
y = data.target
y_encoded = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()

obj = ms.MultiTaskRegressor(regr=HuberRegressor())

obj.fit(X, y_encoded)

print(obj.predict(X))

obj2 = LinearRegression()

obj2.fit(X, y_encoded)

print(obj2.predict(X))