import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml, load_iris, load_digits, load_wine, load_breast_cancer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from time import time
from mlsauce.conformalbayesian import ConformalBayesianRegressor

# -------------------------------
# Load Boston dataset
# -------------------------------
boston = fetch_openml(name="boston", version=1, as_frame=True)
X, y = boston.data.to_numpy(), boston.target.to_numpy().astype(float)
X, y = shuffle(X, y, random_state=0)

# Outer train/test split
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# RandomForest Regressor
# -------------------------------
rf_cb = ConformalBayesianRegressor(
    obj=RandomForestRegressor(),
    hyperparameter_bounds={"n_estimators": [50, 200], "max_depth": [5, 20]},
    n_samples=10,
    #calibration_fraction=0.5,
    scaling_method="standard",
    random_state=42,
    verbose=True,
    n_jobs=-1
)

start = time()
rf_cb.fit(X_train_full, y_train_full)
elapsed_rf = time() - start
print("Elapsed: ", elapsed_rf)
preds_rf = rf_cb.predict(X_test, return_pi=True)
print("preds:", preds_rf)
coverage_rf = rf_cb.get_coverage(y_test, preds_rf.lower, preds_rf.upper)
mse_rf = mean_squared_error(y_test, preds_rf.mean)

# -------------------------------
# Ridge Regressor
# -------------------------------
ridge_cb = ConformalBayesianRegressor(
    obj=Ridge(),
    hyperparameter_bounds={"alpha": [0.01, 100.0]},
    n_samples=20,
    #calibration_fraction=0.5,
    scaling_method="standard",
    random_state=42,
    verbose=True,
    n_jobs=-1
)

start = time()
ridge_cb.fit(X_train_full, y_train_full)
elapsed_ridge = time() - start
print("Elapsed: ", elapsed_ridge)
preds_ridge = ridge_cb.predict(X_test, return_pi=True)
print("preds:", preds_ridge)
coverage_ridge = ridge_cb.get_coverage(y_test, preds_ridge.lower, preds_ridge.upper)
mse_ridge = mean_squared_error(y_test, preds_ridge.mean)

# -------------------------------
# Print Results
# -------------------------------
print("=== Boston Housing: Conformal Prediction (Split, GMM stratify) ===")
print(f"RandomForest - RMSE: {np.sqrt(mse_rf)}, Coverage: {coverage_rf:.3f}, Time: {elapsed_rf:.2f}s")
print(f"Ridge        - RMSE: {np.sqrt(mse_ridge)}, Coverage: {coverage_ridge:.3f}, Time: {elapsed_ridge:.2f}s")

# -------------------------------
# Plot prediction intervals
# -------------------------------
plt.figure(figsize=(12,5))

# RandomForest
plt.subplot(1,2,1)
order = np.argsort(preds_rf.mean)
plt.fill_between(np.arange(len(order)), preds_rf.lower[order], preds_rf.upper[order],
                 color='lightblue', alpha=0.4, label='90% interval')
plt.plot(preds_rf.mean[order], 'b-', label='Predicted')
plt.plot(y_test[order], 'r.', label='True')
plt.title(f"RandomForest: coverage={coverage_rf:.3f}")
plt.xlabel("Sorted samples")
plt.ylabel("Predicted / Interval")
plt.legend()

# Ridge
plt.subplot(1,2,2)
order = np.argsort(preds_ridge.mean)
plt.fill_between(np.arange(len(order)), preds_ridge.lower[order], preds_ridge.upper[order],
                 color='lightgreen', alpha=0.4, label='90% interval')
plt.plot(preds_ridge.mean[order], 'g-', label='Predicted')
plt.plot(y_test[order], 'r.', label='True')
plt.title(f"Ridge: coverage={coverage_ridge:.3f}")
plt.xlabel("Sorted samples")
plt.ylabel("Predicted / Interval")
plt.legend()

plt.tight_layout()
plt.show()

# End of example

# Classification example with Iris dataset

# Load Iris dataset
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
X_iris, y_iris = shuffle(X_iris, y_iris, random_state=0)
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42
)
from mlsauce.conformalbayesian import ConformalBayesianClassifier
# Conformal Bayesian Classifier with Ridge base estimator
cb_clf = ConformalBayesianClassifier(
    obj=RandomForestClassifier(),
    hyperparameter_bounds={"n_estimators": [50, 200], "max_depth": [5, 20]},
    n_samples=10,
   # calibration_fraction=0.5,
    scaling_method="standard",
    random_state=42,
    verbose=True,
    n_jobs=-1
)
cb_clf.fit(X_train_iris, y_train_iris)
preds_iris = cb_clf.predict(X_test_iris)
accuracy_iris = np.mean(preds_iris == y_test_iris)
print(f"Iris Classification Accuracy: {accuracy_iris:.3f}")

# Breast Cancer dataset
from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer()
X_bc, y_bc = breast_cancer.data, breast_cancer.target
X_bc, y_bc = shuffle(X_bc, y_bc, random_state=0)
X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(
    X_bc, y_bc, test_size=0.2, random_state=42
)
cb_clf_bc = ConformalBayesianClassifier(
    obj=RandomForestClassifier(),
    hyperparameter_bounds={"n_estimators": [50, 200], "max_depth": [5, 20]},
    n_samples=10,
    #calibration_fraction=0.5,
    scaling_method="standard",
    random_state=42,
    verbose=True,
    n_jobs=-1
)
cb_clf_bc.fit(X_train_bc, y_train_bc)
preds_bc = cb_clf_bc.predict(X_test_bc)
accuracy_bc = np.mean(preds_bc == y_test_bc)
print(f"Breast Cancer Classification Accuracy: {accuracy_bc:.3f}")

# Wine Quality dataset
from sklearn.datasets import load_wine
wine = load_wine()
X_wine, y_wine = wine.data, wine.target
X_wine, y_wine = shuffle(X_wine, y_wine, random_state=0)
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(
    X_wine, y_wine, test_size=0.2, random_state=42
)
cb_clf_wine = ConformalBayesianClassifier(
    obj=RandomForestClassifier(),
    hyperparameter_bounds={"n_estimators": [50, 200], "max_depth": [5, 20]},
    n_samples=10,
    #calibration_fraction=0.5,
    scaling_method="standard",
    random_state=42,
    verbose=True,
    n_jobs=-1
)
cb_clf_wine.fit(X_train_wine, y_train_wine)
preds_wine = cb_clf_wine.predict(X_test_wine)
accuracy_wine = np.mean(preds_wine == y_test_wine)
print(f"Wine Quality Classification Accuracy: {accuracy_wine:.3f}")

# Digits dataset
from sklearn.datasets import load_digits
digits = load_digits()
X_digits, y_digits = digits.data, digits.target
X_digits, y_digits = shuffle(X_digits, y_digits, random_state=0)
X_train_digits, X_test_digits, y_train_digits, y_test_digits = train_test_split(
    X_digits, y_digits, test_size=0.2, random_state=42
)
cb_clf_digits = ConformalBayesianClassifier(
    obj=RandomForestClassifier(),
    hyperparameter_bounds={"n_estimators": [50, 200], "max_depth": [5, 20]},
    n_samples=10,
    #calibration_fraction=0.5,
    scaling_method="standard",
    random_state=42,
    verbose=True,
    n_jobs=-1
)
cb_clf_digits.fit(X_train_digits, y_train_digits)
preds_digits = cb_clf_digits.predict(X_test_digits)
accuracy_digits = np.mean(preds_digits == y_test_digits)
print(f"Digits Classification Accuracy: {accuracy_digits:.3f}")

