import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
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
    model_class=RandomForestRegressor,
    n_samples=20,
    calibration_fraction=0.5,
    scaling_method="standard",
    random_state=42,
    verbose=True,
    n_jobs=-1
)

start = time()
rf_cb.fit(X_train_full, y_train_full)
print("Elapsed: ", time() - start)
y_pred_rf, lower_rf, upper_rf = rf_cb.predict_interval(X_test, confidence=0.95)
coverage_rf = rf_cb.get_coverage(y_test, lower_rf, upper_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)

# -------------------------------
# Ridge Regressor
# -------------------------------
ridge_cb = ConformalBayesianRegressor(
    model_class=Ridge(),
    hyperparameter_bounds={"alpha": [0.01, 100.0]},
    n_samples=20,
    calibration_fraction=0.5,
    scaling_method="standard",
    random_state=42,
    verbose=True,
    n_jobs=-1
)

start = time()
ridge_cb.fit(X_train_full, y_train_full)
print("Elapsed: ", time() - start)
y_pred_ridge, lower_ridge, upper_ridge = ridge_cb.predict_interval(X_test, confidence=0.95)
coverage_ridge = ridge_cb.get_coverage(y_test, lower_ridge, upper_ridge)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)

# -------------------------------
# Print Results
# -------------------------------
print("=== Boston Housing: Conformal Prediction (Split, GMM stratify) ===")
print(f"RandomForest - RMSE: {np.sqrt(mse_rf)}, Coverage: {coverage_rf:.3f}")
print(f"Ridge        - RMSE: {np.sqrt(mse_ridge)}, Coverage: {coverage_ridge:.3f}")

# -------------------------------
# Plot prediction intervals
# -------------------------------
plt.figure(figsize=(12,5))

# RandomForest
plt.subplot(1,2,1)
order = np.argsort(y_pred_rf)
plt.fill_between(np.arange(len(order)), lower_rf[order], upper_rf[order],
                 color='lightblue', alpha=0.4, label='90% interval')
plt.plot(y_pred_rf[order], 'b-', label='Predicted')
plt.plot(y_test[order], 'r.', label='True')
plt.title(f"RandomForest: coverage={coverage_rf:.3f}")
plt.xlabel("Sorted samples")
plt.ylabel("Predicted / Interval")
plt.legend()

# Ridge
plt.subplot(1,2,2)
order = np.argsort(y_pred_ridge)
plt.fill_between(np.arange(len(order)), lower_ridge[order], upper_ridge[order],
                 color='lightgreen', alpha=0.4, label='90% interval')
plt.plot(y_pred_ridge[order], 'g-', label='Predicted')
plt.plot(y_test[order], 'r.', label='True')
plt.title(f"Ridge: coverage={coverage_ridge:.3f}")
plt.xlabel("Sorted samples")
plt.ylabel("Predicted / Interval")
plt.legend()

plt.tight_layout()
plt.show()