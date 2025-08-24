# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlsauce as ms
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

# Create synthetic functional time series data
np.random.seed(42)
n_samples = 100
n_points = 50

# Create time points
t = np.linspace(0, 10, n_points)

# Create basis functions
def create_basis_function(t, center, width):
    return np.exp(-(t - center)**2 / (2 * width**2))

# Create synthetic functional data with trend and seasonality
X = np.zeros((n_samples, n_points))
for i in range(n_samples):
    # Trend component
    trend = 0.1 * i * np.sin(0.5 * t)
    
    # Seasonal component
    seasonal = 2 * create_basis_function(t, 3 + 0.05*i, 1.5) + \
               1.5 * create_basis_function(t, 6 + 0.03*i, 1.2)
    
    # Random noise
    noise = 0.3 * np.random.normal(size=n_points)
    
    # Combine components
    X[i] = trend + seasonal + noise

# Split into train and test
train_size = 80
X_train, X_test = X[:train_size], X[train_size:]

# Initialize and fit the forecaster
forecaster = ms.GenericFunctionalForecaster(
    n_components=5,
    reduction_method='pca',
    rolling_window=None,
    forecast_method='ar',
    regressor=Ridge(alpha=1.0),
    regressor_params={'alpha': 0.5}  # This will override the alpha in the regressor
)

# Fit the model
forecaster.fit(X_train)

# Get model information
print("Model Information:")
info = forecaster.get_model_info()
for key, value in info.items():
    print(f"{key}: {value}")

# Plot components
forecaster.plot_components(n_plot=5)

# Plot reduced features
forecaster.plot_reduced_features(n_plot=4)

# Forecast future curves
forecast_steps = len(X_test)
forecasts = forecaster.forecast(steps=forecast_steps)

# Plot forecasts
forecaster.plot_forecast(actual=X_test, steps=forecast_steps)

# Calculate forecast error
mse = mean_squared_error(X_test.flatten(), forecasts.flatten())
print(f"Mean Squared Error: {mse:.4f}")

# Try a different reduction method
print("\nTrying KPCA reduction...")
forecaster_nmf = ms.GenericFunctionalForecaster(
    n_components=5,
    reduction_method='kernel_pca',
    rolling_window=None,
    forecast_method='ar',
    regressor=Ridge(alpha=0.5)
)

forecaster_nmf.fit(X_train)
forecasts_nmf = forecaster_nmf.forecast(steps=forecast_steps)
mse_nmf = mean_squared_error(X_test.flatten(), forecasts_nmf.flatten())
print(f"KPCA Mean Squared Error: {mse_nmf:.4f}")

# Compare with a simple benchmark (last observation)
last_observation_forecast = np.tile(X_train[-1], (forecast_steps, 1))
mse_benchmark = mean_squared_error(X_test.flatten(), last_observation_forecast.flatten())
print(f"Benchmark (last observation) MSE: {mse_benchmark:.4f}")

# Plot comparison
plt.figure(figsize=(12, 6))
plt.plot(X_test.flatten(), 'b-', alpha=0.7, label='Actual')
plt.plot(forecasts.flatten(), 'r--', alpha=0.7, label='PCA Forecast')
plt.plot(forecasts_nmf.flatten(), 'g-.', alpha=0.7, label='NMF Forecast')
plt.plot(last_observation_forecast.flatten(), 'm:', alpha=0.7, label='Last Observation')
plt.title('Forecast Comparison')
plt.xlabel('Time Point Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Try a different reduction method
print("\nTrying KPCA reduction...")
forecaster_nmf = ms.GenericFunctionalForecaster(
    n_components=5,
    reduction_method='kernel_pca',
    rolling_window=10,
    forecast_method='ar',
    regressor=Ridge(alpha=0.5)
)

forecaster_nmf.fit(X_train)
forecasts_nmf = forecaster_nmf.forecast(steps=forecast_steps)
mse_nmf = mean_squared_error(X_test.flatten(), forecasts_nmf.flatten())
print(f"KPCA Mean Squared Error: {mse_nmf:.4f}")

# Compare with a simple benchmark (last observation)
last_observation_forecast = np.tile(X_train[-1], (forecast_steps, 1))
mse_benchmark = mean_squared_error(X_test.flatten(), last_observation_forecast.flatten())
print(f"Benchmark (last observation) MSE: {mse_benchmark:.4f}")
