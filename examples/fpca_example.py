#!/usr/bin/env python3
"""
Example demonstrating the use of GenericFunctionalForecaster.

This example shows how to use the GenericFunctionalForecaster class for
functional time series forecasting with different PCA variants and regression methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor

# Import the GenericFunctionalForecaster from mlsauce
try:
    from mlsauce.fpca import GenericFunctionalForecaster
except ImportError:
    print("Please install mlsauce first: pip install mlsauce")
    exit(1)


def generate_functional_data(n_samples=100, n_points=50, noise=0.1):
    """Generate synthetic functional time series data."""
    np.random.seed(42)
    
    # Create time grid
    grid = np.linspace(0, 10, n_points)
    
    # Generate functional data with time-varying patterns
    X = np.zeros((n_samples, n_points))
    
    for i in range(n_samples):
        # Base pattern with time-varying amplitude and phase
        amplitude = 1 + 0.5 * np.sin(i * 0.1)  # Time-varying amplitude
        phase = i * 0.05  # Time-varying phase
        frequency = 1 + 0.2 * np.sin(i * 0.02)  # Time-varying frequency
        
        # Generate functional curve
        curve = amplitude * np.sin(frequency * grid + phase) + \
                0.3 * np.sin(2 * frequency * grid + phase) + \
                0.1 * np.sin(3 * frequency * grid + phase)
        
        # Add noise
        X[i, :] = curve + noise * np.random.randn(n_points)
    
    return X, grid


def plot_functional_data(X, grid, title="Functional Time Series Data"):
    """Plot functional time series data."""
    plt.figure(figsize=(12, 6))
    
    # Plot all curves
    for i in range(min(20, len(X))):  # Plot first 20 curves
        plt.plot(grid, X[i, :], alpha=0.3, linewidth=1)
    
    # Plot mean curve
    mean_curve = X.mean(axis=0)
    plt.plot(grid, mean_curve, 'r-', linewidth=3, label='Mean Curve')
    
    plt.title(title)
    plt.xlabel('Domain')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def compare_forecasting_methods(X, grid, n_components=5):
    """Compare different forecasting methods."""
    print("Comparing different forecasting methods...")
    print("=" * 50)
    
    # Split data into training and testing
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    
    # Test different configurations
    configurations = [
        {
            'name': 'PCA + Ridge',
            'pca_method': 'pca',
            'regression_method': 'ridge',
            'pca_params': {},
            'regression_params': {'alphas': np.logspace(-3, 3, 10)}
        },
        {
            'name': 'Kernel PCA + Linear',
            'pca_method': 'kernel_pca',
            'regression_method': 'linear',
            'pca_params': {},
            'regression_params': {},
            'kernel_params': {'kernel': 'rbf'}
        },
        {
            'name': 'Sparse PCA + Lasso',
            'pca_method': 'sparse_pca',
            'regression_method': 'lasso',
            'pca_params': {'alpha': 0.1},
            'regression_params': {'alphas': np.logspace(-3, 3, 10)}
        }
    ]
    
    results = {}
    
    for config in configurations:
        print(f"\nTesting: {config['name']}")
        print("-" * 30)
        
        # Create forecaster
        forecaster = GenericFunctionalForecaster(
            n_components=n_components,
            pca_method=config['pca_method'],
            regression_method=config['regression_method'],
            pca_params=config.get('pca_params', {}),
            regression_params=config.get('regression_params', {}),
            kernel_params=config.get('kernel_params', {})
        )
        
        # Fit the model
        forecaster.fit(X_train, grid)
        
        # Get model info
        info = forecaster.get_model_info()
        print(f"Model info: {info}")
        
        # Test both forecasting methods
        for method in ['ar', 'coef_ar']:
            try:
                # Forecast
                forecast_steps = len(X_test)
                forecasts = forecaster.forecast(steps=forecast_steps, method=method)
                
                # Calculate error
                mse = np.mean((X_test - forecasts) ** 2)
                mae = np.mean(np.abs(X_test - forecasts))
                
                print(f"  {method.upper()} method - MSE: {mse:.4f}, MAE: {mae:.4f}")
                
                # Store results
                key = f"{config['name']}_{method}"
                results[key] = {
                    'forecasts': forecasts,
                    'mse': mse,
                    'mae': mae,
                    'forecaster': forecaster
                }
                
            except Exception as e:
                print(f"  {method.upper()} method failed: {e}")
    
    return results, X_test


def plot_forecast_comparison(results, X_test, grid):
    """Plot forecast comparison."""
    plt.figure(figsize=(15, 10))
    
    # Plot actual test data
    plt.subplot(2, 2, 1)
    for i in range(min(5, len(X_test))):
        plt.plot(grid, X_test[i, :], 'k-', alpha=0.7, linewidth=1)
    plt.title('Actual Test Data')
    plt.xlabel('Domain')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    
    # Plot forecasts for different methods
    plot_idx = 2
    for key, result in results.items():
        if plot_idx <= 4:
            plt.subplot(2, 2, plot_idx)
            
            # Plot actual (first curve)
            plt.plot(grid, X_test[0, :], 'k-', linewidth=2, label='Actual')
            
            # Plot forecast (first curve)
            plt.plot(grid, result['forecasts'][0, :], 'r--', linewidth=2, label='Forecast')
            
            plt.title(f'{key}\nMSE: {result["mse"]:.4f}')
            plt.xlabel('Domain')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plot_idx += 1
    
    plt.tight_layout()
    plt.show()


def demonstrate_components_and_scores(X, grid):
    """Demonstrate component and score visualization."""
    print("\nDemonstrating component and score visualization...")
    print("=" * 50)
    
    # Create forecaster with PCA
    forecaster = GenericFunctionalForecaster(
        n_components=6,
        pca_method='pca',
        regression_method='ridge'
    )
    
    # Fit the model
    forecaster.fit(X, grid)
    
    # Plot components
    forecaster.plot_components(n_plot=3)
    
    # Plot scores
    forecaster.plot_scores(n_plot=4)
    
    # Get model info
    info = forecaster.get_model_info()
    print(f"Model information: {info}")


def main():
    """Main function demonstrating GenericFunctionalForecaster usage."""
    print("GenericFunctionalForecaster Example")
    print("=" * 50)
    
    # Generate functional data
    X, grid = generate_functional_data(n_samples=100, n_points=50, noise=0.1)
    
    print(f"Generated functional data: {X.shape}")
    print(f"Grid points: {len(grid)}")
    print()
    
    # Plot the data
    plot_functional_data(X, grid)
    
    # Compare forecasting methods
    results, X_test = compare_forecasting_methods(X, grid, n_components=5)
    
    # Plot forecast comparison
    plot_forecast_comparison(results, X_test, grid)
    
    # Demonstrate components and scores
    demonstrate_components_and_scores(X, grid)
    
    print("\nExample completed!")


if __name__ == "__main__":
    main() 