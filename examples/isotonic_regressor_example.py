#!/usr/bin/env python3
"""
Example demonstrating the use of IsotonicRegressor.

This example shows how to use the IsotonicRegressor class which takes a base regressor
and applies isotonic regression as postprocessing in the predict method.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Import the IsotonicRegressor from mlsauce
try:
    from mlsauce.isotonicregressor import IsotonicRegressor
except ImportError:
    print("Please install mlsauce first: pip install mlsauce")
    exit(1)


def generate_non_monotonic_data(n_samples=1000, noise=0.1):
    """Generate synthetic data with non-monotonic relationship."""
    np.random.seed(42)
    X = np.random.uniform(0, 10, n_samples).reshape(-1, 1)
    
    # Create a non-monotonic relationship
    y = 2 * X.flatten() + 0.5 * np.sin(X.flatten()) + noise * np.random.randn(n_samples)
    
    return X, y


def plot_results(X_train, X_test, y_train, y_test, y_pred_base, y_pred_isotonic, title):
    """Plot the results comparing base regressor vs isotonic regressor."""
    plt.figure(figsize=(12, 5))
    
    # Plot training data
    plt.subplot(1, 2, 1)
    plt.scatter(X_train, y_train, alpha=0.6, label='Training Data', s=20)
    plt.scatter(X_test, y_test, alpha=0.6, label='Test Data', s=20, color='red')
    
    # Sort for plotting
    sort_idx = np.argsort(X_test.flatten())
    X_sorted = X_test[sort_idx]
    y_pred_base_sorted = y_pred_base[sort_idx]
    y_pred_isotonic_sorted = y_pred_isotonic[sort_idx]
    
    plt.plot(X_sorted, y_pred_base_sorted, 'b-', label='Base Regressor', linewidth=2)
    plt.plot(X_sorted, y_pred_isotonic_sorted, 'g-', label='Isotonic Regressor', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'{title} - Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot residuals
    plt.subplot(1, 2, 2)
    residuals_base = y_test - y_pred_base
    residuals_isotonic = y_test - y_pred_isotonic
    
    plt.scatter(y_pred_base, residuals_base, alpha=0.6, label='Base Regressor', s=20)
    plt.scatter(y_pred_isotonic, residuals_isotonic, alpha=0.6, label='Isotonic Regressor', s=20, color='red')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'{title} - Residuals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function demonstrating IsotonicRegressor usage."""
    print("IsotonicRegressor Example")
    print("=" * 50)
    
    # Generate data
    X, y = generate_non_monotonic_data(n_samples=1000, noise=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print()
    
    # Test different base regressors
    base_regressors = [
        ("Linear Regression", LinearRegression()),
        ("Ridge Regression", Ridge(alpha=1.0)),
        ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42))
    ]
    
    for name, base_regr in base_regressors:
        print(f"Testing with {name} as base regressor:")
        print("-" * 40)
        
        # Create IsotonicRegressor with the base regressor
        isotonic_regr = IsotonicRegressor(
            regr=base_regr,
            increasing=True,  # You can set this to False for decreasing monotonicity
            out_of_bounds='clip'  # Handle out-of-bounds predictions
        )
        
        # Fit the model
        isotonic_regr.fit(X_train, y_train)
        
        # Make predictions
        y_pred_base = base_regr.predict(X_test)
        y_pred_isotonic = isotonic_regr.predict(X_test)
        
        # Calculate metrics
        mse_base = mean_squared_error(y_test, y_pred_base)
        mse_isotonic = mean_squared_error(y_test, y_pred_isotonic)
        r2_base = r2_score(y_test, y_pred_base)
        r2_isotonic = r2_score(y_test, y_pred_isotonic)
        
        print(f"Base Regressor MSE: {mse_base:.4f}")
        print(f"Isotonic Regressor MSE: {mse_isotonic:.4f}")
        print(f"Base Regressor R²: {r2_base:.4f}")
        print(f"Isotonic Regressor R²: {r2_isotonic:.4f}")
        print(f"Improvement in MSE: {((mse_base - mse_isotonic) / mse_base * 100):.2f}%")
        print()
        
        # Plot results
        plot_results(X_train, X_test, y_train, y_test, y_pred_base, y_pred_isotonic, name)
    
    print("Example completed!")


if __name__ == "__main__":
    main() 