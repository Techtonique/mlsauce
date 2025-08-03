#!/usr/bin/env python3
"""
Simple test script for GenericFunctionalForecaster integration.
"""

import numpy as np

# Test import
try:
    from mlsauce.fpca import GenericFunctionalForecaster
    print("✓ Successfully imported GenericFunctionalForecaster")
except ImportError as e:
    print(f"✗ Failed to import GenericFunctionalForecaster: {e}")
    exit(1)

# Test basic functionality
def test_basic_functionality():
    """Test basic functionality of GenericFunctionalForecaster."""
    print("\nTesting basic functionality...")
    
    # Generate simple test data
    np.random.seed(42)
    n_samples, n_points = 50, 30
    X = np.random.randn(n_samples, n_points)
    grid = np.linspace(0, 1, n_points)
    
    # Test different configurations
    configs = [
        ('pca', 'ridge'),
        ('kernel_pca', 'linear'),
        ('sparse_pca', 'lasso')
    ]
    
    for pca_method, reg_method in configs:
        print(f"\nTesting {pca_method} + {reg_method}:")
        
        try:
            # Create forecaster
            forecaster = GenericFunctionalForecaster(
                n_components=3,
                pca_method=pca_method,
                regression_method=reg_method
            )
            
            # Fit the model
            forecaster.fit(X, grid)
            print(f"  ✓ Model fitting successful")
            
            # Test forecasting
            forecasts = forecaster.forecast(steps=5, method='ar')
            print(f"  ✓ Forecasting successful, shape: {forecasts.shape}")
            
            # Test model info
            info = forecaster.get_model_info()
            print(f"  ✓ Model info: {info['pca_method']}, {info['regression_method']}")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    return True

if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("\n🎉 Integration test passed!")
    else:
        print("\n❌ Integration test failed!") 