import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlsauce as ms 

# ============ QUICK TEST EXAMPLE ============
def quick_test():
    """Quick test of all modes"""
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression

    # Generate data
    np.random.seed(42)
    n = 120
    t = np.arange(n)
    trend = 0.05 * t
    seasonal = 5 * np.sin(2 * np.pi * t / 12)
    y = 100 + trend + seasonal + np.random.randn(n)

    # Test all modes with visualization
    modes = ['classic', 'cox', 'ridge', 'rslope']

    for mode in modes:
        print(f"\n{'='*50}")
        print(f"Mode: {mode}")
        print('='*50)

        est = LinearRegression() if mode == 'rslope' else None
        model = ms.ContextAwareThetaForecaster(mode=mode, tau=12, kernel='temporal', estimator=est, random_state=42)
        model.fit(y)

        forecast = model.predict(h=24)
        params = model.get_params()

        print(f"Alpha: {params['alpha']:.4f}")
        print(f"Drift: {params['b0']:.4f}")
        print(f"Gamma: {params['gamma']:.6f}")
        print(f"24-month forecast: {forecast['mean'][-1]:.2f}")
        print(f"95% PI: [{forecast['lower'][-1]:.2f}, {forecast['upper'][-1]:.2f}]")

        # Plot
        model.plot(forecast, title=f"Unified Theta - {mode.upper()} Mode")
        plt.savefig(f'theta_{mode}.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved: theta_{mode}.png")

    plt.show()


    # Run main comparison (default)
    # Comment this out and uncomment one of the others to run different examples
    pass  # Main example runs automatically above

    # Uncomment to run quick test:
    quick_test()