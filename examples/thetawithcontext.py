import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlsauce as ms 
# ============================================================================
# EXAMPLES
# ============================================================================

# Example 1: AirPassengers - Context-aware forecasting
# ============================================================================
print("=" * 80)
print("EXAMPLE 1: AirPassengers - Context-Aware Theta")
print("=" * 80)

# Load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df = pd.read_csv(url, parse_dates=['Month'], index_col='Month')
y = df['Passengers'].values

print(f"\nDataset: AirPassengers")
print(f"Length: {len(y)} observations")

# Fit context-aware model
model = ms.ContextAwareThetaForecaster(h=24)
model.fit(y)
forecast = model.predict()

print(f"\nMethod: {model.method_}")
print(f"Seasonal detected: {model.is_seasonal_}")
print(f"Seasonal period: {model.seasonal_period_}")
print(f"Alpha (SES): {model.alpha_:.4f}")
print(f"Drift (b₀): {model.b0_:.4f}")
print(f"Gamma (γ): {model.gamma_:.6f}")
print(f"Drift multiplier exp(γ): {np.exp(model.gamma_):.6f}")
print(f"\n24-month forecast: {forecast['mean'][-1]:.1f}")
print(f"95% PI: [{forecast['lower'][-1, -1]:.1f}, {forecast['upper'][-1, -1]:.1f}]")

# Plot
model.plot(figsize=(14, 6))
plt.tight_layout()
plt.show()

# Get detailed results
results = model.get_results()
print(f"\nDiagnostics:")
print(f"  In-sample MAE: {results['diagnostics']['mae']:.2f}")
print(f"  In-sample RMSE: {results['diagnostics']['rmse']:.2f}")
print(f"  Stability: {results['diagnostics']['stable']}")
print(f"  Context effect: {results['diagnostics']['context_effect']}")


# Example 2: Compare Standard vs Context-Aware
# ============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 2: Comparison - Standard Theta vs Context-Aware Theta")
print("=" * 80)

# Standard Theta (no context)
model_standard = ms.ContextAwareThetaForecaster(h=24, use_context=False)
model_standard.fit(y)
forecast_standard = model_standard.predict()

# Context-Aware Theta
model_context = ms.ContextAwareThetaForecaster(h=24, use_context=True)
model_context.fit(y)
forecast_context = model_context.predict()

print(f"\nStandard Theta:")
print(f"  Method: {model_standard.method_}")
print(f"  Gamma: {model_standard.gamma_:.6f}")
print(f"  24-month forecast: {forecast_standard['mean'][-1]:.1f}")

print(f"\nContext-Aware Theta:")
print(f"  Method: {model_context.method_}")
print(f"  Gamma: {model_context.gamma_:.6f}")
print(f"  24-month forecast: {forecast_context['mean'][-1]:.1f}")

print(f"\nDifference: {forecast_context['mean'][-1] - forecast_standard['mean'][-1]:+.1f}")
print(f"Percentage: {100 * (forecast_context['mean'][-1] / forecast_standard['mean'][-1] - 1):+.2f}%")

# Side-by-side comparison plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

model_standard.plot(ax=axes[0], title='Standard Theta (γ=0)')
model_context.plot(ax=axes[1], title='Context-Aware Theta')

plt.tight_layout()
plt.show()


# Example 3: Non-seasonal data (Nile river flow)
# ============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 3: Non-Seasonal Data - Nile River Flow")
print("=" * 80)

nile = np.array([1120, 1160, 963, 1210, 1160, 1160, 813, 1230, 1370, 1140,
                 995, 935, 1110, 994, 1020, 960, 1180, 799, 958, 1140,
                 1100, 1210, 1150, 1250, 1260, 1220, 1030, 1100, 774, 840,
                 874, 694, 940, 833, 701, 916, 692, 1020, 1050, 969,
                 831, 726, 456, 824, 702, 1120, 1100, 832, 764, 821,
                 768, 845, 864, 862, 698, 845, 744, 796, 1040, 759,
                 781, 865, 845, 944, 984, 897, 822, 1010, 771, 676,
                 649, 846, 812, 742, 801, 1040, 860, 874, 848, 890,
                 744, 749, 838, 1050, 918, 986, 797, 923, 975, 815,
                 1020, 906, 901, 1170, 912, 746, 919, 718, 714, 740])

print(f"\nDataset: Nile River Flow")
print(f"Length: {len(nile)} observations")

model = ms.ContextAwareThetaForecaster(h=10)
model.fit(nile, frequency=1)  # Explicitly non-seasonal
forecast = model.predict()

print(f"\nMethod: {model.method_}")
print(f"Seasonal detected: {model.is_seasonal_}")
print(f"Alpha: {model.alpha_:.4f}")
print(f"Drift: {model.b0_:.4f}")
print(f"Gamma: {model.gamma_:.6f}")
print(f"\n10-step forecast: {forecast['mean'][-1]:.1f}")

# Plot
model.plot(figsize=(14, 6))
plt.tight_layout()
plt.show()


# Example 4: Custom attention decay parameter (tau)
# ============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 4: Custom Attention Decay - Comparing τ=6 vs τ=12")
print("=" * 80)

# Shorter decay (more recent focus)
model_tau6 = ms.ContextAwareThetaForecaster(h=24, tau=6)
model_tau6.fit(y)
forecast_tau6 = model_tau6.predict()

# Standard decay
model_tau12 = ms.ContextAwareThetaForecaster(h=24, tau=12)
model_tau12.fit(y)
forecast_tau12 = model_tau12.predict()

print(f"\nτ=6 (faster decay, more recent focus):")
print(f"  Gamma: {model_tau6.gamma_:.6f}")
print(f"  24-month forecast: {forecast_tau6['mean'][-1]:.1f}")

print(f"\nτ=12 (slower decay, longer memory):")
print(f"  Gamma: {model_tau12.gamma_:.6f}")
print(f"  24-month forecast: {forecast_tau12['mean'][-1]:.1f}")

# Comparison plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
model_tau6.plot(ax=axes[0], title='τ=6 (Recent Focus)')
model_tau12.plot(ax=axes[1], title='τ=12 (Longer Memory)')
plt.tight_layout()
plt.show()


# Example 5: Multiple confidence levels
# ============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 5: Multiple Confidence Levels")
print("=" * 80)

model = ms.ContextAwareThetaForecaster(h=12, level=[50, 80, 90, 95, 99])
model.fit(y)
forecast = model.predict()

print(f"\nPrediction intervals for 12-month forecast:")
print(f"Point forecast: {forecast['mean'][-1]:.1f}")
print("\nConfidence Intervals:")
for i, lev in enumerate(model.level):
    print(f"  {lev}%: [{forecast['lower'][-1, i]:.1f}, {forecast['upper'][-1, i]:.1f}]")
    width = forecast['upper'][-1, i] - forecast['lower'][-1, i]
    print(f"       Width: {width:.1f}")

model.plot(figsize=(14, 6))
plt.tight_layout()
plt.show()


# Example 6: Quarterly data
# ============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 6: Quarterly Sales Data")
print("=" * 80)

# Simulated quarterly sales with trend and seasonality
np.random.seed(42)
quarters = 40
trend = np.linspace(100, 200, quarters)
seasonal = np.tile([0.9, 1.0, 1.1, 1.05], quarters // 4)
noise = np.random.normal(0, 5, quarters)
quarterly_sales = trend * seasonal + noise

print(f"\nDataset: Quarterly Sales")
print(f"Length: {len(quarterly_sales)} quarters ({len(quarterly_sales)//4} years)")

model = ms.ContextAwareThetaForecaster(h=8)  # 2 years ahead
model.fit(quarterly_sales, frequency=4)
forecast = model.predict()

print(f"\nMethod: {model.method_}")
print(f"Seasonal detected: {model.is_seasonal_}")
print(f"Seasonal period: {model.seasonal_period_}")
print(f"Alpha: {model.alpha_:.4f}")
print(f"Gamma: {model.gamma_:.6f}")
print(f"\n8-quarter forecast (2 years):")
for i, val in enumerate(forecast['mean'], 1):
    print(f"  Q{i}: {val:.1f}")

model.plot(figsize=(14, 6))
plt.title('Quarterly Sales Forecast')
plt.tight_layout()
plt.show()


# Example 7: Additive seasonality
# ============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 7: Additive Seasonality")
print("=" * 80)

# Create data with additive seasonality
np.random.seed(42)
months = 60
trend = np.linspace(100, 150, months)
seasonal = np.tile([10, 15, 5, -5, -10, -15, -5, 0, 5, 10, 15, 8], months // 12)
noise = np.random.normal(0, 3, months)
additive_data = trend + seasonal + noise

model_mult = ms.ContextAwareThetaForecaster(h=12, seasonal_method='multiplicative')
model_mult.fit(additive_data)

model_add = ms.ContextAwareThetaForecaster(h=12, seasonal_method='additive')
model_add.fit(additive_data)

print(f"\nMultiplicative decomposition:")
print(f"  Gamma: {model_mult.gamma_:.6f}")
print(f"  12-month forecast: {model_mult.predict()['mean'][-1]:.1f}")

print(f"\nAdditive decomposition:")
print(f"  Gamma: {model_add.gamma_:.6f}")
print(f"  12-month forecast: {model_add.predict()['mean'][-1]:.1f}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
model_mult.plot(ax=axes[0], title='Multiplicative Seasonality')
model_add.plot(ax=axes[1], title='Additive Seasonality')
plt.tight_layout()
plt.show()


# Example 8: Accessing fitted values and residuals
# ============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 8: Fitted Values and Residuals Analysis")
print("=" * 80)

model = ms.ContextAwareThetaForecaster(h=24)
model.fit(y)

fitted = model.fitted_
residuals = model.residuals_

print(f"\nFitted values:")
print(f"  Shape: {fitted.shape}")
print(f"  Mean: {np.mean(fitted):.2f}")
print(f"  Std: {np.std(fitted):.2f}")

print(f"\nResiduals:")
print(f"  Shape: {residuals.shape}")
print(f"  Mean: {np.mean(residuals):.4f}")
print(f"  Std: {np.std(residuals):.2f}")
print(f"  MAE: {np.mean(np.abs(residuals)):.2f}")
print(f"  RMSE: {np.sqrt(np.mean(residuals**2)):.2f}")

# Plot residuals
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Residuals over time
axes[0, 0].plot(residuals, 'o-', alpha=0.6)
axes[0, 0].axhline(0, color='r', linestyle='--')
axes[0, 0].set_title('Residuals Over Time')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Residual')
axes[0, 0].grid(True, alpha=0.3)

# Histogram of residuals
axes[0, 1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
axes[0, 1].axvline(0, color='r', linestyle='--')
axes[0, 1].set_title('Residuals Distribution')
axes[0, 1].set_xlabel('Residual')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True, alpha=0.3)

# Q-Q plot (manual)
from scipy import stats
theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
sample_quantiles = np.sort(residuals)
axes[1, 0].scatter(theoretical_quantiles, sample_quantiles, alpha=0.6)
axes[1, 0].plot(theoretical_quantiles, theoretical_quantiles, 'r--')
axes[1, 0].set_title('Q-Q Plot')
axes[1, 0].set_xlabel('Theoretical Quantiles')
axes[1, 0].set_ylabel('Sample Quantiles')
axes[1, 0].grid(True, alpha=0.3)

# ACF of residuals (simple version)
max_lag = 20
acf = [1.0]
for lag in range(1, max_lag + 1):
    corr = np.corrcoef(residuals[lag:], residuals[:-lag])[0, 1]
    acf.append(corr)
axes[1, 1].bar(range(len(acf)), acf, alpha=0.7)
axes[1, 1].axhline(1.96/np.sqrt(len(residuals)), color='r', linestyle='--', alpha=0.5)
axes[1, 1].axhline(-1.96/np.sqrt(len(residuals)), color='r', linestyle='--', alpha=0.5)
axes[1, 1].set_title('ACF of Residuals')
axes[1, 1].set_xlabel('Lag')
axes[1, 1].set_ylabel('ACF')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Example 9: Gamma estimation methods comparison
# ============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 9: Comparing Gamma Estimation Methods")
print("=" * 80)

# Cox partial likelihood
model_cox = ms.ContextAwareThetaForecaster(h=24, gamma_method='cox')
model_cox.fit(y)
forecast_cox = model_cox.predict()

# Ridge regression
model_reg = ms.ContextAwareThetaForecaster(h=24, gamma_method='regression')
model_reg.fit(y)
forecast_reg = model_reg.predict()

print(f"\nCox Partial Likelihood:")
print(f"  Gamma: {model_cox.gamma_:.6f}")
print(f"  24-month forecast: {forecast_cox['mean'][-1]:.1f}")
print(f"  MAE: {model_cox.get_results()['diagnostics']['mae']:.2f}")

print(f"\nRidge Regression:")
print(f"  Gamma: {model_reg.gamma_:.6f}")
print(f"  24-month forecast: {forecast_reg['mean'][-1]:.1f}")
print(f"  MAE: {model_reg.get_results()['diagnostics']['mae']:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
model_cox.plot(ax=axes[0], title='Cox Partial Likelihood')
model_reg.plot(ax=axes[1], title='Ridge Regression')
plt.tight_layout()
plt.show()


# Example 10: Full diagnostic report
# ============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 10: Complete Diagnostic Report")
print("=" * 80)

model = ms.ContextAwareThetaForecaster(h=24, level=[80, 95])
model.fit(y)
results = model.get_results()

print("\n" + "=" * 80)
print("CONTEXT-AWARE THETA FORECASTING REPORT")
print("=" * 80)

print(f"\nDATASET INFORMATION")
print(f"  Observations: {results['diagnostics']['n_obs']}")
print(f"  Seasonal: {results['seasonal']}")
if results['seasonal']:
    print(f"  Seasonal period: {results['seasonal_period']}")
print(f"  Method: {results['method']}")

print(f"\nESTIMATED PARAMETERS")
print(f"  SES smoothing (α): {results['parameters']['alpha']:.6f}")
print(f"  Baseline drift (b₀): {results['parameters']['b0']:.6f}")
print(f"  Context sensitivity (γ): {results['parameters']['gamma']:.6f}")
print(f"  Drift multiplier exp(γ): {np.exp(results['parameters']['gamma']):.6f}")
print(f"  Final SES level (l_n): {results['parameters']['l_n']:.2f}")
print(f"  Innovation variance (σ²): {results['parameters']['sigma2']:.2f}")

print(f"\nDIAGNOSTICS")
print(f"  In-sample MAE: {results['diagnostics']['mae']:.2f}")
print(f"  In-sample RMSE: {results['diagnostics']['rmse']:.2f}")
print(f"  Stability: {'PASS ✓' if results['diagnostics']['stable'] else 'FAIL ✗'}")
print(f"  Stability bound: |γ| < {results['diagnostics']['stability_bound']:.6f}")
print(f"  Context effect: {results['diagnostics']['context_effect']}")

print(f"\nFORECAST SUMMARY")
forecast = results['forecast']
print(f"  Horizon: {len(forecast['mean'])} periods")
print(f"  Point forecast (h={len(forecast['mean'])}): {forecast['mean'][-1]:.2f}")
for i, lev in enumerate(forecast['level']):
    print(f"  {lev}% PI: [{forecast['lower'][-1, i]:.2f}, {forecast['upper'][-1, i]:.2f}]")

print("\n" + "=" * 80)

# Final visualization
model.plot(figsize=(14, 6))
plt.tight_layout()
plt.show()

print("\n✓ All examples completed successfully!")