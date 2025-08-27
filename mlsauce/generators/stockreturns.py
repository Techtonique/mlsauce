import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_synthetic_returns(
    n_days=252 * 10,               # ~10 years of daily data
    mu=0.0002,                     # Daily drift
    kappa=0.05,                    # Vol mean reversion
    theta=0.0001,                  # Long-run variance
    sigma_v=0.01,                  # Vol of vol
    rho=-0.7,                      # Leverage effect
    lambda_jump=0.05,              # Jump intensity (per day)
    jump_size_dist="normal",       # "normal", "log_normal", or "exponential"
    sigma_jump=0.02,               # Jump size (scale parameter)
    noise_dist="normal",           # "normal" or "student_t"
    noise_scale=0.0005,            # Microstructure noise scale
    noise_df=3.0,                  # Degrees of freedom for Student’s t
    regime_params=None,            # Regime switching params
    random_seed=None               # Reproducibility
):
    """
    Generates synthetic stock returns with:
    - Stochastic volatility (Heston-like)
    - Jumps (Poisson-driven, with configurable distribution)
    - Regime switching (Markov)
    - Leverage effect
    - Fat tails (via jumps and noise)
    - Microstructure noise (Gaussian or Student’s t)

    Args:
        jump_size_dist: Jump size distribution ("normal", "log_normal", "exponential").
        noise_dist: Microstructure noise distribution ("normal", "student_t").
        noise_df: Degrees of freedom for Student’s t noise (if used).
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Default regime switching (2 regimes: calm and turbulent)
    if regime_params is None:
        regime_params = {
            'transition_matrix': np.array([[0.99, 0.01], [0.03, 0.97]]),
            'theta_high_multiplier': 3.0,
            'kappa_high_multiplier': 2.0
        }

    # Initialize
    v = np.zeros(n_days)  # Variance
    r = np.zeros(n_days)  # Returns
    regime = np.zeros(n_days, dtype=int)
    v[0] = theta

    # Simulate regime switching (Markov chain)
    for t in range(1, n_days):
        regime[t] = np.random.choice(
            [0, 1],
            p=regime_params['transition_matrix'][regime[t-1]]
        )

    # Simulate returns and volatility
    for t in range(1, n_days):
        # Regime-dependent params
        if regime[t] == 1:  # High-vol regime
            theta_t = theta * regime_params['theta_high_multiplier']
            kappa_t = kappa * regime_params['kappa_high_multiplier']
        else:
            theta_t = theta
            kappa_t = kappa

        # Volatility process (Euler discretization)
        eta = np.random.normal()
        epsilon = rho * eta + np.sqrt(1 - rho**2) * np.random.normal()
        dv = kappa_t * (theta_t - v[t-1]) + sigma_v * np.sqrt(v[t-1]) * eta
        v[t] = max(v[t-1] + dv, 1e-6)  # Ensure positivity

        # Jumps (with configurable distribution)
        if np.random.poisson(lambda_jump) > 0:
            if jump_size_dist == "normal":
                J = np.random.normal(0, sigma_jump)
            elif jump_size_dist == "log_normal":
                J = np.exp(np.random.normal(0, sigma_jump)) - 1  # Log-normal (positive skew)
            elif jump_size_dist == "exponential":
                J = np.random.exponential(sigma_jump) * np.sign(np.random.uniform(-1, 1))  # Double-sided
            else:
                raise ValueError("Invalid jump_size_dist. Use 'normal', 'log_normal', or 'exponential'.")
        else:
            J = 0

        # Returns
        r[t] = mu + np.sqrt(v[t-1]) * epsilon + J

    # Microstructure noise (Gaussian or Student’s t)
    if noise_dist == "normal":
        r += np.random.normal(0, noise_scale, n_days)
    elif noise_dist == "student_t":
        r += np.random.standard_t(noise_df, n_days) * noise_scale / np.sqrt(noise_df / (noise_df - 2))
    else:
        raise ValueError("Invalid noise_dist. Use 'normal' or 'student_t'.")

    # Create DataFrame
    df = pd.DataFrame({
        'returns': r,
        'volatility': np.sqrt(v),
        'regime': regime
    }, index=pd.date_range(start='1970-01-01', periods=n_days))

    return df

def plot_synthetic_returns(df, title="Synthetic Stock Returns Analysis", figsize=(14, 10)):
    """
    Plot synthetic stock returns with multiple panels:
    - Returns over time
    - Volatility (sqrt variance)
    - Regime indicators
    - Distribution vs. normal (QQ plot and histogram)
    - Autocorrelation of returns and squared returns

    Args:
        df (pd.DataFrame): Output from generate_synthetic_returns
            Must have: 'returns', 'volatility', 'regime'
        title (str): Title for the plot
        figsize (tuple): Figure size
    """
    # Set style
    sns.set_style("darkgrid")
    plt.rcParams["figure.dpi"] = 100

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.4, wspace=0.3)

    # -------------------------
    # 1. Returns Over Time
    # -------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df.index, df['returns'], lw=0.8, color='tab:blue', alpha=0.9)
    ax1.set_title("Daily Returns")
    ax1.set_ylabel("Return")
    ax1.axhline(0, color="gray", linestyle="--", lw=0.8)

    # Highlight large jumps (optional)
    threshold = df['returns'].std() * 3
    jumps = df[np.abs(df['returns']) > threshold]
    if not jumps.empty:
        ax1.scatter(jumps.index, jumps['returns'], color='red', s=10, zorder=5, label="Large Moves")
        ax1.legend()

    # -------------------------
    # 2. Volatility
    # -------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df.index, df['volatility'], lw=1.2, color='tab:orange')
    ax2.set_title("Volatility (Latent)")
    ax2.set_ylabel("Volatility")

    # Shade turbulent regimes
    if 'regime' in df.columns:
        turbulent_days = df[df['regime'] == 1]
        if not turbulent_days.empty:
            ax2.fill_between(
                turbulent_days.index,
                df.loc[turbulent_days.index, 'volatility'].min(),
                df.loc[turbulent_days.index, 'volatility'],
                color='red',
                alpha=0.2,
                label="Turbulent Regime"
            )
            ax2.legend()

    # -------------------------
    # 3. Regime Plot
    # -------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.fill_between(df.index, 0, 1, where=(df['regime'] == 0),
                     interpolate=True, color='green', alpha=0.3, label="Calm Regime")
    ax3.fill_between(df.index, 0, 1, where=(df['regime'] == 1),
                     interpolate=True, color='red', alpha=0.3, label="Turbulent Regime")
    ax3.set_ylim(0, 1)
    ax3.set_yticks([])
    ax3.set_title("Regime Switching (Hidden State)")
    ax3.legend(loc="upper right")

    # -------------------------
    # 4. Return Distribution (QQ + Histogram)
    # -------------------------
    from scipy import stats

    ax4 = fig.add_subplot(gs[1, 1])
    stats.probplot(df['returns'], dist="norm", plot=ax4)
    ax4.set_title("QQ Plot (Fat Tails Detection)")
    ax4.get_lines()[0].set_marker('.')
    ax4.get_lines()[0].set_markersize(4)
    ax4.get_lines()[1].set_color('red')
    ax4.get_lines()[1].set_linewidth(1.5)

    # -------------------------
    # 5. Histogram with Normal Fit
    # -------------------------
    ax5 = fig.add_subplot(gs[2, 0])
    mu_norm, std_norm = stats.norm.fit(df['returns'])
    sns.histplot(df['returns'], bins=50, kde=False, stat='density', ax=ax5, alpha=0.7, color='skyblue')
    xmin, xmax = ax5.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu_norm, std_norm)
    ax5.plot(x, p, 'k--', linewidth=1.5, label="Normal Fit")
    ax5.set_title("Return Distribution")
    ax5.set_xlabel("Return")
    ax5.legend()

    # Annotate kurtosis and skew
    kurt = df['returns'].kurtosis()
    skew = df['returns'].skew()
    ax5.text(0.02, 0.9, f"Kurtosis: {kurt:.2f}\nSkewness: {skew:.2f}",
             transform=ax5.transAxes, fontsize=10,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    # -------------------------
    # 6. Autocorrelation
    # -------------------------
    ax6 = fig.add_subplot(gs[2, 1])
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(df['returns']**2, ax=ax6, lags=40, title="ACF of Squared Returns", alpha=0.05)
    ax6.set_xlabel("Lag (Days)")
    ax6.set_ylabel("Autocorrelation")

    # Add title at top
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    # Adjust layout
    plt.tight_layout()
    plt.show()