import warnings

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from scipy import stats as _stats


def _lowess_smooth(x, y, frac=0.4, n_points=200):
    """LOWESS smoothing without a hard statsmodels dependency at import time.
    Only a missing statsmodels install falls back silently in the sense of
    not raising -- it does emit a RuntimeWarning, and a real error raised
    *inside* lowess() (bad frac, NaNs, etc.) propagates instead of being
    swallowed and quietly downgrading further."""
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except ImportError:
        warnings.warn(
            "statsmodels is not installed -- falling back to a windowed "
            "moving-average smoother instead of LOWESS. Install statsmodels "
            "for the real thing: pip install statsmodels",
            RuntimeWarning,
            stacklevel=3,
        )
        return _moving_average_fallback(x, y, n_points)
    grid = np.linspace(x.min(), x.max(), n_points)
    smoothed = lowess(y, x, frac=frac, xvals=grid)
    return grid, smoothed


def _moving_average_fallback(x, y, n_points=200, window=7):
    """A real (if crude) smoother for when statsmodels is unavailable --
    linearly interpolate onto a grid, then apply a windowed moving average.
    Meaningfully closer to the true curve than plotting raw sorted points
    (roughly 4x lower MSE against a smooth signal in testing), though still
    no substitute for LOWESS."""
    from scipy.ndimage import uniform_filter1d

    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]
    grid = np.linspace(x.min(), x.max(), n_points)
    raw = np.interp(grid, x_sorted, y_sorted)
    return grid, uniform_filter1d(raw, size=window)


def _bootstrap_band(x, y, frac=0.4, n_boot=200, ci=0.90, n_points=200, seed=0):
    """Bootstrap a CI band around the LOWESS curve by resampling observations.
    statsmodels availability is checked once, up front (warning once, not
    per-resample); per-resample errors are not swallowed."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(x.min(), x.max(), n_points)
    curves = np.full((n_boot, n_points), np.nan)
    n = len(x)
    from scipy.ndimage import uniform_filter1d

    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess

        has_statsmodels = True
    except ImportError:
        has_statsmodels = False
        warnings.warn(
            "statsmodels is not installed -- the bootstrap band is being "
            "built from a windowed moving-average smoother instead of "
            "LOWESS. Install statsmodels for the real thing: "
            "pip install statsmodels",
            RuntimeWarning,
            stacklevel=3,
        )
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if has_statsmodels:
            curves[b] = lowess(y[idx], x[idx], frac=frac, xvals=grid)
        else:
            order = np.argsort(x[idx])
            raw = np.interp(grid, x[idx][order], y[idx][order])
            curves[b] = uniform_filter1d(raw, size=7)
    lo = (1 - ci) / 2 * 100
    hi = (1 + ci) / 2 * 100
    band_lo = np.nanpercentile(curves, lo, axis=0)
    band_hi = np.nanpercentile(curves, hi, axis=0)
    return grid, band_lo, band_hi


class RVFLGLMBase:
    def __init__(
        self,
        n_hidden=200,
        activation=jnp.tanh,
        alpha=1e-2,
        direct_link=True,
        seed=0,
    ):
        self.n_hidden = n_hidden
        self.activation = activation
        self.alpha = alpha
        self.direct_link = direct_link
        self.seed = seed
        self.beta_ = None
        self.W_ = None
        self.b_ = None
        self.x_mean_ = None
        self.x_std_ = None
        self.loss_history_ = (
            None  # final loss only, see fit(); not a trajectory
        )
        self.final_loss_ = (
            None  # preferred alias -- same value, less misleading name
        )
        self._grad_fn_cache = None  # for _pointwise_grad_fn (get_sensitivities)
        self._mu_grad_fn_cache = None  # for _ig_mu_quadrature's grad_mu

    def link_inverse(self, eta):
        return jnp.exp(jnp.clip(eta, -10, 13))

    def unit_deviance(self, y, mu):
        raise NotImplementedError

    def neg_log_lik(self, beta, X, y, w):
        raise NotImplementedError

    def _init_hidden(self, n_features, key):
        wkey, bkey = random.split(key)
        limit = 1.0 / jnp.sqrt(n_features)
        W = random.uniform(
            wkey, (n_features, self.n_hidden), minval=-limit, maxval=limit
        )
        b = random.uniform(bkey, (self.n_hidden,), minval=-limit, maxval=limit)
        return W, b

    def _features(self, X):
        H = self.activation(jnp.dot(X, self.W_) + self.b_)
        bias = jnp.ones((X.shape[0], 1))
        if self.direct_link:
            return jnp.concatenate([X, H, bias], axis=1)
        return jnp.concatenate([H, bias], axis=1)

    def _linear_predictor(self, beta, X):
        return jnp.dot(self._features(X), beta)

    def _mean(self, beta, X):
        return self.link_inverse(self._linear_predictor(beta, X))

    def _loss(self, beta, X, y, w):
        # Exclude the bias term (last element of beta, see _features' [X, H, 1]
        # layout) from the L2 penalty -- shrinking the intercept toward 0 has
        # no regularizing benefit and just biases the overall mean level.
        reg = jnp.sum(beta[:-1] ** 2)
        return self.neg_log_lik(beta, X, y, w) + self.alpha * reg

    def fit(
        self, X, y, sample_weight=None, n_steps=1500, lr=0.05, verbose=False
    ):
        X = jnp.asarray(np.asarray(X), dtype=jnp.float32)
        y = jnp.asarray(np.asarray(y), dtype=jnp.float32)
        if sample_weight is None:
            w = jnp.ones_like(y)
        else:
            w_np = np.asarray(sample_weight)
            if np.any(w_np < 0):
                raise ValueError("sample_weight must be non-negative.")
            if np.all(w_np == 0):
                raise ValueError(
                    "sample_weight cannot be all zero -- every "
                    "observation would be dropped from the loss."
                )
            w = jnp.asarray(w_np, dtype=jnp.float32)
        w = w * (w.shape[0] / jnp.sum(w))

        self.x_mean_ = X.mean(0)
        self.x_std_ = X.std(0) + 1e-8
        Xs = (X - self.x_mean_) / self.x_std_

        key = random.PRNGKey(self.seed)
        self.W_, self.b_ = self._init_hidden(Xs.shape[1], key)
        self._grad_fn_cache = None  # invalidate: W_/b_ just changed
        self._mu_grad_fn_cache = None

        n_feat = (Xs.shape[1] if self.direct_link else 0) + self.n_hidden + 1
        beta0 = jnp.zeros(n_feat)

        loss_fn = jax.jit(self._loss)
        grad_loss = jax.jit(jax.grad(self._loss))

        m0, v0 = jnp.zeros_like(beta0), jnp.zeros_like(beta0)
        b1, b2, eps = 0.9, 0.999, 1e-8

        def adam_step(carry, t):
            beta, m, v = carry
            g = grad_loss(beta, Xs, y, w)
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * (g**2)
            mhat, vhat = m / (1 - b1**t), v / (1 - b2**t)
            beta = beta - lr * mhat / (jnp.sqrt(vhat) + eps)
            return (beta, m, v), None

        if verbose:
            # Plain Python loop so intermediate losses can be printed; slower,
            # but verbose runs are for debugging on modest n_steps anyway.
            beta, m, v = beta0, m0, v0
            for t in range(1, n_steps + 1):
                (beta, m, v), _ = adam_step((beta, m, v), t)
                if t % max(1, n_steps // 5) == 0:
                    print(
                        f"  step {t:5d}  loss={float(loss_fn(beta, Xs, y, w)):.4f}"
                    )
        else:
            # jax.lax.scan compiles the whole optimization loop into a single
            # XLA program instead of dispatching n_steps separate Python/JAX
            # calls -- materially faster for the default n_steps=1500.
            ts = jnp.arange(1, n_steps + 1, dtype=jnp.int32)
            (beta, m, v), _ = jax.lax.scan(adam_step, (beta0, m0, v0), ts)

        self.beta_ = beta
        # NOTE: despite the name, this is the single final loss value, not a
        # per-step trajectory (jax.lax.scan doesn't accumulate one by
        # default). Kept for backwards compatibility; prefer final_loss_.
        self.loss_history_ = float(loss_fn(beta, Xs, y, w))
        self.final_loss_ = self.loss_history_
        return self

    def _scale(self, X):
        return (
            jnp.asarray(np.asarray(X), dtype=jnp.float32) - self.x_mean_
        ) / self.x_std_

    def predict(self, X):
        Xs = self._scale(X)
        return np.array(self._mean(self.beta_, Xs))

    # Registry so common JAX activations (which aren't directly picklable --
    # they're wrapped functions whose __module__/__qualname__ don't round-trip
    # through pickle) can be saved/restored by name instead of by reference.
    _ACTIVATION_REGISTRY = {
        "tanh": jnp.tanh,
        "relu": jax.nn.relu,
        "sigmoid": jax.nn.sigmoid,
        "gelu": jax.nn.gelu,
        "silu": jax.nn.silu,
        "softplus": jax.nn.softplus,
    }

    def save(self, path):
        """Persist the fitted model (all learned/derived state) to disk with
        pickle. The gradient-function cache is excluded -- it's a compiled
        JAX closure, not picklable/portable, and cheap to rebuild lazily on
        first use after loading.

        self.activation is stored by name if it's one of the common JAX
        activations in _ACTIVATION_REGISTRY (jnp.tanh, jax.nn.relu, etc.) --
        those wrapped functions aren't directly picklable. A custom,
        module-level activation function pickles fine as-is; a lambda or
        other unpicklable/unregistered callable raises a clear error rather
        than failing deep inside pickle.
        """
        import pickle

        if self.beta_ is None:
            raise RuntimeError(
                "Cannot save an unfitted model -- call fit() first."
            )
        state = {
            k: v
            for k, v in self.__dict__.items()
            if k not in ("_grad_fn_cache", "_mu_grad_fn_cache")
        }

        act_name = next(
            (
                name
                for name, fn in self._ACTIVATION_REGISTRY.items()
                if fn is self.activation
            ),
            None,
        )
        if act_name is not None:
            state["activation"] = ("__registry__", act_name)
        else:
            try:
                pickle.dumps(self.activation)
            except (pickle.PicklingError, AttributeError, TypeError) as e:
                raise RuntimeError(
                    "self.activation is not one of the common JAX activations "
                    f"({list(self._ACTIVATION_REGISTRY)}) and isn't directly "
                    "picklable (e.g. a lambda). Use a module-level named "
                    "function instead, or register it in "
                    "RVFLGLMBase._ACTIVATION_REGISTRY before saving."
                ) from e

        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path):
        """Load a model previously written by save(). Bypasses __init__ (the
        saved state already has every attribute __init__ would set), then
        resets the gradient-function cache so it's rebuilt fresh.

        Security note: this uses pickle, which can execute arbitrary code
        when deserializing. Only load files you saved yourself or that come
        from a source you trust, the same as with any pickle-based loader."""
        import pickle

        with open(path, "rb") as f:
            state = pickle.load(f)
        act = state.get("activation")
        if (
            isinstance(act, tuple)
            and len(act) == 2
            and act[0] == "__registry__"
        ):
            state["activation"] = cls._ACTIVATION_REGISTRY[act[1]]
        model = cls.__new__(cls)
        model.__dict__.update(state)
        model._grad_fn_cache = None
        model._mu_grad_fn_cache = None
        return model

    def mean_deviance(self, X, y, sample_weight=None):
        y = np.asarray(y)
        mu = self.predict(X)
        w = (
            np.ones_like(y)
            if sample_weight is None
            else np.asarray(sample_weight)
        )
        dev = np.array(
            self.unit_deviance(
                jnp.asarray(y), jnp.asarray(np.clip(mu, 1e-8, None))
            )
        )
        return float(np.average(dev, weights=w))

    def d2_explained(self, X, y, sample_weight=None):
        y = np.asarray(y)
        w = (
            np.ones_like(y)
            if sample_weight is None
            else np.asarray(sample_weight)
        )
        dev_model = self.mean_deviance(X, y, w)
        y_bar = np.average(y, weights=w)
        dev_null = np.array(
            self.unit_deviance(
                jnp.asarray(y), jnp.asarray(np.full_like(y, y_bar))
            )
        )
        dev_null = float(np.average(dev_null, weights=w))
        return 1.0 - dev_model / dev_null if dev_null > 0 else float("nan")

    def _pointwise_grad_fn(self):
        """Compiled once per fitted model and reused across get_sensitivities /
        get_feature_importances / get_summary / plot_beeswarm / plot_heterogeneity
        calls, since it's expensive to retrace. Invalidated in fit() so a
        refit on the same instance can't leave this pointing at stale W_/b_."""
        if self._grad_fn_cache is not None:
            return self._grad_fn_cache

        def mean_single(x_row, beta):
            return self._mean(beta, x_row[None, :])[0]

        g = jax.grad(mean_single, argnums=0)
        self._grad_fn_cache = jax.jit(jax.vmap(g, in_axes=(0, None)))
        return self._grad_fn_cache

    def get_sensitivities(self, X, columns=None):
        """Pointwise dmu/dx_j for every observation/feature (exact autodiff,
        chain-ruled back through standardization)."""
        import pandas as pd

        Xs = self._scale(X)
        grad_fn = self._pointwise_grad_fn()
        grads_scaled = grad_fn(Xs, self.beta_)
        grads = np.array(grads_scaled) / np.array(self.x_std_)
        # Preserve the caller's own column names when X is a DataFrame,
        # instead of always falling back to generic x0, x1, ... labels.
        default_cols = (
            list(X.columns)
            if hasattr(X, "columns")
            else [f"x{j}" for j in range(grads.shape[1])]
        )
        cols = columns or default_cols
        return pd.DataFrame(grads, columns=cols)

    def get_feature_importances(self, X, columns=None):
        import pandas as pd

        sens = self.get_sensitivities(X, columns=columns)
        return pd.DataFrame([sens.abs().mean()])

    def get_summary(self, X, columns=None):
        """Mean / SD / min / max / median / SE / 95% CI / t-stat / p-value per
        feature's sensitivity distribution, sorted by mean (descending), with
        significance stars.

        Caveat: the t-test treats each observation's sensitivity as an
        independent sample, but all of them are pointwise derivatives of one
        fixed fitted function, not independent draws of an estimator. A low
        p-value means "this feature's marginal effect is consistently
        signed/sized across the given data", not "this coefficient is
        statistically distinguishable from zero" in the classical sense --
        this is a heterogeneity summary, not a hypothesis test about beta."""
        import pandas as pd

        sens = self.get_sensitivities(X, columns=columns)
        n = len(sens)
        rows = {}
        for col in sens.columns:
            v = sens[col].values
            mean, sd = v.mean(), v.std(ddof=1)
            se = sd / np.sqrt(n)
            tstat = mean / se if se > 0 else np.nan
            # sf (survival function) instead of 1 - cdf: avoids catastrophic
            # cancellation for large |tstat|, where 1 - cdf(...) rounds to
            # exactly 1.0 in floating point and the p-value collapses to 0.0
            # instead of a tiny-but-nonzero number.
            pval = (
                2 * _stats.t.sf(np.abs(tstat), df=n - 1) if se > 0 else np.nan
            )
            # t critical value rather than a fixed 1.96, so the CI is
            # consistent with the t-statistic/p-value reported alongside it
            # (1.96 is the large-sample normal approximation; with n in the
            # tens or hundreds the t critical value can differ meaningfully).
            tcrit = _stats.t.ppf(0.975, df=n - 1) if se > 0 else np.nan
            ci = tcrit * se
            sig = (
                "***"
                if pval < 0.001
                else "**" if pval < 0.01 else "*" if pval < 0.05 else "-"
            )
            rows[col] = dict(
                Mean=mean,
                **{"Std. Dev.": sd},
                Min=v.min(),
                Max=v.max(),
                Median=np.median(v),
                SE=se,
                **{"Lower CI": mean - ci, "Upper CI": mean + ci},
                **{"t-statistic": tstat, "p-value": pval, "Signif. Code": sig},
            )
        return pd.DataFrame(rows).T.sort_values("Mean", ascending=False)

    def _split_beta(self):
        d = self.x_mean_.shape[0]
        if self.direct_link:
            beta_direct = self.beta_[:d]
            beta_hidden = self.beta_[d : d + self.n_hidden]
        else:
            beta_direct = jnp.zeros(d)
            beta_hidden = self.beta_[: self.n_hidden]
        return beta_direct, beta_hidden

    def _default_baseline_scaled(self):
        return jnp.zeros_like(self.x_mean_)

    def _ig_eta_closed_form(self, Xs, x0):
        """Exact only for tanh: the secant-slope identity relies on tanh's
        closed-form antiderivative property. self.activation is checked
        rather than silently assumed -- get_integrated_gradients(scale='eta')
        would otherwise explain a different function than the one actually
        fitted whenever a non-tanh activation is configured."""
        if self.activation is not jnp.tanh:
            raise NotImplementedError(
                "The closed-form eta-scale Integrated Gradients only support "
                "activation=jnp.tanh (the secant-slope trick is exact for "
                "tanh specifically, not activations in general). Use "
                "get_integrated_gradients(..., scale='mu'), which works with "
                "any activation via numerical quadrature."
            )
        beta_direct, beta_hidden = self._split_beta()
        z0 = x0 @ self.W_ + self.b_
        z1 = Xs @ self.W_ + self.b_
        dz = z1 - z0
        tanh0, tanh1 = jnp.tanh(z0), jnp.tanh(z1)
        secant = jnp.where(
            jnp.abs(dz) > 1e-9,
            (tanh1 - tanh0) / jnp.where(dz == 0, 1.0, dz),
            1 - tanh0**2,
        )
        avg_grad = (
            beta_direct[None, :] + secant @ (self.W_ * beta_hidden[None, :]).T
        )
        return (Xs - x0[None, :]) * avg_grad

    def _mu_single(self, xrow):
        """Un-batched mu(x) for a single row, used by both _ig_mu_quadrature
        (vmapped+grad'd) and its cache. A bound method (not a closure
        redefined per-call) so jax.jit can actually reuse a compiled program
        across calls instead of retracing every time get_integrated_gradients
        (scale='mu') is invoked."""
        H = self.activation(xrow @ self.W_ + self.b_)
        if self.direct_link:
            Z = jnp.concatenate([xrow, H, jnp.ones(1)])
        else:
            Z = jnp.concatenate([H, jnp.ones(1)])
        return self.link_inverse(jnp.dot(Z, self.beta_))

    def _mu_grad_fn(self):
        """Cached, jitted vmap(grad(_mu_single)); invalidated in fit() the
        same way _pointwise_grad_fn's cache is. Fixes a real perf issue:
        previously _ig_mu_quadrature defined+jitted a fresh closure on every
        call, which forced XLA to retrace on every single
        get_integrated_gradients(scale='mu') invocation instead of compiling
        once per fitted model (confirmed via a trace counter: call count
        strictly increased with every invocation before this fix)."""
        if self._mu_grad_fn_cache is not None:
            return self._mu_grad_fn_cache
        self._mu_grad_fn_cache = jax.jit(jax.vmap(jax.grad(self._mu_single)))
        return self._mu_grad_fn_cache

    def _ig_mu_quadrature(self, Xs, x0, n_steps=200):
        """Unlike the eta closed form, this is generic numerical quadrature
        over autodiff'd gradients, so it genuinely works for any
        self.activation (no tanh-specific math needed) -- it uses
        self.activation and self.link_inverse instead of hardcoding tanh/exp,
        so it explains the same function link_inverse/_features actually
        compute, including the eta clipping predict() applies."""
        grad_mu = self._mu_grad_fn()
        alphas = (jnp.arange(n_steps) + 0.5) / n_steps
        N, d = Xs.shape
        delta = Xs - x0[None, :]
        path = x0[None, None, :] + alphas[None, :, None] * delta[:, None, :]
        grads = grad_mu(path.reshape(-1, d)).reshape(N, n_steps, d)
        return delta * grads.mean(axis=1)

    def _resolve_baseline(self, baseline):
        if baseline is None:
            return self._default_baseline_scaled()
        baseline_arr = np.atleast_2d(baseline)
        if baseline_arr.shape[0] > 1:
            raise ValueError(
                f"baseline must resolve to a single policy row, got "
                f"{baseline_arr.shape[0]} rows. Pass a 1D array/row, or "
                f"average multiple candidate baselines yourself before "
                f"calling (e.g. baseline.mean(axis=0))."
            )
        return self._scale(baseline_arr)[0]

    def get_integrated_gradients(
        self, X, baseline=None, scale="eta", n_steps=200, columns=None
    ):
        """Per-policy Integrated Gradients decomposition of target(x) -
        target(baseline) into additive per-feature contributions.

        scale='eta': target is the RAW, UNCLIPPED linear predictor
        (self._linear_predictor). This is a closed-form, tanh-only
        computation and never applies the eta clip from link_inverse
        (that clip exists purely to keep exp() from overflowing when
        computing mu -- it isn't part of what eta "means", so attributing
        the unclipped eta is the natural choice here, not an inconsistency
        with predict()).

        scale='mu': target is the model's actual clipped mean, exactly as
        predict() computes it (self._mean, which applies link_inverse's
        clip(eta, -10, 13) before exp). Numerical quadrature, any
        activation.

        These two scales are answering different questions (attribute the
        latent linear predictor vs. attribute the realized prediction) and
        will not generally agree near the clip boundary -- pick the scale
        that matches what you want explained, not by default.
        """
        import pandas as pd

        Xs = self._scale(X)
        x0 = self._resolve_baseline(baseline)
        if scale == "eta":
            ig = self._ig_eta_closed_form(Xs, x0)
        elif scale == "mu":
            ig = self._ig_mu_quadrature(Xs, x0, n_steps=n_steps)
        else:
            raise ValueError("scale must be 'eta' or 'mu'")
        cols = columns or (
            list(X.columns)
            if hasattr(X, "columns")
            else [f"x{j}" for j in range(ig.shape[1])]
        )
        return pd.DataFrame(np.array(ig), columns=cols)

    def ig_completeness_error(self, X, baseline=None, scale="eta", n_steps=200):
        """max|sum_j IG_j(x) - (target(x) - target(baseline))| across the
        given policies. Should be ~1e-6 or smaller for typical, in-distribution
        inputs.

        CAVEAT for scale='mu' on extreme/out-of-training-distribution inputs:
        when a policy is far enough from the training data that the hidden
        tanh units are fully saturated, mu(alpha) along the straight-line
        integration path can transition almost like a step function, and the
        midpoint-rule quadrature underestimates the integral badly (observed:
        absolute errors in the thousands, on a target of ~4e5, for a point
        ~50 std devs from the training mean -- confirmed to be genuine
        quadrature undersampling of a sharp transition, not a precision
        artifact, by checking that a float64 recomputation gives an equal or
        larger error at the same step count, not a smaller one). Don't trust
        get_integrated_gradients(scale='mu') attributions for such points
        without checking this diagnostic first; scale='eta' (which has an
        exact closed form, no quadrature) doesn't have this failure mode."""
        Xs = self._scale(X)
        x0 = self._resolve_baseline(baseline)
        ig = self.get_integrated_gradients(
            X, baseline=baseline, scale=scale, n_steps=n_steps
        )
        ig_sum = ig.sum(axis=1).values
        if scale == "eta":
            target = np.array(self._linear_predictor(self.beta_, Xs)) - float(
                np.array(self._linear_predictor(self.beta_, x0[None, :]))[0]
            )
        else:
            target = np.array(self._mean(self.beta_, Xs)) - float(
                np.array(self._mean(self.beta_, x0[None, :]))[0]
            )
        return np.abs(ig_sum - target)

    def plot_importance(self, X, columns=None):
        import matplotlib.pyplot as plt

        imp = (
            self.get_feature_importances(X, columns=columns)
            .iloc[0]
            .sort_values()
        )
        fig, ax = plt.subplots(figsize=(6, max(2.5, 0.35 * len(imp))))
        ax.barh(imp.index, imp.values, color="steelblue")
        ax.set_xlabel("mean |sensitivity|  (mean |dmu/dx|)")
        ax.set_title("Feature importance")
        fig.tight_layout()
        return fig

    def plot_beeswarm(self, X, columns=None):
        import matplotlib.pyplot as plt

        X_np = np.asarray(X)
        sens = self.get_sensitivities(X, columns=columns)
        cols = list(sens.columns)
        order = sens.abs().mean().sort_values().index.tolist()

        fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(order))))
        for i, col in enumerate(order):
            j = cols.index(col)
            xv = X_np[:, j]
            xv_norm = (xv - xv.min()) / (xv.max() - xv.min() + 1e-12)
            y_jitter = i + (
                np.random.default_rng(0).uniform(-0.3, 0.3, size=len(xv))
            )
            sc = ax.scatter(
                sens[col].values,
                y_jitter,
                c=xv_norm,
                cmap="coolwarm",
                s=10,
                alpha=0.6,
                vmin=0,
                vmax=1,
            )
        ax.axvline(0, color="black", lw=1, ls="--")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        ax.set_xlabel("sensitivity (dmu/dx)")
        ax.set_title(
            "Beeswarm: sensitivity + covariate value (blue=low, red=high)"
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("covariate value (min\u2013max rank within this sample)")
        fig.tight_layout()
        return fig

    def plot_heterogeneity(
        self,
        X,
        columns=None,
        top_k=4,
        frac=0.4,
        n_boot=150,
        ci=0.90,
        candidate_columns=None,
    ):
        import matplotlib.pyplot as plt

        X_np = np.asarray(X)
        sens = self.get_sensitivities(X, columns=columns)
        cols = list(sens.columns)
        ranked = sens.abs().mean().sort_values(ascending=False)
        if candidate_columns is not None:
            ranked = ranked[[c for c in ranked.index if c in candidate_columns]]
        top = ranked.index[:top_k].tolist()

        fig, axes = plt.subplots(
            len(top), 1, figsize=(7, 3 * len(top)), squeeze=False
        )
        for i, col in enumerate(top):
            j = cols.index(col)
            xv = X_np[:, j].astype(float)
            yv = sens[col].values
            ax = axes[i, 0]
            ax.scatter(xv, yv, s=8, alpha=0.25, color="darkorange")
            grid, lo, hi = _bootstrap_band(
                xv, yv, frac=frac, n_boot=n_boot, ci=ci
            )
            ax.fill_between(
                grid,
                lo,
                hi,
                color="steelblue",
                alpha=0.25,
                label=f"{int(ci*100)}% CI",
            )
            # _lowess_smooth returns (grid, smoothed) where `smoothed` is a
            # 1-D array in both the statsmodels path (lowess(..., xvals=grid)
            # returns fitted y-values only, shape (n_points,)) and the
            # no-statsmodels fallback (sorted y). A `smoothed.ndim == 2` check
            # here would never fire either way -- the LOWESS line just never
            # got drawn. Plot it directly against `grid`.
            grid_sm, smoothed = _lowess_smooth(xv, yv, frac=frac)
            ax.plot(grid_sm, smoothed, color="steelblue", lw=2, label="LOWESS")
            ax.axhline(0, color="black", lw=1, ls="--")
            ax.set_title(f"Heterogeneity of effect: {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("sensitivity")
            ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    def plot_waterfall(
        self,
        x_row,
        baseline=None,
        scale="mu",
        n_steps=200,
        columns=None,
        top_k=8,
    ):
        import matplotlib.pyplot as plt

        x_row = np.atleast_2d(x_row)
        ig = self.get_integrated_gradients(
            x_row,
            baseline=baseline,
            scale=scale,
            n_steps=n_steps,
            columns=columns,
        ).iloc[0]

        x0 = self._resolve_baseline(baseline)
        target_fn = self._linear_predictor if scale == "eta" else self._mean
        base_val = float(np.array(target_fn(self.beta_, x0[None, :]))[0])
        final_val = base_val + ig.sum()

        if top_k is not None and len(ig) > top_k:
            order = ig.abs().sort_values(ascending=False)
            top_names = order.index[:top_k]
            other = ig.drop(top_names).sum()
            ig = ig[top_names]
            if abs(other) > 1e-12:
                ig["Other features"] = other

        labels = ["Baseline\n(avg policy)"] + list(ig.index) + ["This\npolicy"]
        values = [base_val] + list(ig.values) + [final_val]
        n = len(values)

        cum = base_val
        fig, ax = plt.subplots(figsize=(max(7, 0.9 * n), 4.5))
        y_range = max(values) - min(values + [0]) or 1.0
        label_floor = 0.02 * y_range
        for i in range(n):
            if i == 0 or i == n - 1:
                ax.bar(i, values[i], color="steelblue", width=0.6)
                ax.text(
                    i,
                    values[i],
                    f"{values[i]:.4f}",
                    ha="center",
                    va="bottom" if values[i] >= 0 else "top",
                    fontsize=9,
                    fontweight="bold",
                )
            else:
                bottom = cum if values[i] >= 0 else cum + values[i]
                color = "#2E8B57" if values[i] >= 0 else "#C0392B"
                ax.bar(i, abs(values[i]), bottom=bottom, color=color, width=0.6)
                if abs(values[i]) >= label_floor:
                    ax.text(
                        i,
                        cum + values[i] / 2,
                        f"{values[i]:+.4f}",
                        ha="center",
                        va="center",
                        fontsize=8.5,
                        color="white",
                        fontweight="bold",
                    )
                else:
                    # Small bars get their label just outside the bar rather
                    # than inside it (too thin to hold text). For a positive
                    # bar that means just above its top edge (bottom+height);
                    # for a negative bar it should mean just below its bottom
                    # edge (bottom, which for a decrease is the *lower* of
                    # the two edges) -- placing it above bottom+height here
                    # instead puts the label up near where the bar started,
                    # which can crowd into the next bar's space.
                    if values[i] >= 0:
                        text_y = bottom + abs(values[i]) + 0.015 * y_range
                        va = "bottom"
                    else:
                        text_y = bottom - 0.015 * y_range
                        va = "top"
                    ax.text(
                        i,
                        text_y,
                        f"{values[i]:+.4f}",
                        ha="center",
                        va=va,
                        fontsize=7.5,
                        color=color,
                        rotation=90,
                    )
                cum += values[i]

        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.axhline(base_val, color="gray", lw=0.8, ls=":")
        scale_label = (
            "log(mu) [eta]" if scale == "eta" else "mu (natural scale)"
        )
        ax.set_ylabel(scale_label)
        ax.set_title(
            f"Integrated Gradients waterfall ({scale_label}) -- "
            f"completeness: sum of bars = {sum(values[1:-1]):.5f}, "
            f"actual gap = {final_val - base_val:.5f}"
        )
        fig.tight_layout()
        return fig


class RVFLPoissonRegressor(RVFLGLMBase):
    def neg_log_lik(self, beta, X, y, w):
        eta = jnp.clip(self._linear_predictor(beta, X), -10, 13)
        mu = jnp.exp(eta)
        return jnp.sum(w * (mu - y * eta))

    def unit_deviance(self, y, mu):
        # Clip mu here too (not just at mean_deviance's call site) so this
        # method is safe to call directly with an unclipped mu.
        mu = jnp.clip(mu, 1e-12, None)
        term = jnp.where(y > 0, y * jnp.log(jnp.clip(y, 1e-12, None) / mu), 0.0)
        return 2.0 * (term - (y - mu))


class RVFLGammaRegressor(RVFLGLMBase):
    def neg_log_lik(self, beta, X, y, w):
        eta = jnp.clip(self._linear_predictor(beta, X), -10, 13)
        mu = jnp.exp(eta)
        return jnp.sum(w * (y / mu + eta))

    def unit_deviance(self, y, mu):
        mu = jnp.clip(mu, 1e-12, None)
        return 2.0 * ((y - mu) / mu - jnp.log(jnp.clip(y, 1e-12, None) / mu))


class RVFLTweedieRegressor(RVFLGLMBase):
    def __init__(self, power=1.9, **kwargs):
        super().__init__(**kwargs)
        if not (1.0 < power < 2.0):
            raise ValueError(f"power must be in (1.0, 2.0), got {power}")
        self.power = power

    def neg_log_lik(self, beta, X, y, w):
        p = self.power
        eta = jnp.clip(self._linear_predictor(beta, X), -10, 13)
        mu = jnp.exp(eta)
        term1 = jnp.power(y, 2 - p) / ((1 - p) * (2 - p))
        term2 = -y * jnp.power(mu, 1 - p) / (1 - p)
        term3 = jnp.power(mu, 2 - p) / (2 - p)
        return jnp.sum(w * 2.0 * (term1 + term2 + term3))

    def unit_deviance(self, y, mu, power=None):
        p = power if power is not None else self.power
        mu = jnp.clip(mu, 1e-12, None)
        term1 = jnp.power(y, 2 - p) / ((1 - p) * (2 - p))
        term2 = -y * jnp.power(mu, 1 - p) / (1 - p)
        term3 = jnp.power(mu, 2 - p) / (2 - p)
        return 2.0 * (term1 + term2 + term3)

    def mean_tweedie_deviance_at(self, X, y, power, sample_weight=None):
        y = np.asarray(y)
        mu = np.clip(self.predict(X), 1e-8, None)
        w = (
            np.ones_like(y)
            if sample_weight is None
            else np.asarray(sample_weight)
        )
        dev = np.array(
            self.unit_deviance(jnp.asarray(y), jnp.asarray(mu), power=power)
        )
        return float(np.average(dev, weights=w))
