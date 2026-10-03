"""Bootstrap inference (protocol R6)."""
import numpy as np
from scipy.signal import lfilter

SEED = 20261002


def _block(n: int) -> float:
    return max(1.0, n ** (1 / 3))


def stationary_bootstrap_ci(x, stat=np.mean, reps: int = 10000, alpha: float = 0.05, block: float | None = None,
                            seed: int = SEED) -> tuple[float, float]:
    """Percentile CI from the Politis-Romano stationary bootstrap (keeps serial dependence)."""
    from arch.bootstrap import StationaryBootstrap

    x = np.asarray(x, dtype=float)
    bs = StationaryBootstrap(block or _block(len(x)), x, seed=seed)
    draws = np.array([stat(d[0][0]) for d in bs.bootstrap(reps)])
    return float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


def event_bootstrap_pvalue(excess, reps: int = 10000, seed: int = SEED) -> float:
    """Two-sided p-value for H0: mean(excess) = 0, independent events (bootstrap of the recentred sample)."""
    x = np.asarray(excess, dtype=float)
    rng = np.random.default_rng(seed)
    obs = x.mean()
    centred = x - obs
    draws = centred[rng.integers(0, len(x), (reps, len(x)))].mean(axis=1)
    return float((np.sum(np.abs(draws) >= abs(obs)) + 1) / (reps + 1))


def ar1_fit(x) -> tuple[float, float, float, np.ndarray]:
    """OLS AR(1): x_t = c + phi x_{t-1} + v_t. Returns (phi, c, sigma_v, residuals).

    x may be one array or a list of contiguous pieces (pooled fit that never bridges a data gap).
    """
    pieces = x if isinstance(x, list) else [x]
    pieces = [np.asarray(p, dtype=float) for p in pieces]
    lagged = np.concatenate([p[:-1] for p in pieces])
    current = np.concatenate([p[1:] for p in pieces])
    X = np.column_stack([np.ones(len(lagged)), lagged])
    coef, *_ = np.linalg.lstsq(X, current, rcond=None)
    resid = current - X @ coef
    return float(coef[1]), float(coef[0]), float(resid.std(ddof=2)), resid


def _forward_sum(r: np.ndarray, h: int) -> np.ndarray:
    """y[t] = r[t+1] + ... + r[t+h]; NaN where the window runs past the end."""
    c = np.r_[0.0, np.cumsum(r)]
    n = len(r)
    y = np.full(n, np.nan)
    y[: n - h] = c[h + 1:] - c[1: n - h + 1]
    return y


def _slope(y: np.ndarray, x: np.ndarray) -> float:
    ok = ~np.isnan(y)
    xc = x[ok] - x[ok].mean()
    return float(np.dot(xc, y[ok] - y[ok].mean()) / np.dot(xc, xc))


def null_predictive_pvalue(r, x, h: int, reps: int = 999, seed: int = SEED) -> float:
    """Two-sided p-value for the slope of sum(r[t+1..t+h]) on x[t] under H0: no predictability.

    The null world keeps x's AR(1) persistence and the contemporaneous correlation between return and
    predictor shocks (jointly resampled residual pairs), which is exactly what creates Stambaugh bias.
    r[t] is the return realised on day t; x[t] is known at the end of day t.
    """
    r = np.asarray(r, dtype=float)
    x = np.asarray(x, dtype=float)
    beta_obs = _slope(_forward_sum(r, h), x)
    phi, c, _, v = ar1_fit(x)
    u = r[1:] - r[1:].mean()  # return shocks aligned with predictor shocks v (same day t)
    rng = np.random.default_rng(seed)
    n = len(x)
    k = rng.integers(0, len(v), (reps, n - 1))
    # AR(1) paths for all resamples at once: xs[t] = c + phi * xs[t-1] + v*
    body = lfilter([1.0], [1.0, -phi], c + v[k], axis=1, zi=np.full((reps, 1), phi * x[0]))[0]
    xs = np.column_stack([np.full(reps, x[0]), body])
    rs = np.column_stack([np.zeros(reps), r[1:].mean() + u[k]])
    cs = np.cumsum(rs, axis=1)
    ys = np.full((reps, n), np.nan)
    ys[:, : n - h] = cs[:, h:] - cs[:, : n - h]  # sum r[t+1..t+h]
    xv, yv = xs[:, : n - h], ys[:, : n - h]
    xc = xv - xv.mean(axis=1, keepdims=True)
    betas = (xc * (yv - yv.mean(axis=1, keepdims=True))).sum(axis=1) / (xc ** 2).sum(axis=1)
    hits = int(np.sum(np.abs(betas) >= abs(beta_obs)))
    return float((hits + 1) / (reps + 1))
