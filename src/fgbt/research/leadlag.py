"""Lead-lag and predictive-regression tools (P2)."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def hac_regression(y, x, lag: int) -> dict:
    """OLS y = a + b x with Newey-West (HAC) standard errors; lag = horizon for overlapping returns."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    ok = ~(np.isnan(y) | np.isnan(x))
    model = sm.OLS(y[ok], sm.add_constant(x[ok]))
    fit = model.fit(cov_type="HAC", cov_kwds={"maxlags": lag}) if lag > 0 else model.fit()
    return {"beta": float(fit.params[1]), "se": float(fit.bse[1]), "t": float(fit.tvalues[1]),
            "p": float(fit.pvalues[1]), "n": int(ok.sum())}


def lead_lag_corr(x: pd.Series, r: pd.Series, leads) -> pd.Series:
    """corr(x[t], r[t+k]) for each lead k (positive k: x leads r)."""
    return pd.Series({k: x.corr(r.shift(-k)) for k in leads}, name="corr")


def jonckheere_terpstra(groups, alternative: str = "decreasing") -> tuple[float, float]:
    """Jonckheere-Terpstra trend test across ordered groups (normal approximation, no tie correction).

    'decreasing' tests H1: values fall from the first group to the last (H1 in the protocol).
    """
    gs = [np.asarray(g, dtype=float) for g in groups]
    if alternative == "decreasing":
        gs = gs[::-1]
    elif alternative != "increasing":
        raise ValueError("alternative must be 'increasing' or 'decreasing'")
    j = 0.0
    for a in range(len(gs)):
        for b in range(a + 1, len(gs)):
            diff = gs[b][:, None] - gs[a][None, :]
            j += (diff > 0).sum() + 0.5 * (diff == 0).sum()
    n = np.array([len(g) for g in gs])
    nn = n.sum()
    mean = (nn ** 2 - (n ** 2).sum()) / 4
    var = (nn ** 2 * (2 * nn + 3) - (n ** 2 * (2 * n + 3)).sum()) / 72
    z = (j - mean) / np.sqrt(var)
    return float(j), float(stats.norm.sf(z))


def granger_pvalues(r: pd.Series, driver: pd.Series, maxlag: int = 5) -> dict:
    """Granger test: does `driver` (stationary, e.g. daily change in F&G) help predict r beyond r's own lags?

    Returns {lag: p-value of the SSR F-test}.
    """
    import warnings

    from statsmodels.tsa.stattools import grangercausalitytests

    df = pd.concat([r, driver], axis=1).dropna().to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = grangercausalitytests(df, maxlag=maxlag)
    return {lag: float(res[lag][0]["ssr_ftest"][1]) for lag in res}
