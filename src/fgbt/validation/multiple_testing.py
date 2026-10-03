"""Multiple-testing and overfitting controls (protocol R6, Gate P6)."""
import math
from itertools import combinations

import numpy as np
from scipy import stats

from fgbt.validation.bootstrap import SEED, _block

EULER_GAMMA = 0.5772156649015329


def holm(pvals) -> np.ndarray:
    """Holm step-down adjusted p-values (family-wise error control)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def expected_max_sharpe(n_trials: int, var_trials_sr: float) -> float:
    """E[max SR] across n_trials independent zero-skill trials (Bailey & Lopez de Prado 2014, eq. for SR0)."""
    if n_trials <= 1:
        return 0.0
    z = stats.norm.ppf
    return math.sqrt(var_trials_sr) * ((1 - EULER_GAMMA) * z(1 - 1 / n_trials) + EULER_GAMMA * z(1 - 1 / (n_trials * math.e)))


def deflated_sharpe(sr: float, n_obs: int, n_trials: int, var_trials_sr: float, skew: float, kurt: float) -> float:
    """Probability that the true Sharpe exceeds the best expected under pure luck. All SR inputs per period.

    kurt is raw (not excess) kurtosis; a normal distribution has kurt = 3.
    """
    sr0 = expected_max_sharpe(n_trials, var_trials_sr)
    denom = math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return float(stats.norm.cdf((sr - sr0) * math.sqrt(n_obs - 1) / denom))


def _sharpe(m: np.ndarray) -> np.ndarray:
    sd = m.std(axis=0, ddof=1)
    return np.where(sd > 0, m.mean(axis=0) / np.where(sd > 0, sd, 1), -np.inf)


def pbo_cscv(perf: np.ndarray, n_blocks: int = 16) -> float:
    """Probability of backtest overfitting via combinatorially symmetric cross-validation (Bailey et al. 2017).

    perf: T x N matrix of per-period returns of N candidate configurations.
    """
    perf = np.asarray(perf, dtype=float)
    t, n = perf.shape
    if n_blocks % 2 or n_blocks < 2:
        raise ValueError("n_blocks must be an even number >= 2")
    blocks = np.array_split(np.arange(t), n_blocks)
    lambdas = []
    for is_ids in combinations(range(n_blocks), n_blocks // 2):
        is_rows = np.concatenate([blocks[i] for i in is_ids])
        oos_rows = np.concatenate([blocks[i] for i in range(n_blocks) if i not in is_ids])
        best = int(np.argmax(_sharpe(perf[is_rows])))
        oos = _sharpe(perf[oos_rows])
        rank = stats.rankdata(oos)[best]  # 1 = worst, n = best
        w = rank / (n + 1)
        lambdas.append(math.log(w / (1 - w)))
    return float(np.mean(np.array(lambdas) <= 0))


def spa_pvalue(benchmark_returns, model_returns, reps: int = 2000, block: float | None = None, seed: int = SEED) -> float:
    """Hansen's SPA test: H0 = no model beats the benchmark. Uses losses = -returns (arch implementation)."""
    from arch.bootstrap import SPA

    b = -np.asarray(benchmark_returns, dtype=float)
    m = -np.asarray(model_returns, dtype=float)
    if m.ndim == 1:
        m = m[:, None]
    block = block or _block(len(b))
    spa = SPA(b, m, block_size=block, reps=reps, seed=seed)
    spa.compute()
    return float(spa.pvalues["consistent"])
