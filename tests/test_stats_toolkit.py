"""Statistics toolkit tests: hand-checkable cases plus simulation calibration (known answers)."""
import math
from datetime import date

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from fgbt.validation.bootstrap import ar1_fit, event_bootstrap_pvalue, null_predictive_pvalue, stationary_bootstrap_ci
from fgbt.validation.multiple_testing import deflated_sharpe, holm, pbo_cscv, spa_pvalue
from fgbt.validation.trial_registry import count_trials, log_trial
from fgbt.research.events import (
    cross_back_events, forward_returns, independent_events, random_entry_pvalue,
)
from fgbt.research.leadlag import granger_pvalues, hac_regression, jonckheere_terpstra, lead_lag_corr
from fgbt.splits import HoldoutLockedError


def binom_band(n, p=0.05, q=0.995):
    lo, hi = stats.binom.ppf([1 - q, q], n, p)
    return lo / n, hi / n


# ------------------------------------------------------------------ trial registry
def test_trial_registry_appends_counts_and_locks_holdout(tmp_path):
    f = tmp_path / "trials.csv"
    log_trial(f, "H2", "DEV", {"T": 20, "h": 60}, {"mean": 0.01})
    log_trial(f, "H2", "DEV", {"h": 60, "T": 20}, {"mean": 0.01})  # same config, key order irrelevant
    log_trial(f, "H3", "VAL", {"T": 25}, {"mean": -0.002})
    df = pd.read_csv(f)
    assert len(df) == 3 and df.config_hash[0] == df.config_hash[1]
    assert count_trials(f) == 3 and count_trials(f, "H2") == 2 and count_trials(f, distinct=True) == 2
    with pytest.raises(HoldoutLockedError):
        log_trial(f, "H2", "HOLD", {"T": 20}, {})
    with pytest.raises(HoldoutLockedError):
        log_trial(f, "H2", "hold", {"T": 20}, {})  # label case cannot bypass the lock
    with pytest.raises(ValueError):
        log_trial(f, "H2", "HOLDOUT", {"T": 20}, {})  # unknown labels are rejected, not silently logged
    log_trial(f, "H2", "HOLD", {"T": 20}, {}, unlock_holdout=True)
    assert count_trials(f) == 4


# ------------------------------------------------------------------ multiple testing
def test_holm_hand_computed():
    adj = holm([0.01, 0.04, 0.03, 0.005])
    # sorted: .005*4=.02, .01*3=.03, .03*2=.06, .04*1=.04 -> monotone max -> .06
    assert np.allclose(adj, [0.03, 0.06, 0.06, 0.02])


def test_deflated_sharpe_reproduces_bailey_lopez_de_prado_example():
    # Bailey & Lopez de Prado (2014): annual SR 2.5, 1250 daily obs, skew -3, kurt 10, N=100, V[SR]=0.5 (annual) -> DSR ~ 0.90
    sr_d = 2.5 / math.sqrt(250)
    dsr = deflated_sharpe(sr_d, n_obs=1250, n_trials=100, var_trials_sr=0.5 / 250, skew=-3, kurt=10)
    assert dsr == pytest.approx(0.90, abs=0.01)
    assert deflated_sharpe(sr_d, 1250, 1, 0.5 / 250, 0, 3) > dsr  # fewer trials -> higher confidence


def test_pbo_noise_is_about_half_and_true_edge_is_low():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.01, (1000, 20))
    assert 0.3 < pbo_cscv(noise, n_blocks=10) < 0.7
    edge = noise.copy()
    edge[:, 0] += 0.004  # one strategy with a real, persistent edge
    assert pbo_cscv(edge, n_blocks=10) < 0.1


def test_spa_false_rejection_rate_is_controlled_and_power_exists():
    # One seed cannot test a null (a single benchmark draw can be genuinely beaten); calibrate over many.
    rng = np.random.default_rng(2)
    rejections, n_sims = 0, 120
    for i in range(n_sims):
        bench = rng.normal(0.0003, 0.01, 750)
        models = rng.normal(0.0003, 0.01, (750, 5))
        rejections += spa_pvalue(bench, models, reps=300, seed=i) < 0.05
    lo, hi = binom_band(n_sims)
    assert rejections / n_sims <= hi  # family-wise error at or below nominal
    bench = rng.normal(0.0003, 0.01, 1500)
    models = rng.normal(0.0003, 0.01, (1500, 5))
    models[:, 2] += 0.0015
    assert spa_pvalue(bench, models, seed=3) < 0.05


# ------------------------------------------------------------------ bootstrap
def test_stationary_bootstrap_ci_covers_true_mean_at_nominal_rate():
    rng = np.random.default_rng(4)
    hits, n_sims = 0, 300
    for _ in range(n_sims):
        e = rng.normal(size=400)
        x = np.empty(400)
        x[0] = e[0]
        for t in range(1, 400):  # AR(1) phi=0.5: iid bootstrap would under-cover
            x[t] = 0.5 * x[t - 1] + e[t]
        lo, hi = stationary_bootstrap_ci(x, reps=300, seed=int(rng.integers(1e9)))
        hits += lo <= 0.0 <= hi
    assert hits / n_sims > 0.88  # nominal 0.95; tolerance for block-bootstrap small-sample bias


def test_event_bootstrap_pvalue_calibrated_and_powerful():
    rng = np.random.default_rng(5)
    ps = [event_bootstrap_pvalue(rng.normal(0, 1, 30), reps=500, seed=i) for i in range(400)]
    lo, hi = binom_band(400)
    assert lo <= np.mean(np.array(ps) < 0.05) <= hi + 0.03
    assert event_bootstrap_pvalue(rng.normal(1.0, 1, 30), reps=2000, seed=9) < 0.01


def test_ar1_fit_recovers_parameters():
    rng = np.random.default_rng(6)
    x = np.empty(5000)
    x[0] = 50
    for t in range(1, 5000):
        x[t] = 2.5 + 0.95 * x[t - 1] + rng.normal(0, 4)
    phi, c, sigma, _ = ar1_fit(x)
    assert phi == pytest.approx(0.95, abs=0.01) and sigma == pytest.approx(4, abs=0.15)


def _stambaugh_world(rng, n, beta, rho=0.7, phi=0.97):
    u_v = rng.multivariate_normal([0, 0], [[1, rho * 4], [rho * 4, 16]], n)
    x = np.empty(n)
    x[0] = 50
    for t in range(1, n):
        x[t] = 1.5 + phi * x[t - 1] + u_v[t, 1]
    r = np.r_[0, beta * (x[:-1] - 50) + u_v[1:, 0]]  # r[t] depends on x[t-1]
    return x, r


def test_null_predictive_bootstrap_is_calibrated_under_stambaugh_bias():
    rng = np.random.default_rng(7)
    ps = [null_predictive_pvalue(*_stambaugh_world(rng, 600, 0.0)[::-1], h=5, reps=199, seed=s) for s in range(200)]
    lo, hi = binom_band(200)
    assert lo <= np.mean(np.array(ps) < 0.05) <= hi + 0.03


def test_null_predictive_bootstrap_detects_real_predictability():
    rng = np.random.default_rng(8)
    x, r = _stambaugh_world(rng, 2000, -0.02)
    assert null_predictive_pvalue(r, x, h=5, reps=499, seed=1) < 0.01


# ------------------------------------------------------------------ events
def _fg(values, start="2024-01-02"):
    from conftest import nyse_sessions
    return pd.Series(values, index=nyse_sessions(start, len(values)), dtype=float)


def test_independent_events_require_sep_false_days():
    v = [50] * 25 + [20, 18, 30, 19] + [50] * 25 + [15] + [50] * 5
    fg = _fg(v)
    ev = independent_events(fg, lambda s: s < 25, sep=20)
    # 25 then 28 are one episode (1 false day between); 54 follows 25 false days -> new episode
    assert ev == [fg.index[25], fg.index[54]]


def test_independent_events_off_by_one_matches_protocol():
    base = [50] * 30
    fg19 = _fg(base[:20] + [10] + [50] * 19 + [10] + base)   # 19 false days between -> same episode
    fg20 = _fg(base[:20] + [10] + [50] * 20 + [10] + base)   # 20 false days between -> new episode
    cond = lambda s: s < 25  # noqa: E731
    assert len(independent_events(fg19, cond, sep=20)) == 1
    assert len(independent_events(fg20, cond, sep=20)) == 2


def test_episode_running_at_series_start_is_not_an_event():
    fg = _fg([10, 12] + [50] * 30 + [10] + [50] * 5)
    assert independent_events(fg, lambda s: s < 25, sep=20) == [fg.index[32]]


def test_cross_back_independence_resets_on_suppressed_crosses():
    # crosses at 21, 23, 25, 27 (choppy) then 43: 43 is only 15 false days after the cross at 27 -> suppressed
    v = [50] * 20 + [20, 30, 20, 30, 20, 30, 20, 30] + [50] * 14 + [20, 30] + [50] * 30
    fg = _fg(v)
    assert cross_back_events(fg, 25, sep=20) == [fg.index[21]]


def test_forward_returns_enter_next_nyse_open_with_guards():
    from conftest import make_bars
    bars = make_bars([(100, 101, 99, 100), (102, 103, 98, 101), (104, 106, 103, 105), (108, 110, 107, 109)])
    out = forward_returns([bars.index[0]], bars, h=2, rth_bars=True, ratio_comparable=True)
    row = out.iloc[0]
    assert row.entry_session == bars.index[1] and row.exit_session == bars.index[3]
    assert row.log_return == pytest.approx(math.log(108 / 102))
    assert row.mae == pytest.approx(math.log(98 / 102)) and row.mfe == pytest.approx(math.log(106 / 102))
    late = forward_returns([bars.index[2]], bars, h=2, rth_bars=True, ratio_comparable=True)
    assert late.empty and late.attrs["dropped"] == 1  # exit beyond data -> dropped, never truncated
    with pytest.raises(ValueError, match="RTH"):
        forward_returns([bars.index[0]], bars, h=2, rth_bars=False, ratio_comparable=True)
    with pytest.raises(ValueError, match="ratio"):
        forward_returns([bars.index[0]], bars, h=2, rth_bars=True, ratio_comparable=False)


def test_forward_returns_never_shifts_entry_past_a_missing_bar():
    from conftest import make_bars
    bars = make_bars([(100, 101, 99, 100)] * 6)
    gappy = bars.drop(bars.index[1])  # the next NYSE session after index[0] is missing
    out = forward_returns([bars.index[0]], gappy, h=2, rth_bars=True, ratio_comparable=True)
    assert out.empty and out.attrs["dropped"] == 1


def test_forward_returns_rejects_non_nyse_sessions():
    from conftest import make_bars
    from datetime import date as _d
    bars = make_bars([(100, 101, 99, 100)] * 3, start="2024-07-02")
    bars.index = [_d(2024, 7, 2), _d(2024, 7, 3), _d(2024, 7, 4)]  # July 4: CME Globex only
    with pytest.raises(ValueError, match="non-NYSE"):
        forward_returns([bars.index[0]], bars, h=1, rth_bars=True, ratio_comparable=True)


def test_random_entry_pvalue_calibration():
    from conftest import random_walk_bars
    bars = random_walk_bars(n=1500, seed=12)
    rng = np.random.default_rng(13)
    idx = rng.choice(np.arange(10, 1400), 25, replace=False)
    obs = forward_returns([bars.index[i] for i in idx], bars, h=20, rth_bars=True, ratio_comparable=True)
    p = random_entry_pvalue(obs.log_return.mean(), bars, n_events=len(obs), h=20, sims=2000, seed=1)
    assert 0.01 < p < 0.99  # random dates are not special


# ------------------------------------------------------------------ lead-lag
def test_hac_regression_recovers_slope_and_widens_se():
    # HAC matters when both regressor and errors are persistent (as with F&G and overlapping returns)
    rng = np.random.default_rng(14)
    n = 3000
    x = np.empty(n)
    x[0] = 0.0
    e = rng.normal(size=n)
    for t in range(1, n):
        x[t] = 0.9 * x[t - 1] + e[t]
    y = 0.5 * x + np.convolve(rng.normal(size=n + 19), np.ones(20) / 20, "valid")
    res = hac_regression(y, x, lag=20)
    assert res["beta"] == pytest.approx(0.5, abs=0.05)
    naive = hac_regression(y, x, lag=0)
    assert res["se"] > 1.5 * naive["se"]


def test_jonckheere_terpstra_direction():
    rng = np.random.default_rng(15)
    groups = [rng.normal(m, 1, 60) for m in (0.6, 0.3, 0.0, -0.3)]  # decreasing
    stat, p_dec = jonckheere_terpstra(groups, alternative="decreasing")
    assert p_dec < 0.001
    _, p_inc = jonckheere_terpstra(groups, alternative="increasing")
    assert p_inc > 0.99


def test_lead_lag_corr_alignment():
    rng = np.random.default_rng(16)
    x = pd.Series(rng.normal(size=300))
    r = x.shift(2)  # r[t+2] == x[t]: x leads r by 2 sessions
    c = lead_lag_corr(x, r, leads=[0, 1, 2, 3])
    assert c.loc[2] == pytest.approx(1.0)
    assert abs(c.loc[0]) < 0.2 and abs(c.loc[3]) < 0.2


def test_granger_detects_planted_lead_and_not_noise():
    rng = np.random.default_rng(17)
    d = pd.Series(rng.normal(size=1500))
    r_lead = 0.3 * d.shift(1).fillna(0) + pd.Series(rng.normal(size=1500))
    assert granger_pvalues(r_lead, d, maxlag=2)[1] < 1e-6
    assert granger_pvalues(pd.Series(rng.normal(size=1500)), d, maxlag=2)[1] > 0.001
