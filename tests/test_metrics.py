import math

import numpy as np
import pandas as pd
import pytest

from fgbt.backtest.metrics import equity_metrics, trade_metrics


def test_equity_metrics_hand_computed():
    # capital 1000; daily P&L +10, -20, +30, -10 -> equity 10, -10, 20, 10
    eq = pd.Series([10.0, -10.0, 20.0, 10.0])
    m = equity_metrics(eq, capital=1000.0, position=pd.Series([1, 1, 0, 1]))
    r = np.array([0.01, -0.02, 0.03, -0.01])
    assert m["sharpe"] == pytest.approx(r.mean() / r.std(ddof=1) * math.sqrt(252))
    assert m["sortino"] == pytest.approx(r.mean() / math.sqrt(np.mean(np.minimum(r, 0) ** 2)) * math.sqrt(252))
    # peak 10 then trough -10: drawdown 20 USD = 20 / (1000 + 10)
    assert m["max_dd_usd"] == pytest.approx(20.0)
    assert m["max_dd_pct"] == pytest.approx(20.0 / 1010.0)
    assert m["max_dd_sessions"] == 1  # one session below the previous peak (session index 1)
    assert m["time_in_market"] == pytest.approx(0.75)
    assert m["cagr"] == pytest.approx((1010 / 1000) ** (252 / 4) - 1)


def test_trade_metrics_hand_computed():
    tr = pd.DataFrame({"net_usd": [100.0, -50.0, 200.0, -25.0, 0.0, 10.0, 5.0]})
    m = trade_metrics(tr)
    assert m["n_trades"] == 7
    assert m["win_rate"] == pytest.approx(4 / 7)  # net > 0
    assert m["profit_factor"] == pytest.approx(315 / 75)
    assert m["expectancy_usd"] == pytest.approx(240 / 7)
    assert m["payoff_ratio"] == pytest.approx((315 / 4) / (75 / 2))
    assert m["top5_share"] == pytest.approx((200 + 100 + 10 + 5 + 0) / 240)


def test_trade_metrics_no_losses_and_empty():
    assert trade_metrics(pd.DataFrame({"net_usd": [5.0]}))["profit_factor"] == math.inf
    assert trade_metrics(pd.DataFrame({"net_usd": []}))["n_trades"] == 0


def test_max_dd_pct_is_largest_percentage_not_largest_usd():
    m = equity_metrics(pd.Series([-100.0, 10000.0, 9895.0]), capital=1000.0)
    assert m["max_dd_pct"] == pytest.approx(0.10)  # 100 / 1000, not 105 / 11000
    assert m["max_dd_usd"] == pytest.approx(105.0)


def test_top5_share_undefined_for_losing_total_and_ex_top5():
    m = trade_metrics(pd.DataFrame({"net_usd": [300.0, 200.0, -1000.0]}))
    assert math.isnan(m["top5_share"])
    m = trade_metrics(pd.DataFrame({"net_usd": [50.0, 40, 30, 20, 10, 5, -8]}))
    assert m["net_ex_top5_usd"] == pytest.approx(-3.0)
