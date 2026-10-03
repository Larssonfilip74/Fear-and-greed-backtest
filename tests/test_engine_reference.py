"""Gate P3: engine equity must equal an independent vectorized calculation to the cent."""
import numpy as np
import pandas as pd
import pytest

from conftest import random_walk_bars
from fgbt.backtest.engine import run
from fgbt.backtest.orders import Order

PV, TICK, HALF_COMM = 50.0, 0.25, 2.5
ROLL = 12.5 + 5.0


def buy_and_hold(ctx):
    return [Order("BUY", "MOO", 1)] if ctx.i == 0 else []


def sma_cross(ctx, fast=50, slow=200):
    if len(ctx.bars) < slow:
        return []
    c = ctx.bars["close"]
    up = c.iloc[-fast:].mean() > c.iloc[-slow:].mean()
    if up and ctx.position == 0:
        return [Order("BUY", "MOO", 1)]
    if not up and ctx.position > 0:
        return [Order("SELL", "MOO", 1)]
    return []


def vectorized(bars: pd.DataFrame, pos: np.ndarray) -> np.ndarray:
    """pos[t] = contracts held during session t (entered/exited at the open). Independent P&L formula."""
    o, c, roll = bars.open.values, bars.close.values, bars.roll.values
    prev_pos = np.r_[0, pos[:-1]]
    prev_close = np.r_[np.nan, c[:-1]]
    pnl = np.zeros(len(bars))
    for t in range(len(bars)):
        held = min(prev_pos[t], pos[t])  # carried from yesterday's close
        bought = max(pos[t] - prev_pos[t], 0)
        sold = max(prev_pos[t] - pos[t], 0)
        if held:
            pnl[t] += held * (c[t] - prev_close[t]) * PV
        if bought:
            pnl[t] += bought * (c[t] - (o[t] + TICK)) * PV - bought * HALF_COMM
        if sold:
            pnl[t] += sold * ((o[t] - TICK) - prev_close[t]) * PV - sold * HALF_COMM
        if roll[t]:  # only contracts carried in and still held after the open roll
            pnl[t] -= held * ROLL
    return np.cumsum(pnl)


def test_buy_and_hold_matches_vectorized_to_the_cent():
    bars = random_walk_bars()
    res = run(bars, buy_and_hold)
    expected = vectorized(bars, np.ones(len(bars), dtype=int))
    assert np.max(np.abs(res.equity.values - expected)) < 0.01


def test_sma_cross_matches_vectorized_to_the_cent():
    bars = random_walk_bars(seed=11)
    res = run(bars, sma_cross)
    c = bars.close
    # signal from closes through t-1, position applies to session t
    up = (c.rolling(50).mean() > c.rolling(200).mean()).shift(1)
    valid = c.rolling(200).count().shift(1) >= 200
    pos = (up & valid).fillna(False).astype(int).values
    assert pos.sum() > 0 and (np.diff(pos) != 0).sum() >= 4  # the test exercises several round trips
    expected = vectorized(bars, pos)
    assert np.max(np.abs(res.equity.values - expected)) < 0.01
    assert (res.position.values == pos).all()


def test_truncation_invariance():
    bars = random_walk_bars(seed=3)
    full = run(bars, sma_cross)
    for k in (400, 777, 1200):
        part = run(bars.iloc[:k], sma_cross)
        assert np.allclose(part.equity.values, full.equity.values[:k], atol=1e-9)
        closed_before = full.trades[full.trades.exit_session < bars.index[k]]
        pd.testing.assert_frame_equal(part.trades.iloc[:len(closed_before)].reset_index(drop=True),
                                      closed_before.reset_index(drop=True))
