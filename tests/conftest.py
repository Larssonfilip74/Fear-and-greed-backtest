import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest


def nyse_sessions(start: str, n: int) -> list:
    s = xcals.get_calendar("XNYS", start="2010-01-01").sessions
    s = s[s >= pd.Timestamp(start)][:n]
    return [d.date() for d in s]


def make_bars(rows, start="2024-01-02", roll_on=()):
    """rows: list of (open, high, low, close). Index = consecutive NYSE sessions from `start`."""
    idx = nyse_sessions(start, len(rows))
    df = pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close"], dtype=float)
    df["roll"] = [i in roll_on for i in range(len(rows))]
    return df


def random_walk_bars(n=1500, seed=7, start="2012-01-03"):
    rng = np.random.default_rng(seed)
    close = 2000 + np.cumsum(rng.normal(0.3, 15, n))
    close = np.round(close * 4) / 4  # tick grid
    open_ = np.round((np.r_[close[0], close[:-1]] + rng.normal(0, 4, n)) * 4) / 4
    high = np.maximum(open_, close) + np.round(np.abs(rng.normal(0, 6, n)) * 4) / 4
    low = np.minimum(open_, close) - np.round(np.abs(rng.normal(0, 6, n)) * 4) / 4
    rows = list(zip(open_, high, low, close))
    return make_bars(rows, start=start, roll_on=set(range(60, n, 63)))


@pytest.fixture
def bars_factory():
    return make_bars
