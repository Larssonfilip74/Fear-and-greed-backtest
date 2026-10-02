"""Data validation helpers (protocol Gate P1)."""
from datetime import date
from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd


@lru_cache(maxsize=1)
def _nyse_sessions() -> pd.DatetimeIndex:
    return xcals.get_calendar("XNYS", start="2005-01-01").sessions


def missing_trading_days(index) -> list[date]:
    """NYSE sessions between min(index) and max(index) that are absent from index."""
    days = sorted(index)
    sessions = _nyse_sessions()
    window = sessions[(sessions >= pd.Timestamp(days[0])) & (sessions <= pd.Timestamp(days[-1]))]
    have = set(days)
    return [s.date() for s in window if s.date() not in have]


def extra_non_trading_days(index) -> list[date]:
    sessions = {s.date() for s in _nyse_sessions()}
    return sorted(d for d in index if d not in sessions)


def gap_runs(missing: list[date]) -> list[tuple[date, date, int]]:
    """Group consecutive missing sessions into (first, last, n_sessions) runs."""
    sessions = list(_nyse_sessions().date)
    pos = {d: i for i, d in enumerate(sessions)}
    runs = []
    for d in missing:
        if runs and pos[d] == pos[runs[-1][1]] + 1:
            runs[-1] = (runs[-1][0], d, runs[-1][2] + 1)
        else:
            runs.append((d, d, 1))
    return runs
