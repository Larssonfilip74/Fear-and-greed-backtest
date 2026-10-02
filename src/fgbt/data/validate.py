"""Data validation helpers (protocol Gate P1)."""
from datetime import date
from functools import lru_cache

import exchange_calendars as xcals
import numpy as np
import pandas as pd


CALENDAR_START = "1990-01-01"  # covers the full ES1!/NQ1! daily history (1997/1999 onward)


@lru_cache(maxsize=4)
def _sessions(cal: str) -> pd.DatetimeIndex:
    """Single cached source for exchange session calendars (XNYS, CMES)."""
    return xcals.get_calendar(cal, start=CALENDAR_START).sessions


def _nyse_sessions() -> pd.DatetimeIndex:
    return _sessions("XNYS")


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


def gap_runs(missing: list[date], calendar: str = "XNYS") -> list[tuple[date, date, int]]:
    """Group consecutive missing sessions of `calendar` into (first, last, n_sessions) runs."""
    sessions = list(_sessions(calendar).date)
    pos = {d: i for i, d in enumerate(sessions)}
    runs = []
    for d in missing:
        if runs and pos[d] == pos[runs[-1][1]] + 1:
            runs[-1] = (runs[-1][0], d, runs[-1][2] + 1)
        else:
            runs.append((d, d, 1))
    return runs


# ---------------------------------------------------------------- prices (TradingView exports)

def missing_sessions(index) -> dict:
    """CME sessions absent between min and max of index, split into regular days and NYSE-holiday Globex days."""
    days = sorted(index)
    cme = _sessions("CMES")
    nyse = {d.date() for d in _nyse_sessions()}
    window = [s.date() for s in cme[(cme >= pd.Timestamp(days[0])) & (cme <= pd.Timestamp(days[-1]))]]
    have = set(days)
    miss = [d for d in window if d not in have]
    return {"regular": [d for d in miss if d in nyse], "nyse_holiday": [d for d in miss if d not in nyse]}


def extra_sessions(index) -> list[date]:
    cme = {s.date() for s in _sessions("CMES")}
    return sorted(d for d in index if d not in cme)


def ohlc_violations(df: pd.DataFrame) -> list:
    o, h, l, c = (df[k] for k in ("open", "high", "low", "close"))
    bad = (l > o) | (l > c) | (h < o) | (h < c) | (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    return list(df.index[bad.to_numpy()])


def nan_bars(df: pd.DataFrame) -> list:
    """Bars with any missing OHLC value (NaN passes every comparison silently, so it is checked explicitly)."""
    return list(df.index[df[["open", "high", "low", "close"]].isna().any(axis=1).to_numpy()])


def off_tick_grid(df: pd.DataFrame, tick: float) -> list:
    """Bars with any price not on the tick grid (only meaningful for unadjusted series)."""
    x = df[["open", "high", "low", "close"]].to_numpy() / tick
    bad = (np.abs(x - np.round(x)) > 1e-6).any(axis=1)
    return list(df.index[bad])


def return_outliers(df: pd.DataFrame, k: float = 8.0) -> list:
    """Bars whose close-to-close log return deviates > k robust sigmas (MAD-based). Listed for review, never dropped."""
    r = np.log(df["close"]).diff().dropna()
    med = r.median()
    sigma = 1.4826 * (r - med).abs().median()
    if sigma == 0:
        return []
    return list(r.index[((r - med).abs() > k * sigma).to_numpy()])


def _third_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    return date(year, month, 1 + (4 - first.weekday()) % 7 + 14)


def detect_rolls(badj: pd.Series, raw: pd.Series, tol: float = 0.01) -> pd.DataFrame:
    """Roll sessions = steps in (back-adjusted - raw) close. Expected within 12 days before a quarterly 3rd-Friday expiry."""
    common = badj.index.intersection(raw.index)
    diff = (badj[common] - raw[common]).sort_index()
    step = diff.diff()
    rows = []
    for d, g in step[step.abs() > tol].items():
        expiry = next(e for e in (_third_friday(d.year + k, m) for k in (0, 1) for m in (3, 6, 9, 12)) if e >= d)
        rows.append({"session": d, "gap_points": float(g), "expiry": expiry,
                     "in_expected_window": 0 <= (expiry - d).days <= 12})
    return pd.DataFrame(rows, columns=["session", "gap_points", "expiry", "in_expected_window"])
