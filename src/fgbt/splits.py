"""Data splits (protocol §4). The holdout is locked in code."""
from datetime import date

import pandas as pd

DEV = (date(2011, 1, 3), date(2020, 12, 31))
VAL = (date(2021, 2, 1), date(2024, 12, 31))
HOLDOUT_START = date(2025, 1, 1)
HOLDOUT_END = date(2026, 9, 30)


class HoldoutLockedError(RuntimeError):
    pass


def research_view(df: pd.DataFrame, unlock_holdout: bool = False) -> pd.DataFrame:
    """Return the rows a research step may use; raise if holdout rows are requested while locked.

    Index must be dates (datetime.date or Timestamp). Data-integrity checks (no returns, no signals)
    are the only callers allowed to bypass this, and must say so in their report.
    """
    if unlock_holdout:
        return df
    idx = pd.to_datetime(pd.Index(df.index))
    return df[idx < pd.Timestamp(HOLDOUT_START)]


def segment(d) -> str:
    d = pd.Timestamp(d).date()
    if DEV[0] <= d <= DEV[1]:
        return "DEV"
    if VAL[0] <= d <= VAL[1]:
        return "VAL"
    if HOLDOUT_START <= d <= HOLDOUT_END:
        return "HOLD"
    return "NONE"


def assert_no_holdout(index, unlock_holdout: bool = False) -> None:
    if unlock_holdout:
        return
    idx = pd.to_datetime(pd.Index(index))
    if (idx >= pd.Timestamp(HOLDOUT_START)).any():
        raise HoldoutLockedError("Holdout data (>= 2025-01-01) reached a research step while locked")
