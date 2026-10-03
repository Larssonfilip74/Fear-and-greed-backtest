"""Point-in-time availability rules (RESEARCH_PROTOCOL.md rule R1).

CNN stamps the F&G value for trading date d at 00:00 UTC of d, but its components keep updating
until ~23:59 UTC of d. The final value is therefore treated as known only at 00:00 UTC of d+1.
The user trades the US cash session only, so the earliest fill is the next NYSE regular open.
"""
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd


def fg_available_at(fg_date: date) -> datetime:
    return datetime.combine(fg_date + timedelta(days=1), time(0, 0), tzinfo=timezone.utc)


@lru_cache(maxsize=1)
def _rth_opens() -> pd.Series:
    cal = xcals.get_calendar("XNYS", start="2005-01-01")
    return cal.schedule["open"]  # tz-aware UTC, one row per session (holidays excluded)


@lru_cache(maxsize=1)
def rth_open_map() -> dict:
    """NYSE session date -> regular-session open (UTC). Single source for the engine and availability rules."""
    opens = _rth_opens()
    return {ts.date(): o for ts, o in zip(opens.index, opens)}


def first_rth_open_after(ts: datetime) -> datetime:
    """First NYSE regular-session open at or after `ts` (UTC-aware)."""
    opens = _rth_opens()
    idx = opens.searchsorted(pd.Timestamp(ts), side="left")
    if idx >= len(opens):
        raise ValueError(f"No RTH open after {ts}; extend the calendar")
    return opens.iloc[idx].to_pydatetime()
