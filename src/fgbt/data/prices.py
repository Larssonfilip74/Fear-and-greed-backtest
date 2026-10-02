"""TradingView chart-export loader (DATA.md §2)."""
import io
import re
from datetime import date
import pandas as pd

from fgbt.data.validate import _sessions

EXCHANGE_TZ = "America/Chicago"
_NAME = re.compile(r"^(ES|NQ)1_(ETH|RTH)_(1W|1D|4H|1H|30m)_(badj|raw)\.csv$")
DAILY_OR_LONGER = {"1D", "1W"}


def parse_filename(name: str) -> dict:
    m = _NAME.match(name)
    if not m:
        raise ValueError(f"unexpected export file name {name!r}; see DATA.md §2")
    return {"symbol": m[1], "session": m[2], "timeframe": m[3], "adjusted": m[4] == "badj"}


def trading_date(ts_utc: pd.Timestamp) -> date:
    """CME trading date of a daily/weekly bar.

    TradingView may stamp a futures daily bar at local midnight of the trading date or at the 17:00 CT
    session open of the evening before. A stamp at/after 17:00 CT belongs to the next CME session.
    """
    local = ts_utc.tz_convert(EXCHANGE_TZ)
    d = local.date()
    if local.hour >= 17:
        sessions = _sessions("CMES")
        i = sessions.searchsorted(pd.Timestamp(d), side="right")
        return sessions[i].date()
    return d


def parse_tradingview_csv(text: str, timeframe: str) -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(text))
    if raw.empty:
        raise ValueError("export contains no bars")
    cols = {c: c.strip().lower() for c in raw.columns}
    raw = raw.rename(columns=cols)
    for need in ("time", "open", "high", "low", "close"):
        if need not in raw.columns:
            raise ValueError(f"missing column {need!r}; got {list(raw.columns)}")
    t = raw["time"]
    if pd.api.types.is_numeric_dtype(t):
        ts = pd.to_datetime(t.astype("int64"), unit="s", utc=True)
    else:
        ts = pd.to_datetime(t, utc=True)
    keep = ["open", "high", "low", "close"] + (["volume"] if "volume" in raw.columns else [])
    df = raw[keep].astype(float)
    if timeframe in DAILY_OR_LONGER:
        df.index = pd.Index([trading_date(x) for x in ts], name="session")
    else:
        df.index = pd.DatetimeIndex(ts, name="time")
    if df.index.duplicated().any():
        raise ValueError(f"duplicate bars: {list(df.index[df.index.duplicated()][:5])}")
    if not df.index.is_monotonic_increasing:
        raise ValueError("bars are not sorted by time")
    return df
