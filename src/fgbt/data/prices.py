"""TradingView chart-export loader (DATA.md §2)."""
import io
import re
from datetime import date
import numpy as np
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


def ratio_adjust(badj: pd.DataFrame, raw: pd.DataFrame, tol: float = 0.01) -> pd.DataFrame:
    """Ratio-adjusted OHLC from an additive back-adjusted / unadjusted pair (same sessions).

    Offset D = badj - raw close is constant between rolls and steps at each roll session s. Continuity of the
    back-adjusted series gives new_{s-1} = old_{s-1} - step (old = raw close of session s-1, still the old contract),
    so the ratio factor for that roll is new/old = (old - step) / old.
    History before s is multiplied by the product of all later factors, so every daily return equals the
    held contract's true percentage move and the latest prices equal the raw prices.
    """
    common = badj.index.intersection(raw.index)
    b, r = badj.loc[common].sort_index(), raw.loc[common].sort_index()
    offset = (b["close"] - r["close"]).to_numpy()
    step = np.r_[0.0, np.diff(offset)]
    close = r["close"].to_numpy()
    factor_at = np.ones(len(common))
    for i in np.nonzero(np.abs(step) > tol)[0]:
        factor_at[i] = (close[i - 1] - step[i]) / close[i - 1]
    # multiplier for session t = product of roll factors at sessions strictly after t
    after = np.r_[np.cumprod(factor_at[::-1])[::-1][1:], 1.0]
    out = r[["open", "high", "low", "close"]].mul(after, axis=0)
    out["roll"] = np.abs(step) > tol
    return out
