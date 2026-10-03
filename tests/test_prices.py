from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from fgbt.data.prices import parse_filename, parse_tradingview_csv, trading_date
from fgbt.data.validate import (
    detect_rolls, extra_sessions, missing_sessions, ohlc_violations, off_tick_grid, return_outliers,
)


def _unix(y, m, d, hh=0, mm=0, tz="America/Chicago"):
    return int(pd.Timestamp(datetime(y, m, d, hh, mm)).tz_localize(tz).timestamp())


def test_parse_filename():
    assert parse_filename("ES1_ETH_1D_badj.csv") == {"symbol": "ES", "session": "ETH", "timeframe": "1D", "adjusted": True}
    assert parse_filename("NQ1_RTH_4H_raw.csv")["adjusted"] is False
    with pytest.raises(ValueError):
        parse_filename("es_daily.csv")


@pytest.mark.parametrize("ts,expected", [
    (_unix(2024, 3, 5, 0, 0), date(2024, 3, 5)),      # bar stamped at local midnight of the trading date
    (_unix(2024, 3, 4, 17, 0), date(2024, 3, 5)),     # bar stamped at the 17:00 CT session open of the next date
    (_unix(2024, 3, 8, 17, 0), date(2024, 3, 11)),    # Friday 17:00 CT is not a session open; next session is Monday
])
def test_trading_date_handles_both_daily_stamp_conventions(ts, expected):
    assert trading_date(pd.Timestamp(ts, unit="s", tz="UTC")) == expected


def test_parse_unix_csv_with_indicator_column():
    text = ("time,open,high,low,close,Volume,MA\n"
            f"{_unix(2024, 3, 5)},5100,5110,5090,5105,1000,1\n"
            f"{_unix(2024, 3, 6)},5105,5120,5100,5118,1200,2\n")
    df = parse_tradingview_csv(text, timeframe="1D")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert list(df.index) == [date(2024, 3, 5), date(2024, 3, 6)]
    assert df.loc[date(2024, 3, 6), "close"] == 5118


def test_parse_iso_csv_intraday_keeps_utc_timestamps():
    text = "time,open,high,low,close\n2024-03-05T14:30:00Z,1,2,0.5,1.5\n2024-03-05T18:30:00Z,1.5,2,1,1.75\n"
    df = parse_tradingview_csv(text, timeframe="4H")
    assert df.index[0] == pd.Timestamp("2024-03-05 14:30", tz="UTC")
    assert "volume" not in df.columns


def test_parse_rejects_duplicates_and_unsorted():
    dup = f"time,open,high,low,close\n{_unix(2024, 3, 5)},1,2,0.5,1\n{_unix(2024, 3, 5)},1,2,0.5,1\n"
    with pytest.raises(ValueError, match="duplicate"):
        parse_tradingview_csv(dup, timeframe="1D")
    uns = f"time,open,high,low,close\n{_unix(2024, 3, 6)},1,2,0.5,1\n{_unix(2024, 3, 5)},1,2,0.5,1\n"
    with pytest.raises(ValueError, match="sorted"):
        parse_tradingview_csv(uns, timeframe="1D")


def _bars(rows, start="2024-03-04"):
    idx = [d.date() for d in pd.bdate_range(start, periods=len(rows))]
    return pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close"], dtype=float)


def test_ohlc_violations_and_tick_grid():
    df = _bars([(100, 101, 99, 100.5), (100, 99, 101, 100), (100, 102, 98, 100.1), (0, 1, -1, 0.5)])
    bad = ohlc_violations(df)
    assert set(bad) == {df.index[1], df.index[3]}
    assert off_tick_grid(df, 0.25) == [df.index[2]]  # close 100.1 is not a multiple of 0.25


def test_missing_and_extra_sessions_against_cme():
    # 2024-07-03 .. 2024-07-09: CME trades the 4th of July (abbreviated), NYSE does not
    idx = [date(2024, 7, 3), date(2024, 7, 5), date(2024, 7, 6), date(2024, 7, 9)]
    miss = missing_sessions(idx)
    assert miss["regular"] == [date(2024, 7, 8)]
    assert miss["nyse_holiday"] == [date(2024, 7, 4)]
    assert extra_sessions(idx) == [date(2024, 7, 6)]  # Saturday


def test_return_outliers():
    closes = [100.0 + 0.1 * (i % 3) for i in range(300)]
    closes[200] = 140.0
    df = _bars([(c, c, c, c) for c in closes], start="2023-01-02")
    out = return_outliers(df, k=8)
    assert df.index[200] in out and df.index[201] in out


def test_detect_rolls_finds_known_gap_in_expected_window():
    # raw front-month series vs back-adjusted: badj - raw steps by +12.5 at the 2024-03-14 roll
    idx = [d.date() for d in pd.bdate_range("2024-03-01", "2024-03-29")]
    raw = pd.Series([5000.0 + i for i in range(len(idx))], index=idx)
    adj_shift = pd.Series([0.0 if d < date(2024, 3, 14) else 12.5 for d in idx], index=idx)
    badj = raw - 12.5 + adj_shift  # history shifted down by the gap, current contract unadjusted
    rolls = detect_rolls(badj, raw, tol=0.01)
    assert list(rolls["session"]) == [date(2024, 3, 14)]
    assert rolls.iloc[0]["gap_points"] == pytest.approx(-12.5)  # badj-raw stepped +12.5: new contract 12.5 lower
    assert bool(rolls.iloc[0]["in_expected_window"]) is True


@pytest.mark.parametrize("y,m,expected", [(2024, 3, date(2024, 3, 15)), (2024, 6, date(2024, 6, 21)),
                                          (2025, 12, date(2025, 12, 19)), (2026, 5, date(2026, 5, 15))])
def test_third_friday(y, m, expected):
    from fgbt.data.validate import _third_friday
    assert _third_friday(y, m) == expected


def test_roll_after_december_expiry_maps_to_next_march():
    idx = [d.date() for d in pd.bdate_range("2024-12-20", "2025-01-10")]
    raw = pd.Series(5000.0, index=idx)
    badj = raw - pd.Series([5.0 if d < date(2024, 12, 27) else 0.0 for d in idx], index=idx)
    r = detect_rolls(badj, raw)
    assert r.iloc[0]["expiry"] == date(2025, 3, 21) and not r.iloc[0]["in_expected_window"]


def test_pre_2005_bars_get_distinct_trading_dates():
    a = trading_date(pd.Timestamp(_unix(1999, 3, 4, 17, 0), unit="s", tz="UTC"))
    b = trading_date(pd.Timestamp(_unix(1999, 3, 5, 0, 0), unit="s", tz="UTC"))
    assert a == b == date(1999, 3, 5)
    assert trading_date(pd.Timestamp(_unix(1999, 3, 5, 17, 0), unit="s", tz="UTC")) == date(1999, 3, 8)


def test_empty_export_rejected_and_nan_bars_detected():
    from fgbt.data.validate import nan_bars
    with pytest.raises(ValueError, match="no bars"):
        parse_tradingview_csv("time,open,high,low,close\n", timeframe="1D")
    df = parse_tradingview_csv(f"time,open,high,low,close\n{_unix(2024, 3, 5)},1,2,0.5,\n", timeframe="1D")
    assert nan_bars(df) == [date(2024, 3, 5)]
    assert ohlc_violations(df) == []  # why the explicit NaN check is needed



def test_ratio_adjust_returns_equal_true_contract_moves_across_a_roll():
    from fgbt.data.prices import ratio_adjust
    idx = [d.date() for d in pd.bdate_range("2024-03-11", periods=5)]
    # old contract days 0-2; roll on day 3 into a contract trading G=50 higher (contango): new was 4090 on day 2
    raw_close = [4000.0, 4020.0, 4040.0, 4130.9, 4150.0]
    raw = pd.DataFrame({k: raw_close for k in ("open", "high", "low", "close")}, index=idx)
    badj = raw.copy()
    badj.iloc[:3] = raw.iloc[:3] + 50.0  # Panama back-adjustment raises history onto the new contract level
    ra = ratio_adjust(badj, raw)
    r = np.log(ra["close"]).diff().to_numpy()
    assert bool(ra["roll"].iloc[3]) and ra["roll"].sum() == 1
    assert ra["close"].iloc[-1] == pytest.approx(4150.0)  # latest prices unchanged
    assert r[3] == pytest.approx(np.log(4130.9 / 4090.0))  # the new contract's own move, no roll jump
    assert r[1] == pytest.approx(np.log(4020 / 4000))  # pre-roll returns are the old contract's true moves
    assert r[4] == pytest.approx(np.log(4150.0 / 4130.9))
    # and the roll detector reports the gap as new - old = +50
    assert detect_rolls(badj["close"], raw["close"]).iloc[0]["gap_points"] == pytest.approx(50.0)
