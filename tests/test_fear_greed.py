import json
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from fgbt.data.fear_greed import (
    ERA_SPLIT,
    build_series,
    implied_component_count,
    parse_cnn_json,
    parse_old_csv,
)
from fgbt.data.validate import missing_trading_days


def _ms(y, m, d, hh=0, mm=0, ss=0):
    return datetime(y, m, d, hh, mm, ss, tzinfo=timezone.utc).timestamp() * 1000


def _cnn_payload(points, current_ts="2026-10-01T23:59:56+00:00"):
    return {
        "fear_and_greed": {"score": points[-1][1], "timestamp": current_ts, "previous_close": 30.8},
        "fear_and_greed_historical": {"data": [{"x": x, "y": y, "rating": "fear"} for x, y in points]},
    }


def test_parse_cnn_json_tolerates_text_prefix_and_drops_live_point():
    pts = [(_ms(2026, 9, 30), 30.26), (_ms(2026, 10, 1), 28.09), (_ms(2026, 10, 1, 23, 59, 56), 28.09)]
    text = "Data retrieved successfully.\n" + json.dumps(_cnn_payload(pts))
    hist, current = parse_cnn_json(text)
    assert list(hist.index) == [date(2026, 9, 30), date(2026, 10, 1)]
    assert hist.loc[date(2026, 9, 30)] == pytest.approx(30.26)
    assert current["previous_close"] == 30.8


def test_parse_cnn_json_rejects_duplicate_dates():
    pts = [(_ms(2026, 9, 30), 30.0), (_ms(2026, 9, 30), 31.0)]
    with pytest.raises(ValueError, match="duplicate"):
        parse_cnn_json(json.dumps(_cnn_payload(pts)))


def test_parse_old_csv_accepts_iso_and_us_dates():
    s = parse_old_csv("Date,Fear Greed\n2011-01-03,68\n1/4/2011,67\n,\n")
    assert list(s.index) == [date(2011, 1, 3), date(2011, 1, 4)]
    assert list(s.values) == [68.0, 67.0]


def test_build_series_splits_eras_and_sets_availability():
    old = pd.Series({date(2021, 1, 28): 45.0, date(2021, 1, 29): 38.0, date(2021, 2, 1): 99.0})
    new = pd.Series({date(2021, 1, 29): 11.0, date(2021, 2, 1): 43.4})
    df = build_series(old, new)
    # OLD used strictly before ERA_SPLIT, NEW from ERA_SPLIT on; no mixing
    assert ERA_SPLIT == date(2021, 2, 1)
    assert df.loc[date(2021, 1, 29), "fg"] == 38.0 and df.loc[date(2021, 1, 29), "era"] == "OLD"
    assert df.loc[date(2021, 2, 1), "fg"] == 43.4 and df.loc[date(2021, 2, 1), "era"] == "NEW"
    assert df.loc[date(2021, 2, 1), "available_at"] == pd.Timestamp("2021-02-02 00:00", tz="UTC")


def test_build_series_rejects_out_of_range_values():
    with pytest.raises(ValueError, match="range"):
        build_series(pd.Series({date(2011, 1, 3): 101.0}), pd.Series(dtype=float))


@pytest.mark.parametrize("value,k", [(28.0857142857143, 7), (60.46666666666667, 6), (56.239999999999995, 5)])
def test_implied_component_count(value, k):
    assert implied_component_count(value) == k


def test_missing_trading_days_uses_nyse_calendar():
    # 2025-12-24..2025-12-29: Christmas (25th) is a holiday, 27/28 weekend
    idx = [date(2025, 12, 24), date(2025, 12, 29)]
    assert missing_trading_days(idx) == [date(2025, 12, 26)]
