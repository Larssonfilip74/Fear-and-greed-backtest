from datetime import date, datetime, timezone

import pytest

from fgbt.instruments import INSTRUMENTS, round_trip_cost_usd
from fgbt.data.align import fg_available_at, first_rth_open_after


def test_contract_specs():
    es, nq = INSTRUMENTS["ES"], INSTRUMENTS["NQ"]
    assert (es.point_value, es.tick_size, es.tick_value) == (50.0, 0.25, 12.5)
    assert (nq.point_value, nq.tick_size, nq.tick_value) == (20.0, 0.25, 5.0)
    assert INSTRUMENTS["MES"].tick_value == 1.25
    assert INSTRUMENTS["MNQ"].tick_value == 0.5


def test_round_trip_cost_base_and_stress():
    # base: $5 commission+fees RT + 1 tick slippage per side on ES = 5 + 2*12.5
    assert round_trip_cost_usd("ES", slippage_ticks_per_side=1) == pytest.approx(30.0)
    assert round_trip_cost_usd("ES", slippage_ticks_per_side=3) == pytest.approx(80.0)


def test_fg_value_available_only_after_midnight_utc_next_day():
    # F&G for Wed 2026-09-30 is final at 00:00 UTC Thu 2026-10-01 (20:00 ET, EDT).
    assert fg_available_at(date(2026, 9, 30)) == datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc)


def test_rth_trader_earliest_fill_is_next_rth_open_summer_and_winter():
    # EDT: RTH open 09:30 ET = 13:30 UTC
    assert first_rth_open_after(fg_available_at(date(2026, 9, 30))) == datetime(
        2026, 10, 1, 13, 30, tzinfo=timezone.utc)
    # EST: RTH open 09:30 ET = 14:30 UTC
    assert first_rth_open_after(fg_available_at(date(2025, 12, 1))) == datetime(
        2025, 12, 2, 14, 30, tzinfo=timezone.utc)


def test_friday_signal_fills_monday_and_skips_holidays():
    # Fri 2025-11-28 (day after Thanksgiving) -> Mon 2025-12-01 open
    assert first_rth_open_after(fg_available_at(date(2025, 11, 28))) == datetime(
        2025, 12, 1, 14, 30, tzinfo=timezone.utc)
    # Wed 2025-12-24 -> Christmas closed -> Fri 2025-12-26
    assert first_rth_open_after(fg_available_at(date(2025, 12, 24))) == datetime(
        2025, 12, 26, 14, 30, tzinfo=timezone.utc)


def test_never_fills_before_availability():
    avail = datetime(2026, 10, 1, 13, 31, tzinfo=timezone.utc)  # 1 minute after the open
    assert first_rth_open_after(avail) == datetime(2026, 10, 2, 13, 30, tzinfo=timezone.utc)
