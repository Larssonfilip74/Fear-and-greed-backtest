"""Engine tests. ES: $50/pt, tick 0.25 = $12.50, commission $5.00 RT -> $2.50 per side per contract."""
from datetime import date

import pandas as pd
import pytest

from conftest import make_bars
from fgbt.backtest.engine import run
from fgbt.backtest.orders import Order

FLAT = (100.0, 101.0, 99.0, 100.0)


def scripted(plan):
    """Strategy returning the orders in plan[i] at the decision point before session i."""
    def decide(ctx):
        return plan.get(ctx.i, [])
    return decide


def test_moo_round_trip_exact_usd():
    bars = make_bars([(100, 102, 99, 101), (103, 104, 102, 103), (105, 106, 104, 105)])
    res = run(bars, scripted({0: [Order("BUY", "MOO", 1)], 2: [Order("SELL", "MOO", 1)]}))
    t = res.trades.iloc[0]
    assert t.entry_price == 100.25 and t.exit_price == 104.75  # 1 tick adverse each side
    assert t.gross_usd == pytest.approx(4.5 * 50)  # 225.00
    assert t.costs_usd == pytest.approx(5.00)
    assert t.net_usd == pytest.approx(220.00)
    assert t.holding_sessions == 2
    assert res.equity.iloc[-1] == pytest.approx(220.00)


def test_limit_touch_does_not_fill_trade_through_does():
    bars = make_bars([FLAT, (100, 101, 98.0, 100), (100, 101, 97.75, 100)])
    res = run(bars, scripted({1: [Order("BUY", "LIMIT", 1, price=98.0, valid_sessions=2)]}))
    # session 1 low == 98.00 (touch only) -> no fill; session 2 low 97.75 = limit - 1 tick -> fill at 98.00
    assert len(res.fills) == 1
    f = res.fills[0]
    assert f.session == bars.index[2] and f.price == 98.0


def test_limit_gap_below_fills_at_open_and_expires():
    bars = make_bars([FLAT, (97.0, 99, 96, 98), FLAT, FLAT])
    res = run(bars, scripted({1: [Order("BUY", "LIMIT", 1, price=98.0, valid_sessions=1)]}))
    assert res.fills[0].price == 97.0  # opened through the limit: better price, no slippage
    res2 = run(bars, scripted({2: [Order("BUY", "LIMIT", 1, price=95.0, valid_sessions=2)]}))
    assert res2.fills == []  # never traded through within 2 sessions, then expired


def test_stop_buy_gap_intraday_and_no_fill():
    gap = make_bars([FLAT, (103, 104, 102, 103)])
    r = run(gap, scripted({1: [Order("BUY", "STOP", 1, price=101.0)]}))
    assert r.fills[0].price == 103.25  # max(open, stop) + 1 tick
    intra = make_bars([FLAT, (100, 102, 99, 101)])
    r = run(intra, scripted({1: [Order("BUY", "STOP", 1, price=101.5)]}))
    assert r.fills[0].price == 101.75
    none = make_bars([FLAT, (100, 101.25, 99, 101)])
    assert run(none, scripted({1: [Order("BUY", "STOP", 1, price=101.5)]})).fills == []


def test_moc_fill_uses_close():
    bars = make_bars([(100, 102, 99, 101.5), FLAT])
    r = run(bars, scripted({0: [Order("BUY", "MOC", 1)]}))
    assert r.fills[0].price == 101.75


def test_roll_cost_only_while_holding():
    # roll on sessions 1 and 3; position held over session 1 only
    bars = make_bars([FLAT] * 5, roll_on={1, 3})
    r = run(bars, scripted({0: [Order("BUY", "MOO", 1)], 2: [Order("SELL", "MOO", 1)]}))
    t = r.trades.iloc[0]
    assert t.costs_usd == pytest.approx(5.00 + 12.50 + 5.00)  # commissions + one roll (1 tick + $5)
    # P&L: buy 100.25, sell 99.75 -> -0.5 pt = -25 ; net = -25 - 22.5
    assert t.net_usd == pytest.approx(-47.50)
    assert r.equity.iloc[-1] == pytest.approx(-47.50)


def test_tranches_average_entry_and_fifo_exits():
    bars = make_bars([(100, 101, 99, 100), (95, 96, 94, 95), (90, 91, 89, 90), (110, 111, 109, 110)])
    plan = {0: [Order("BUY", "MOO", 1, tag="t1")], 1: [Order("BUY", "MOO", 1, tag="t2")],
            2: [Order("BUY", "MOO", 2, tag="t3")], 3: [Order("SELL", "MOO", 3)]}
    r = run(bars, scripted(plan))
    tr = r.trades
    assert list(tr.tag) == ["t1", "t2", "t3"]  # FIFO: t1, t2 fully, then 1 of the 2 t3 contracts
    assert list(tr.qty) == [1, 1, 1]
    assert list(tr.entry_price) == [100.25, 95.25, 90.25]
    assert (tr.exit_price == 109.75).all()
    assert r.position.iloc[-1] == 1  # one t3 contract still open
    # open lot marked at last close 110: (110 - 90.25)*50 - 2.5 entry commission
    realized = ((109.75 - 100.25) + (109.75 - 95.25) + (109.75 - 90.25)) * 50 - 3 * 5.0
    assert r.equity.iloc[-1] == pytest.approx(realized + (110 - 90.25) * 50 - 2.5)


def test_daily_mark_to_market_and_mae_mfe():
    bars = make_bars([(100, 101, 99, 100.5), (100.5, 103, 98, 102), (102, 104, 101, 103.5), (104, 105, 103, 104)])
    r = run(bars, scripted({0: [Order("BUY", "MOO", 1)], 3: [Order("SELL", "MOO", 1)]}))
    entry = 100.25
    expected = [(100.5 - entry) * 50 - 2.5, (102 - entry) * 50 - 2.5, (103.5 - entry) * 50 - 2.5,
                (103.75 - entry) * 50 - 5.0]
    assert list(r.equity.round(6)) == pytest.approx(expected)
    t = r.trades.iloc[0]
    # held through sessions 0..2 (MOO exit on 3 excludes session 3): low 98, high 104
    assert t.mae_points == pytest.approx(98 - entry)
    assert t.mfe_points == pytest.approx(104 - entry)


def test_long_only_guard():
    bars = make_bars([FLAT, FLAT])
    with pytest.raises(ValueError, match="long-only"):
        run(bars, scripted({1: [Order("SELL", "MOO", 1)]}))


def test_context_sees_only_closed_bars_and_available_fg():
    bars = make_bars([FLAT] * 4, start="2024-01-02")  # Tue..Fri
    fg = pd.DataFrame({"fg": [10.0, 20.0, 30.0, 40.0]}, index=bars.index)
    fg["available_at"] = [pd.Timestamp(d) + pd.Timedelta(days=1) for d in fg.index]
    fg["available_at"] = fg["available_at"].dt.tz_localize("UTC")
    seen = []

    def decide(ctx):
        seen.append((ctx.i, len(ctx.bars), list(ctx.fg.index)))
        assert ctx.bars.index.max() < ctx.session if len(ctx.bars) else True
        return []

    run(bars, decide, fg=fg)
    # before session i: exactly i closed bars; F&G of every earlier session (available 00:00 UTC next day)
    assert [s[1] for s in seen] == [0, 1, 2, 3]
    assert seen[2][2] == list(bars.index[:2])


def test_reading_unavailable_fg_fails():
    bars = make_bars([FLAT] * 3)
    fg = pd.DataFrame({"fg": [10.0, 20.0, 30.0]}, index=bars.index)
    fg["available_at"] = [pd.Timestamp(d, tz="UTC") + pd.Timedelta(days=1) for d in fg.index]

    def cheat(ctx):
        ctx.fg.loc[ctx.session]  # today's value is not known before today's open
        return []

    with pytest.raises(KeyError):
        run(bars, cheat, fg=fg)


def test_bars_must_be_nyse_sessions():
    bars = make_bars([FLAT, FLAT])
    bars.index = [date(2024, 1, 6), date(2024, 1, 7)]  # weekend
    with pytest.raises(ValueError, match="NYSE session"):
        run(bars, scripted({}))


def test_moc_entry_excludes_entry_day_range_from_mae():
    bars = make_bars([(100, 120, 80, 100), (100, 102, 99, 101), (101, 103, 100, 102)])
    r = run(bars, scripted({0: [Order("BUY", "MOC", 1)], 2: [Order("SELL", "MOO", 1)]}))
    t = r.trades.iloc[0]
    # bought at the close of session 0 (100.25): its 80-120 range must not count; held through session 1 only
    assert t.mae_points == pytest.approx(99 - 100.25)
    assert t.mfe_points == pytest.approx(102 - 100.25)


def test_limit_open_equal_to_limit_is_touch_not_fill():
    bars = make_bars([FLAT, (98.0, 100, 98.0, 99)])
    assert run(bars, scripted({1: [Order("BUY", "LIMIT", 1, price=98.0)]})).fills == []


def test_timestamp_index_accepted():
    bars = make_bars([(100, 102, 99, 101), (103, 104, 102, 103)])
    bars.index = pd.to_datetime(bars.index)
    r = run(bars, scripted({0: [Order("BUY", "MOO", 1)]}))
    assert r.position.iloc[-1] == 1


def test_no_roll_cost_when_sold_at_open_of_roll_day_or_bought_that_day():
    bars = make_bars([FLAT] * 3, roll_on={1})
    r = run(bars, scripted({0: [Order("BUY", "MOO", 1)], 1: [Order("SELL", "MOO", 1)]}))
    assert r.trades.iloc[0].costs_usd == pytest.approx(5.00)
    r = run(bars, scripted({1: [Order("BUY", "MOO", 1)], 2: [Order("SELL", "MOO", 1)]}))
    assert r.trades.iloc[0].costs_usd == pytest.approx(5.00)


def test_cancel_removes_working_orders_before_fills():
    from fgbt.backtest.orders import Cancel
    bars = make_bars([FLAT, FLAT, (100, 101, 90, 95)])
    plan = {0: [Order("BUY", "LIMIT", 1, price=95.0, valid_sessions=5, tag="dip")], 2: [Cancel("dip")]}
    assert run(bars, scripted(plan)).fills == []
    plan[2] = [Cancel("other")]
    assert len(run(bars, scripted(plan)).fills) == 1
    plan[2] = [Cancel()]
    assert run(bars, scripted(plan)).fills == []
