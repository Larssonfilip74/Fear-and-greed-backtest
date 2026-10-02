"""Daily-bar, event-driven backtest engine (protocol §3, §6).

Timeline of one session i:
  1. decision point (pre-open): strategy sees closed bars [0, i) and F&G rows available before the open of i
  2. roll cost charged if session i is a roll session and a position is carried in
  3. MOO fills -> 4. working LIMIT/STOP buys (intraday) -> 5. MOC fills
  6. mark-to-market at the close
No-look-ahead holds by construction: nothing else is ever passed to the strategy.
The strategy returns Orders and/or Cancel(tag) objects; a Cancel removes working LIMIT/STOP orders before fills.
MAE/MFE: for lots bought intraday (LIMIT/STOP) the whole entry-day range is counted. Without intraday data the
path is unknown, so this is the conservative choice (MAE may be overstated, never understated).
Prices must be back-adjusted (continuous P&L across rolls); roll costs are charged explicitly.
"""
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

import pandas as pd

from fgbt.backtest.costs import CostModel
from fgbt.backtest.orders import Cancel, Fill, Order
from fgbt.data.align import rth_open_map


@dataclass(frozen=True)
class Context:
    i: int
    session: date
    open_time: pd.Timestamp
    bars: pd.DataFrame  # closed bars only (strictly before `session`)
    fg: pd.DataFrame | None  # F&G rows with available_at <= open_time
    position: int
    pending: tuple  # working LIMIT/STOP orders carried from earlier sessions


@dataclass
class _Lot:
    entry_session: date
    entry_idx: int
    entry_price: float
    qty: int
    tag: str
    low: float
    high: float
    roll_cost_per_contract: float = 0.0
    entered_at_close: bool = False


@dataclass
class Result:
    equity: pd.Series  # cumulative net P&L in USD, marked at each close
    position: pd.Series  # contracts held at each close
    trades: pd.DataFrame
    fills: list = field(default_factory=list)


TRADE_COLUMNS = ["entry_session", "exit_session", "qty", "entry_price", "exit_price", "points", "gross_usd",
                 "costs_usd", "net_usd", "holding_sessions", "mae_points", "mfe_points", "tag"]


def run(bars: pd.DataFrame, strategy: Callable[[Context], list], fg: pd.DataFrame | None = None,
        symbol: str = "ES", costs: CostModel | None = None) -> Result:
    costs = costs or CostModel.for_symbol(symbol)
    pv = costs.inst.point_value
    tick = costs.inst.tick_size
    opens_map = rth_open_map()
    sessions = [pd.Timestamp(d).date() for d in bars.index]  # accept date or Timestamp index
    for d in sessions:
        if d not in opens_map:
            raise ValueError(f"{d} is not an NYSE session")
    if sessions != sorted(sessions) or len(set(sessions)) != len(sessions):
        raise ValueError("bars index must be strictly increasing")
    o, h, l, c = (bars[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    roll = bars["roll"].to_numpy(dtype=bool) if "roll" in bars else [False] * len(bars)
    fg_sorted, fg_avail = None, None
    if fg is not None:
        fg_sorted = fg.sort_index()
        if not fg_sorted["available_at"].is_monotonic_increasing:
            raise ValueError("fg.available_at must increase with the date index")
        fg_avail = pd.DatetimeIndex(fg_sorted["available_at"])

    lots: list[_Lot] = []
    working: list[tuple[Order, int]] = []  # (order, last session index it may fill)
    cash = 0.0  # realized P&L incl. all costs paid so far
    fills, trades, equity, position = [], [], [], []

    def buy(i, raw_price, order, slip=True):
        nonlocal cash
        px = costs.slipped(raw_price, "BUY") if slip else raw_price
        cash -= costs.commission_per_side * order.qty
        lots.append(_Lot(sessions[i], i, px, order.qty, order.tag, low=float("inf"), high=float("-inf"),
                         entered_at_close=order.kind == "MOC"))
        fills.append(Fill(sessions[i], "BUY", order.kind, order.qty, px, order.tag))
        return px

    def sell(i, raw_price, order, held_through_today):
        nonlocal cash
        qty = order.qty
        if qty > sum(lt.qty for lt in lots):
            raise ValueError(f"long-only: cannot sell {qty} on {sessions[i]} with position {sum(x.qty for x in lots)}")
        px = costs.slipped(raw_price, "SELL")
        fills.append(Fill(sessions[i], "SELL", order.kind, qty, px, order.tag))
        while qty:
            lot = lots[0]
            q = min(qty, lot.qty)
            low, high = lot.low, lot.high
            if held_through_today:  # MOC exit: today's range was experienced while holding
                low, high = min(low, l[i]), max(high, h[i])
            points = px - lot.entry_price
            comm = 2 * costs.commission_per_side * q
            roll_usd = lot.roll_cost_per_contract * q
            gross = points * pv * q
            cash += gross - costs.commission_per_side * q
            trades.append([lot.entry_session, sessions[i], q, lot.entry_price, px, points, gross, comm + roll_usd,
                           gross - comm - roll_usd, i - lot.entry_idx,
                           (low - lot.entry_price) if low != float("inf") else min(0.0, points),
                           (high - lot.entry_price) if high != float("-inf") else max(0.0, points), lot.tag])
            lot.qty -= q
            qty -= q
            if lot.qty == 0:
                lots.pop(0)

    for i, s in enumerate(sessions):
        open_time = opens_map[s]
        fg_view = None
        if fg_sorted is not None:
            k = int(fg_avail.searchsorted(open_time, side="right"))
            fg_view = fg_sorted.iloc[:k]
        ctx = Context(i, s, open_time, bars.iloc[:i], fg_view, sum(lt.qty for lt in lots),
                      tuple(od for od, _ in working))
        out = list(strategy(ctx) or [])
        for od in out:
            if not isinstance(od, (Order, Cancel)):
                raise TypeError(f"strategy returned {od!r}")
        for cx in (x for x in out if isinstance(x, Cancel)):  # 2. cancels act before anything fills
            working = [(od, last) for od, last in working if cx.tag is not None and od.tag != cx.tag]
        new_orders = [x for x in out if isinstance(x, Order)]

        for od in new_orders:  # 3. MOO
            if od.kind == "MOO":
                buy(i, o[i], od) if od.side == "BUY" else sell(i, o[i], od, held_through_today=False)

        if roll[i]:  # roll cost only for contracts carried in from yesterday and still held after the open
            carried = [lt for lt in lots if lt.entry_idx < i]
            for lot in carried:
                lot.roll_cost_per_contract += costs.roll_cost_per_contract
            cash -= costs.roll_cost_per_contract * sum(lt.qty for lt in carried)
        working += [(od, i + od.valid_sessions - 1) for od in new_orders if od.kind in ("LIMIT", "STOP")]

        still = []  # 4. intraday LIMIT/STOP buys
        for od, last in working:
            if od.kind == "LIMIT":
                if o[i] <= od.price - tick:  # opened through the limit
                    buy(i, o[i], od, slip=False)
                    continue
                if l[i] <= od.price - tick:
                    buy(i, od.price, od, slip=False)
                    continue
            else:  # STOP
                if h[i] >= od.price:
                    buy(i, max(o[i], od.price), od)
                    continue
            if last > i:
                still.append((od, last))
        working = still

        for od in new_orders:  # 5. MOC
            if od.kind == "MOC":
                buy(i, c[i], od) if od.side == "BUY" else sell(i, c[i], od, held_through_today=True)

        for lot in lots:  # range experienced while holding (a lot bought at today's close saw none of it)
            if not (lot.entered_at_close and lot.entry_idx == i):
                lot.low, lot.high = min(lot.low, l[i]), max(lot.high, h[i])
        unreal = sum((c[i] - lt.entry_price) * pv * lt.qty for lt in lots)
        equity.append(cash + unreal)
        position.append(sum(lt.qty for lt in lots))

    idx = pd.Index(sessions, name="session")
    tr = pd.DataFrame(trades, columns=TRADE_COLUMNS).astype(
        {"qty": "int64", "holding_sessions": "int64", "tag": "str",
         **{k: "float64" for k in ("entry_price", "exit_price", "points", "gross_usd", "costs_usd", "net_usd",
                                   "mae_points", "mfe_points")}})
    return Result(pd.Series(equity, index=idx, name="equity_usd"), pd.Series(position, index=idx, name="position"),
                  tr, fills)
