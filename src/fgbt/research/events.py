"""Event definitions and event-study measurements (protocol §5, H2/H3).

Timing: an event on F&G date d is tradable at the open of the first bar session strictly after d
(F&G(d) is available at 00:00 UTC d+1, before any later NYSE open; see fgbt.data.align).
"""
from typing import Callable

import numpy as np
import pandas as pd

from fgbt.validation.bootstrap import SEED


def independent_events(fg: pd.Series, cond: Callable[[pd.Series], pd.Series], sep: int = 20) -> list:
    """First session of each episode (protocol §5): cond true after >= sep observed sessions with cond false.

    Separation counts observed rows only. A data hole never manufactures 'false' days, so episodes either side of a
    gap are merged, the conservative direction (fewer, more independent events). The series start also needs
    sep observed false days, so an episode already running when the data begins is never counted as new.
    Run on the full research series and filter event dates by segment afterwards (never slice first).
    """
    flags = cond(fg).to_numpy()
    out, last_true = [], -1
    for i, f in enumerate(flags):
        if f:
            if i - last_true - 1 >= sep:
                out.append(fg.index[i])
            last_true = i
    return out


def cross_back_events(fg: pd.Series, threshold: float, sep: int = 20) -> list:
    """H3: F&G crosses back to >= threshold after being below it on the previous session.

    ('below T within the last 20 days' in the protocol is implied by 'below T yesterday'.) Independence is
    measured from the last cross-back of any kind, accepted or suppressed, with the same >= sep false-days
    rule as independent_events.
    """
    v = fg.to_numpy()
    crosses = pd.Series(np.r_[False, (v[1:] >= threshold) & (v[:-1] < threshold)], index=fg.index)
    return independent_events(crosses, lambda c: c, sep=sep)


def forward_returns(event_dates, bars: pd.DataFrame, h: int, *, rth_bars: bool, ratio_comparable: bool) -> pd.DataFrame:
    """Open-to-open log return over h sessions from the first NYSE RTH open after each event, with MAE/MFE.

    Guards (protocol R1, §5):
      - rth_bars must be True: bar opens must be the 09:30 ET RTH open. A Globex/ETH daily bar opens the evening
        before, i.e. before F&G(d) is available at 00:00 UTC d+1.
      - ratio_comparable must be True: log returns need ratio-adjusted (or roll-free) prices, not the additive
        back-adjusted series (use fgbt.data.prices.ratio_adjust).
      - bars must be NYSE sessions. The entry is the first NYSE session after the event date; if that session is
        missing from bars the event is dropped (counted in .attrs['dropped']), never shifted to a later bar.
    MAE/MFE use lows/highs of the sessions held (entry session .. session before the exit open).
    """
    if not rth_bars:
        raise ValueError("forward_returns needs RTH bars (opens at 09:30 ET); ETH opens precede F&G availability")
    if not ratio_comparable:
        raise ValueError("forward_returns needs ratio-comparable prices; see fgbt.data.prices.ratio_adjust")
    from fgbt.data.align import rth_open_map

    nyse = np.array(sorted(rth_open_map()))
    idx = pd.Index([pd.Timestamp(d).date() for d in bars.index])
    extra = [d for d in idx if d not in rth_open_map()]
    if extra:
        raise ValueError(f"bars contain non-NYSE sessions, e.g. {extra[:3]}")
    pos = {d: i for i, d in enumerate(idx)}
    o, hi, lo = bars["open"].to_numpy(float), bars["high"].to_numpy(float), bars["low"].to_numpy(float)
    rows, dropped = [], 0
    for d in event_dates:
        d = pd.Timestamp(d).date()
        k = int(np.searchsorted(nyse, d, side="right"))
        nxt = nyse[k] if k < len(nyse) else None
        if nxt not in pos:
            dropped += 1
            continue
        e = pos[nxt]
        x = e + h
        if x >= len(idx):
            dropped += 1
            continue
        rows.append({"event": d, "entry_session": idx[e], "exit_session": idx[x],
                     "log_return": float(np.log(o[x] / o[e])),
                     "mae": float(np.log(lo[e:x].min() / o[e])), "mfe": float(np.log(hi[e:x].max() / o[e]))})
    out = pd.DataFrame(rows, columns=["event", "entry_session", "exit_session", "log_return", "mae", "mfe"])
    out.attrs["dropped"] = dropped
    return out


def all_forward_returns(bars: pd.DataFrame, h: int) -> np.ndarray:
    """Drift benchmark: every h-session open-to-open log return available in `bars`."""
    o = bars["open"].to_numpy(float)
    return np.log(o[h:] / o[:-h])


def random_entry_pvalue(observed_mean: float, bars: pd.DataFrame, n_events: int, h: int, sims: int = 10000,
                        seed: int = SEED) -> float:
    """One-sided p-value: share of random n_events-entry samples (same h, same bars window) with mean >= observed.

    This is the exposure-matched random-entry null of protocol §9 for fixed-horizon events.
    """
    pool = all_forward_returns(bars, h)
    rng = np.random.default_rng(seed)
    draws = pool[rng.integers(0, len(pool), (sims, n_events))].mean(axis=1)
    return float((np.sum(draws >= observed_mean) + 1) / (sims + 1))
