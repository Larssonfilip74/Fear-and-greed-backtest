"""Performance metrics (protocol §10). Returns are daily P&L over a fixed capital base (no compounding)."""
import math

import numpy as np
import pandas as pd

ANN = 252


def equity_metrics(equity_usd: pd.Series, capital: float, position: pd.Series | None = None) -> dict:
    eq = equity_usd.to_numpy(dtype=float)
    r = np.diff(np.r_[0.0, eq]) / capital
    sd = r.std(ddof=1) if len(r) > 1 else float("nan")
    downside = math.sqrt(np.mean(np.minimum(r, 0) ** 2)) if len(r) else float("nan")
    peak = np.maximum.accumulate(np.r_[0.0, eq])[1:]
    dd = peak - eq
    dd_pct = dd / (capital + peak)
    under, longest = 0, 0
    for x in dd:
        under = under + 1 if x > 0 else 0
        longest = max(longest, under)
    years = len(eq) / ANN
    end_wealth = (capital + eq[-1]) / capital if len(eq) else 1.0
    cagr = end_wealth ** (1 / years) - 1 if years > 0 and end_wealth > 0 else float("nan")
    max_dd_pct = float(dd_pct.max()) if len(dd) else 0.0
    return {
        "net_usd": float(eq[-1]) if len(eq) else 0.0,
        "cagr": cagr,
        "ann_vol": sd * math.sqrt(ANN),
        "sharpe": r.mean() / sd * math.sqrt(ANN) if sd and sd > 0 else float("nan"),
        "sortino": r.mean() / downside * math.sqrt(ANN) if downside and downside > 0 else float("nan"),
        "max_dd_usd": float(dd.max()) if len(dd) else 0.0,
        "max_dd_pct": float(max_dd_pct),
        "max_dd_sessions": longest,
        "calmar": cagr / max_dd_pct if max_dd_pct > 0 else float("nan"),
        "time_in_market": float((position.to_numpy() > 0).mean()) if position is not None and len(position) else float("nan"),
    }


def trade_metrics(trades: pd.DataFrame) -> dict:
    net = trades["net_usd"].to_numpy(dtype=float) if len(trades) else np.array([])
    wins, losses = net[net > 0], net[net < 0]
    total = net.sum()
    return {
        "n_trades": int(len(net)),
        "win_rate": float(len(wins) / len(net)) if len(net) else float("nan"),
        "expectancy_usd": float(net.mean()) if len(net) else float("nan"),
        "profit_factor": (wins.sum() / -losses.sum()) if len(losses) else (math.inf if len(wins) else float("nan")),
        "payoff_ratio": (wins.mean() / -losses.mean()) if len(wins) and len(losses) else float("nan"),
        # share is only meaningful for a profitable total; P6 uses net_ex_top5_usd ("positive without the 5 best")
        "top5_share": float(np.sort(net)[::-1][:5].sum() / total) if len(net) and total > 0 else float("nan"),
        "net_ex_top5_usd": float(np.sort(net)[::-1][5:].sum()) if len(net) else 0.0,
        "worst_trade_usd": float(net.min()) if len(net) else float("nan"),
    }
