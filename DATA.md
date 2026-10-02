# Data: sources, provenance, export checklist

## 0. Before ANY price data is committed
Switch the GitHub repo to **private** (GitHub → Settings → General → Danger Zone → Change visibility).
TradingView/CME data generally may not be redistributed. While the repo is public, `.gitignore` blocks
`data/raw/tradingview/`. Delete those two lines only after the switch.

## 1. Fear & Greed (no action needed from you)
- OLD index 2011-01-03 → 2021-01-29 and NEW (CNN API) 2021-02-01 → today come from the public mirror
  `whit3rabbit/fear-greed-data`, pinned to a commit SHA recorded in `data/raw/fear_greed/MANIFEST.json`.
- If the environment allows `production.dataviz.cnn.io`, the NEW series is also pulled directly from CNN and cross-checked against the mirror.
- Known issues (measured during planning, re-verified in P1):
  - a break between eras;
  - the 2021-02 → 2022-04 values are a backfill;
  - a 2020-06-05 → 2020-07-09 hole;
  - `previous_close` differs from the stored history.

## 2. TradingView export checklist (ES and NQ)
**Step 0: pilot.** Export only file #2 below for ES and push it. I validate the format (timestamps, columns,
sessions) before you spend time on the rest.

How to export: open the chart → set symbol, timeframe and settings below → **scroll left until no older bars load**
→ layout menu (arrow next to the layout name, top right) → **Export chart data…** → time format **UNIX timestamp**
→ Export. Menu names can differ slightly between versions. If something is missing, tell me what you see.

Settings for every file:
- Symbol `CME_MINI:ES1!` (and `CME_MINI:NQ1!`)
- No indicators on the chart
- Chart timezone: any (UNIX timestamps are UTC)

| # | Timeframe | Session | Back-adjust (B-ADJ) | File name |
|---|---|---|---|---|
| 1 | 1W | Electronic/extended (full Globex) | ON | `ES1_ETH_1W_badj.csv` |
| 2 | 1D | Electronic/extended | ON | `ES1_ETH_1D_badj.csv` |
| 3 | 1D | Electronic/extended | **OFF** | `ES1_ETH_1D_raw.csv` |
| 4 | 1D | **Regular trading hours** | ON | `ES1_RTH_1D_badj.csv` |
| 5 | 4H | Regular trading hours | ON | `ES1_RTH_4H_badj.csv` |
| 6 | 1H | Regular trading hours | ON | `ES1_RTH_1H_badj.csv` |

Repeat with `NQ1_…` names. Put the files in `data/raw/tradingview/`.

Notes:
- **B-ADJ / back-adjustment** is the continuous-futures adjustment toggle (bottom toolbar or Chart settings → Symbol).
  Files 2 and 3 differ only in this setting. Their difference tells me the exact roll dates and gap sizes.
- **Session** is under Chart settings → Symbol → Session. If "Regular trading hours" is not offered for ES1!, skip
  files 4–6 and export **30m, electronic session, B-ADJ ON** instead. Tell me, because it limits intraday depth.
- Also note whether "Use settlement as close on daily interval" is ticked (Chart settings → Symbol).
- On Essential/Plus (~10k bars): 1D/1W cover all history, 4H RTH covers all history (~500 bars/yr), and 1H RTH covers ~5.7 years.

## 3. Raw-data integrity
Every raw file is listed in `data/raw/*/MANIFEST.json` with its source, export date, settings and SHA-256.
Raw files are never edited. All cleaning happens in code, producing `data/processed/`.
