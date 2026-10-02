# Research Protocol: ES/NQ × CNN Fear & Greed (pre-registration)

Status: **APPROVED v1 by the user on 2026-10-02 (Gate P0 passed). Frozen.** No result on any price data had been computed at approval.
Once approved, this file is frozen. Any change goes into §11, the amendment log, with a date and reason, before the result it affects is computed.

## 1. Question
Does the CNN Fear & Greed index (F&G) predict *forward* returns or risk of ES (S&P 500 E-mini) well enough
to build a **long-only, long-horizon** strategy that is profitable after costs and beats buy-and-hold on a
risk-adjusted basis, and is executable by a trader who places orders **only during the US cash session (RTH)**?
- Primary instrument: ES. NQ is used only to validate rules frozen on ES (no re-fitting).
- Shorts (H8) are out of scope until the user explicitly approves them, and only after long-side results are reported.

## 2. Data
| Series | Source | Span | Note |
|---|---|---|---|
| F&G OLD index | whit3rabbit/fear-greed-data `datasets/archive/fear-greed-pre-2021.csv` (pinned SHA) | 2011-01-03 → 2021-01-29 | Third-party scrape of money.cnn.com; median error 2 points vs archive.org snapshots |
| F&G NEW index | CNN API (via the same mirror, or direct if the host is allowed) | 2021-02-01 → | 2021-02 → 2022-04 is a backfill (not point-in-time) |
| ES1!, NQ1! | User's TradingView exports (DATA.md) | max available | Back-adjusted for P&L; unadjusted for roll checks |

The OLD and NEW eras are **never treated as one homogeneous series**. A median gap of about 6.5 points between them in 2021–22, and different distributions, are documented facts.
Missing F&G (e.g. 2020-06-05 → 2020-07-09) is never forward-filled for more than 3 trading days. Inside a longer gap no new signal can fire.

## 3. Timing (rule R1: no look-ahead)
- The F&G value for date d becomes **available at 00:00 UTC on d+1** (`fgbt.data.align.fg_available_at`).
- Every entry or exit triggered by F&G(d) fills **no earlier than the first NYSE regular open after availability**
  (`first_rth_open_after`, NYSE calendar with holidays). Default fill = RTH open (09:30 ET) of the next trading day.
- Price-based conditions (e.g. a moving-average filter) use only bars whose close is before the decision time.
- Mandatory tests:
  - truncation invariance: signals on data cut at time t equal signals on the full data, for every t;
  - shift test: delaying F&G by one more day must not systematically *improve* results (if it does, that is a leak signature to investigate).

## 4. Splits
| Segment | Span | Use |
|---|---|---|
| Development (DEV) | 2011-01-03 → 2020-12-31 | Hypothesis tests, parameter selection |
| Validation (VAL) | 2021-02-01 → 2024-12-31 | Same-sign replication, walk-forward out-of-sample |
| **Holdout (HOLD)** | 2025-01-01 → 2026-09-30 | **Locked.** One single run of the frozen strategy at Gate P7 |

The holdout is enforced in code: loaders raise an error for dates ≥ 2025-01-01 unless `unlock_holdout=True`, which only `scripts/07_holdout.py` passes.

## 5. Definitions
- **Return:** price change in points on the back-adjusted series between fills, converted to USD per contract.
  For statistics, returns are log returns on ratio-comparable prices.
- **Event (independent episode):** the first day the condition becomes true after ≥ 20 trading days in which it was false.
  Statistical inference counts events, never overlapping days.
- **Horizons h (trading days):** 5, 20, 60, 120, 250.
- **CNN bands:** extreme fear [0, 25), fear [25, 45), neutral [45, 55), greed [55, 75), extreme greed [75, 100].
- **Percentile variant:** F&G converted to its percentile within the trailing 504 trading days of the *same era*,
  available from the 253rd day of each era. This is the robustness version of every threshold rule.
- **Drift benchmark:** the distribution of all h-day returns starting at an RTH open within the same segment.

## 6. Costs and execution (rule R5)
- ES: $5.00 round turn (commission + fees) + 1 tick ($12.50) of slippage per side. **Base cost = $30 per contract per round trip.**
  Stress cases: 2 and 3 ticks per side. NQ: $5.00 + 1 tick ($5.00) per side.
- Roll: positions held across a quarterly roll pay an extra 1 tick + $5.00 per roll.
- Fills: market-on-open at the RTH open. A limit order fills only if price trades *through* it by ≥ 1 tick.
- No stop-losses (user decision). Risk is controlled by position size and an account-level drawdown rule (P8).

## 7. Hypotheses (long side), all two-sided at α = 0.05 after Holm correction within each family
| ID | Statement | Grid |
|---|---|---|
| H1 | Mean forward h-day return differs across CNN bands, and the ordering is monotone decreasing in F&G (Jonckheere–Terpstra test). | 5 bands × 5 h |
| H2 | Event "F&G < T" (entry at next RTH open) has a mean h-day return above the drift benchmark. | T ∈ {10,15,20,25} × h ∈ {20,60,120,250} |
| H3 | Event "F&G crosses back ≥ T after being < T within the last 20 days" has a smaller median MAE than the matching H2 event, with mean return not lower by more than its 90% CI. | same grid as H2 |
| H4 | H2/H3 excess returns are higher when ES > its 200-day SMA at the event than when below. | filter ∈ {200d SMA, 40w SMA} |
| H5 | Exiting when F&G ≥ E (next RTH open) gives a higher Sharpe than holding for a fixed h, for the same entry. | E ∈ {50, 75} vs h ∈ {60,120,250} |
| H6 | F&G at d predicts realized volatility over d+1 → d+20 (negative relation). | 1 test per era |
| H7 | Buying in 3 equal tranches at T, T−5, T−10 lowers the average entry cost vs a single entry at T, with no lower expectancy. | T ∈ {25, 20} |

Inference:
- event-level tests (stationary/block bootstrap with 10,000 resamples, seed 20261002);
- Newey–West HAC for overlapping-day regressions (lag = h);
- a null-preserving bootstrap for predictive regressions (simulate F&G with its estimated AR persistence and
  innovation correlation under no predictability) to neutralize Stambaugh bias.

Every evaluated configuration is appended to `results/trials.csv` (rule R3).

## 8. Gates
| Gate | Pass condition |
|---|---|
| P1 Data | All validation checks pass or each exception is documented and accepted by the user |
| P2 Signal | ≥ 1 hypothesis significant in DEV after Holm correction **and** the same sign in VAL **and** mean excess ≥ 3× base round-trip cost |
| P3 Engine | All unit tests pass. Buy-and-hold and SMA strategy results match an independent vectorized calculation to $0.01 |
| P4 Strategy (DEV) | Profit factor ≥ 1.3; expectancy > 0 at 2× slippage; beats the exposure-matched random-entry null at p < 0.05; ≥ 30 trades, otherwise the verdict is capped at PROVISIONAL |
| P6 Validation | Walk-forward out-of-sample Sharpe > 0 and ≥ 50% of in-sample; PBO ≤ 0.25; Deflated Sharpe ≥ 0.95; positive with frozen rules on NQ; no sign flip under ±20% parameter changes or a 1-day entry delay; positive without the 5 best trades |
| P7 Holdout | One run. Accepted if the holdout Sharpe is above the 10th percentile of the bootstrap distribution of out-of-sample Sharpe |

Verdicts: **ACCEPT** (all gates pass), **PROVISIONAL** (positive and robust but underpowered: paper-trade or minimal size only), **REJECT**.

## 9. Benchmarks (rule R7), reported for every strategy
1. Buy-and-hold ES, rolled, net of roll costs.
2. Exposure-matched random entry: 10,000 simulations with the same number of trades and the same holding-time distribution.
3. The same rule with the F&G condition removed, to measure what F&G adds.

## 10. Metrics
Net P&L, CAGR on the notional base, annualized volatility, Sharpe, Sortino, max drawdown and its duration, Calmar,
time in market, number of trades, win rate, average win and loss, payoff ratio, profit factor, expectancy per trade and per setup,
MAE/MFE, worst trade, and P&L share of the top 5 trades.
**Win rate is reported, never optimized.**

## 11. Amendment log
| Date (UTC) | Section | Change | Reason | Results seen before change? |
|---|---|---|---|---|
| 2026-10-02 | §2, §7, §8 (P6), P8 | **A1: CNN value revisions.** The P1 integrity check (reports/phase1_fear_greed.md §2) shows CNN revises each published value for about 40 sessions (first print vs settled: mean abs ≈1–1.4 points, P90 ≈2.9, max 5.5; up to 10.5 at ages 6–10), and never afterwards. History therefore holds settled values that a live trader never saw. Rules added: (a) primary results use settled history, the only full record that exists; (b) **first-print perturbation test, mandatory at Gate P6:** 1,000 Monte Carlo runs in which every F&G value used for a decision is replaced by settled + a random draw from the empirical first-print error distribution (`data/processed/fg_first_print_errors.csv`, ages 0–1). The rule passes only if the sign of its excess return is unchanged in ≥ 95% of runs and median expectancy is ≥ 50% of the unperturbed value; (c) every threshold rule reports the share of its signals that fall within ±3 points of the threshold; (d) the P8 live routine logs the first-print value daily. | Data-integrity finding (no prices, returns or signals involved) | **No** |
