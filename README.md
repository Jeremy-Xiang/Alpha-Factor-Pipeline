# alpha-factor-pipeline

A systematic equity factor research pipeline, end to end: a 17-factor
library spanning momentum, technical, value, and quality signals →
information-coefficient analysis to find out which factors actually have
measured predictive power → walk-forward gradient boosting on the
survivors → a long-short portfolio backtest, evaluated honestly (gross and
net of costs, decomposed by leg, against a benchmark) rather than just
reported as a single flattering Sharpe ratio.

This is the closest thing in this whole project series to an actual quant
researcher's day job — not one strategy, but the research *process* that
factor-driven funds run to find out which signals are worth building on.

## The factor library

| Category | Factors | How computed |
|---|---|---|
| Momentum | `mom_1m`, `mom_3m`, `mom_6m`, `mom_12m`, `mom_12m_ex_1m` | Trailing returns over different windows, directly from price history |
| Technical | `rsi_14`, `volatility_21d`, `volume_trend`, `pct_from_52w_high`, `macd_hist` | Standard technical indicators, directly from price/volume history |
| Value | `earnings_yield`, `book_to_market`, `fcf_yield` | 1/PE, 1/PB, and FCF yield from a fundamentals snapshot |
| Quality | `profit_margin`, `roe`, `revenue_growth`, `quality_low_leverage` | From the same fundamentals snapshot (leverage sign-flipped so higher is always "more bullish" by construction, for readability) |

**An honest simplification, stated up front:** value/quality factors are a
single static snapshot per ticker, broadcast across the whole backtest
window — not true point-in-time fundamentals with reporting lags. Momentum
and technical factors don't have this problem at all (they're recomputed
fresh from price history at every date, so they're naturally point-in-time
correct). See "Possible next steps" for what a rigorous fix looks like.

## The headline bug this project actually found

While building the synthetic offline fallback (this sandbox can't reach
Yahoo Finance — see `src/data.py`), both the price-history generator and
the fundamentals generator seeded their random generator with
`stable_seed(ticker)` — the same deterministic seed, independently, in two
different functions. That looked fine: same input, two different
`np.random.default_rng()` calls, surely independent draws.

It measured as a **1.0 correlation** between each ticker's price drift and
its synthetic P/E ratio:

```python
rng1 = np.random.default_rng(stable_seed(ticker))
drift = rng1.uniform(-0.0002, 0.0009)        # first draw

rng2 = np.random.default_rng(stable_seed(ticker))
pe_ratio = rng2.uniform(8, 45)               # also the first draw — same seed
```

`rng.uniform(a, b)` is `a + (b-a) * u` for the same underlying raw uniform
`u` when both generators start from the same seed and this is each
generator's first draw. Drift and P/E weren't independently random at
all — they were the *same number*, linearly rescaled twice. Every
value/quality factor's measured IC was, before the fix, partly measuring
this seeding artifact rather than anything resembling real value investing.

**The fix:** salt the fundamentals seed differently (`stable_seed(f"{ticker}::fundamentals")`
instead of `stable_seed(ticker)`), so the two generators are actually
independent streams. Verified directly:

```
Correlation between drift and pe_ratio, same seed:         1.0000
Correlation after salting the fundamentals seed:           -0.3654  (25 tickers — just small-N noise)
Correlation across 500 synthetic tickers, post-fix:        -0.0733  (converges to ~0, as it should)
```

And the downstream effect on the actual IC analysis — `earnings_yield`'s
information ratio dropped from **-0.489 to -0.043** once the artifact was
removed, which is the difference between "this factor looks like it has
real, exploitable signal" and "this factor has no measured edge on this
universe," which is the correct, honest conclusion for what's otherwise
randomly-assigned synthetic data.

**The general lesson:** seeding two "independent" synthetic data
generators from the same deterministic input is an easy, very-easy-to-miss
way to manufacture fake correlation between things that are supposed to be
unrelated — and the failure mode is silent. The pipeline runs fine either
way and produces plausible-looking numbers regardless. The only way this
surfaced was deliberately testing the assumption ("are these really
independent?") instead of trusting that two different function calls with
different-looking code must produce unrelated randomness.

## Methodology

1. **Panel construction** (`src/factors.py`): for every ticker, compute all
   17 factors plus a 21-trading-day forward return, then stack into one
   long-format panel (`date`, `ticker`, factor columns, target column).
   Rows with any NaN — early rolling-window warmup, or the last 21 days
   where the forward return isn't knowable yet — are dropped outright
   rather than filled.
2. **Rebalance dates** (`src/ic_analysis.py`): every 21st trading date, so
   consecutive rebalance periods never share any of their forward-return
   window. Overlapping windows would make consecutive IC observations
   artificially correlated with each other.
3. **IC analysis**: at each rebalance date, Spearman rank-correlate every
   factor's cross-sectional values against the realized forward return.
   Summarize each factor's IC time series into mean IC, IC volatility, IC
   information ratio (mean/std — *this* is the actual quality measure, not
   mean IC alone), and hit rate.
4. **Factor selection**: keep only factors with `|IC information ratio| >=
   0.15` (configurable via `--ic-threshold`). This is a real filter — see
   results below for which factors it actually keeps and drops on the
   sample universe, not just the keepers.
5. **Walk-forward model** (`src/model.py`): a gradient-boosted regressor
   (`sklearn.ensemble.HistGradientBoostingRegressor`), retrained at every
   rebalance date on an **expanding window of strictly earlier** periods
   only, predicting that date's cross-section. The model never sees a
   period's data before being evaluated on it.
6. **Long-short backtest** (`src/backtest.py`): rank each period's
   cross-section by predicted return, go long the top 30%, short the
   bottom 30%, equal-weight within each leg. Reports gross AND net of a
   simple 10bps-per-round-trip cost assumption, plus each leg separately
   and an equal-weight benchmark over the same periods.

## Results from a sample run (25-ticker universe, 3 years)

```
Selected 6/17 factors (|IC_IR| >= 0.15):
  volume_trend, mom_3m, revenue_growth, mom_12m, mom_12m_ex_1m, mom_6m

Long-short (gross)             ann.return=+0.167  Sharpe=+1.09  maxDD=-0.068
Long-short (net of costs)      ann.return=+0.140  Sharpe=+0.93  maxDD=-0.074
Long leg only                  ann.return=+0.305  Sharpe=+2.44  maxDD=-0.026
Short leg only                 ann.return=+0.109  Sharpe=+1.25  maxDD=-0.083
Equal-weight benchmark         ann.return=+0.252  Sharpe=+4.27  maxDD=-0.009
```

**The benchmark's Sharpe is higher than the long-short strategy's. That's
not a failure — read why before concluding the pipeline doesn't work:**
this synthetic universe has broadly positive per-ticker drift (most
synthetic tickers go up over the period), so an equal-weight "buy
everything" benchmark captures that broad, low-volatility positive drift
almost for free. A genuinely market-neutral long-short book *intentionally
gives up that broad-market exposure* in exchange for a cross-sectional
spread that depends entirely on whether the selected factors have real
predictive power — and on this synthetic universe, with `mom_3m`'s
measured IC actually pointing in the *mean-reverting* direction rather
than continuation, that spread is real but modest.

The number that actually matters for judging a market-neutral strategy
isn't "does its Sharpe beat a long-only benchmark's" — it's whether the
strategy's returns are doing something a benchmark can't. Measured
directly: **correlation between the long-short strategy and the benchmark
is 0.319** — meaningfully decorrelated from the market exposure the
benchmark captures, which is the actual diversification value a
market-neutral sleeve is supposed to provide, Sharpe comparison aside.

This honest framing matters more than the specific numbers: on real market
data, with a properly point-in-time fundamentals pipeline and a larger
universe, the selected factors and resulting Sharpes would come out
completely differently — the point of this README section is to show the
*evaluation discipline*, not to claim a specific edge was found in
synthetic random-walk data.

## Running it

```bash
pip install -r requirements.txt

python run_pipeline.py --basket                                    # built-in 25-ticker universe
python run_pipeline.py --tickers AAPL,MSFT,GOOGL,... --period 3y
python run_pipeline.py --tickers-file my_tickers.txt --ic-threshold 0.1
```

Prints the full IC summary table, selected factors, walk-forward
performance summary (gross/net/by-leg/benchmark), and the strategy-vs-
benchmark correlation discussion, then saves `plots/ic_summary.png`
(factor quality bar chart) and `plots/equity_curves.png` (all five return
streams plotted together).

If `yfinance` can't reach the network, `src/data.py` and
`src/fundamentals.py` fall back to clearly-labeled synthetic data (with
the seed-independence fix described above) — useful for testing the full
pipeline end to end, never for drawing real conclusions about a stock.

## Project structure

```
alpha-factor-pipeline/
├── run_pipeline.py        # entry point
├── plots/
└── src/
    ├── data.py              # OHLCV (yfinance + synthetic fallback)
    ├── fundamentals.py        # fundamentals snapshot (yfinance + synthetic fallback)
    ├── factors.py              # the 17-factor library + panel construction
    ├── ic_analysis.py            # IC computation, summary, factor selection
    ├── model.py                    # walk-forward gradient boosting
    └── backtest.py                   # long-short portfolio construction + evaluation
```

## Running the tests

```bash
pytest tests/ -v
```

The suite pins the behaviors that actually caught bugs during development
(see the sections above), not ceremony coverage — every test encodes a
check where the wrong answer was at some point the actual behavior.

## Possible next steps

- **Point-in-time fundamentals.** The single biggest methodological gap
  here — replace the static snapshot with a real fundamentals time series
  that respects reporting lags (you don't know Q1's numbers until weeks
  into Q2). This would let the value/quality factors' IC numbers be
  trusted the same way the momentum/technical factors' already can be.
- **Real factor library expansion.** 17 factors is a reasonable starting
  library, not an exhaustive one — sector-relative versions of every
  factor (rank within sector, not just across the whole universe) usually
  add real signal cheaply.
- **Swap in real market data and a larger universe.** Everything here runs
  the same way against real `yfinance` history; the synthetic fallback
  exists only because this sandbox can't reach the network. A real run
  would need a meaningfully larger universe (100+ names) before the IC
  estimates are statistically trustworthy — 25 names and 23 rebalance
  periods is enough to demonstrate the methodology, not enough to make a
  real allocation decision on.
- **Sector/industry neutralization** in the long-short construction (long
  top-ranked-within-sector vs. just top-ranked-overall) — reduces the risk
  that the long-short spread is secretly just a sector bet.
