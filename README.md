# alpha-factor-pipeline

A factor research pipeline that mirrors how systematic equity funds actually evaluate signals before trading on them: build a factor library, measure each factor's predictive power empirically, train a model only on factors that clear the bar, and report backtest results honestly — gross, net of costs, per leg, against a benchmark that often wins.

## What it does

Seventeen factors across four categories:

**Momentum/technical** — `mom_1m`, `mom_3m`, `mom_6m`, `mom_12m`, `mom_12m_ex_1m` (the classic "skip the last month" variant), `rsi_14`, `volatility_21d`, `volume_trend`, `pct_from_52w_high`, `macd_hist`. Computed fresh at every date from price and volume history, so they're naturally point-in-time correct.

**Value/quality** — `earnings_yield`, `book_to_market`, `fcf_yield`, `profit_margin`, `roe`, `revenue_growth`, `quality_low_leverage`. Pulled from a fundamentals snapshot. These use a static snapshot per ticker broadcast across the full backtest window, which introduces lookahead bias — the README for the value factors' IC numbers needs to be read with that caveat in mind. The momentum/technical numbers don't have this problem.

For each rebalance date (every 21 trading days), the pipeline computes a cross-sectional Spearman rank correlation between each factor's values and the realized 21-day forward return. That correlation is the information coefficient (IC). A factor's IC information ratio (mean IC divided by IC standard deviation) is the quality measure that determines whether it gets used. Factors with `|IC_IR| < 0.15` are dropped before any model training happens.

The survivors feed a `HistGradientBoostingRegressor` trained on an expanding window of strictly prior periods. At each rebalance date, the model predicts that date's cross-section using only data from before it — no lookahead. Long top 30% by predicted return, short bottom 30%, equal-weight within each leg.

## Results

```
Long-short (gross)         ann.return=+0.167  Sharpe=+1.09  maxDD=-0.068
Long-short (net of costs)  ann.return=+0.140  Sharpe=+0.93  maxDD=-0.074
Long leg only              ann.return=+0.305  Sharpe=+2.44  maxDD=-0.026
Equal-weight benchmark     ann.return=+0.252  Sharpe=+4.27  maxDD=-0.009
Correlation to benchmark:  0.319
```

The benchmark's Sharpe is higher than the strategy's. That's the correct outcome to report, and worth understanding rather than hiding. The equal-weight benchmark captures broad positive drift across the universe almost for free — low volatility, high Sharpe. A market-neutral long-short book intentionally gives that up. The right comparison isn't Sharpe vs. Sharpe; it's whether the strategy's returns are decorrelated from the benchmark's exposure. A 0.319 correlation means the long-short is doing something the benchmark isn't, which is the actual point of running a market-neutral strategy.

## The seeding bug

Both the synthetic price-history generator and the synthetic fundamentals generator originally seeded from `stable_seed(ticker)` — the same seed, each in its own fresh `np.random.default_rng()` call. Two separate generators, same seed, first draw each: `rng.uniform(a, b) = a + (b-a) * u` for the same underlying `u`. Drift and P/E ratio weren't independent — they were the same random draw rescaled twice.

Measured as a **1.0 correlation** between price drift and P/E ratio across tickers. `earnings_yield`'s IC information ratio: -0.489 before the fix, -0.043 after. The difference between "this factor has exploitable signal" and "this factor is noise" was entirely the seeding scheme.

Fix: salt the fundamentals seed differently — `stable_seed(f"{ticker}::fundamentals")` — so the two generators pull from independent streams. Confirmed across 500 synthetic tickers, post-fix correlation is ~0.07 (small-sample noise around zero).

The failure mode is silent. The pipeline runs fine either way and produces plausible-looking IC numbers. The only way this surfaced was explicitly testing the assumption that the two generators were independent.

## Running it

```bash
pip install -r requirements.txt

python run_pipeline.py --basket                           # built-in 25-ticker universe
python run_pipeline.py --tickers AAPL,MSFT,GOOGL,JPM,XOM
python run_pipeline.py --tickers-file tickers.txt --ic-threshold 0.10
```

Prints the full IC summary table, selected factors, per-leg performance, and the benchmark correlation discussion. Saves `plots/ic_summary.png` (factor quality bar chart) and `plots/equity_curves.png`.

yfinance pulls live data. Synthetic fallback kicks in if the network's unavailable, labeled as such.

## Running the tests

```bash
pytest tests/ -v
```

Eight tests. The seeding bug has its own test: takes 400 synthetic tickers, measures the correlation between drift and P/E, asserts it's below 0.15. If the seed salting ever gets reverted, that test fails immediately.

## Structure

```
alpha-factor-pipeline/
├── run_pipeline.py     # CLI entry point
├── src/
│   ├── factors.py      # 17-factor library + panel construction
│   ├── ic_analysis.py  # IC computation, summary, factor selection
│   ├── model.py        # walk-forward gradient boosting
│   ├── backtest.py     # long-short construction + evaluation
│   ├── data.py         # OHLCV loader
│   ├── fundamentals.py # fundamentals snapshot
│   └── seed.py         # collision-resistant seeding
└── tests/test_core.py
```
