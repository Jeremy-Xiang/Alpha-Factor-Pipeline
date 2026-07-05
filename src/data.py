"""
data.py — Multi-ticker OHLCV loader, same yfinance-first/synthetic-fallback
pattern as every sibling project. The one thing that matters specifically
for this project: every synthetic ticker is built on the SAME date range
(pd.bdate_range ending today), so cross-sectional alignment across tickers
is automatic — no reindexing/forward-filling needed to compare factor
values across stocks on the same date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .seed import stable_seed


def load_ohlcv(ticker: str, period: str = "3y") -> pd.DataFrame:
    try:
        import yfinance as yf

        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df is None or df.empty:
            raise RuntimeError("yfinance returned no data")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception as exc:  # noqa: BLE001
        print(f"[data.py] Live fetch failed for {ticker} ({exc}). Using synthetic fallback.")
        return _synthetic_ohlcv(seed=stable_seed(ticker))


def _synthetic_ohlcv(n_days: int = 756, seed: int = 0, start_price: float = 100.0) -> pd.DataFrame:
    """~3 years of daily data (756 ~= 3 * 252 trading days)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)

    # A mix of drift regimes so some synthetic tickers are genuinely
    # better/worse performers than others -- otherwise every factor would
    # be trying to predict noise with no real cross-sectional signal at all.
    drift = rng.uniform(-0.0002, 0.0009)
    vol = rng.uniform(0.012, 0.028)
    daily_returns = rng.normal(loc=drift, scale=vol, size=n_days)
    close = start_price * np.exp(np.cumsum(daily_returns))

    open_ = np.empty(n_days)
    open_[0] = start_price
    open_[1:] = close[:-1] * (1 + rng.normal(0, 0.003, size=n_days - 1))
    intraday_range = np.abs(rng.normal(0.006, 0.004, size=n_days)) * close
    high = np.maximum(open_, close) + intraday_range
    low = np.minimum(open_, close) - intraday_range
    volume = rng.integers(1_000_000, 30_000_000, size=n_days).astype(float)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates
    )


def load_universe(tickers: list[str], period: str = "3y") -> dict[str, pd.DataFrame]:
    out = {}
    for t in tickers:
        df = load_ohlcv(t, period=period)
        if len(df) >= 300:  # need enough history for 12m momentum + walk-forward
            out[t] = df
        else:
            print(f"[data.py] Skipping {t}: not enough history ({len(df)} rows)")
    return out
