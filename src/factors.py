"""
factors.py — The factor library. Two categories, computed very differently:

- Momentum/technical factors are computed fresh at every date from price
  and volume history alone. They're naturally point-in-time correct —
  there's no lookahead risk, since today's RSI only ever uses data up
  through today.
- Value/quality factors come from a single static fundamentals snapshot
  per ticker (see fundamentals.py) and are broadcast across every date in
  the backtest window. This is a real simplification, not a hidden one —
  see fundamentals.py's docstring and the README for what it means for how
  to read this pipeline's value/quality IC numbers specifically.

All factors are oriented so that, by construction, a HIGHER value is the
"more bullish on this factor's own logic" direction (e.g. quality_low_leverage
is the negative of debt-to-equity) — purely for interpretability. The
empirical IC analysis later is what actually tells you whether that
intuition holds up; orientation doesn't presuppose the answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FORWARD_RETURN_DAYS = 21  # ~1 month ahead, matching the monthly rebalance cadence

FACTOR_NAMES = [
    # momentum / technical — computed from price & volume history
    "mom_1m", "mom_3m", "mom_6m", "mom_12m", "mom_12m_ex_1m",
    "rsi_14", "volatility_21d", "volume_trend", "pct_from_52w_high", "macd_hist",
    # value / quality — static snapshot, broadcast across dates (see caveat above)
    "earnings_yield", "book_to_market", "fcf_yield",
    "profit_margin", "roe", "revenue_growth", "quality_low_leverage",
]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def compute_technical_factors(df: pd.DataFrame) -> pd.DataFrame:
    close, volume = df["Close"], df["Volume"]
    out = pd.DataFrame(index=df.index)

    out["mom_1m"] = close.pct_change(21)
    out["mom_3m"] = close.pct_change(63)
    out["mom_6m"] = close.pct_change(126)
    out["mom_12m"] = close.pct_change(252)
    # Classic "skip the most recent month" momentum: return from t-252 to t-21,
    # excluding short-term reversal effects in the most recent month.
    out["mom_12m_ex_1m"] = close.shift(21).pct_change(231)

    out["rsi_14"] = _rsi(close, 14)

    daily_ret = close.pct_change()
    out["volatility_21d"] = daily_ret.rolling(21).std() * np.sqrt(252)

    out["volume_trend"] = volume.rolling(20).mean() / volume.rolling(60).mean()

    high_52w = close.rolling(252, min_periods=1).max()
    out["pct_from_52w_high"] = close / high_52w - 1

    ema12, ema26 = close.ewm(span=12).mean(), close.ewm(span=26).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9).mean()
    out["macd_hist"] = macd_line - macd_signal

    out["forward_return_21d"] = close.shift(-FORWARD_RETURN_DAYS) / close - 1
    return out


def attach_fundamental_factors(tech_df: pd.DataFrame, fundamentals: dict) -> pd.DataFrame:
    df = tech_df.copy()
    pe, pb = fundamentals.get("pe_ratio"), fundamentals.get("pb_ratio")
    df["earnings_yield"] = (1.0 / pe) if pe else np.nan
    df["book_to_market"] = (1.0 / pb) if pb else np.nan
    df["fcf_yield"] = fundamentals.get("fcf_yield")
    df["profit_margin"] = fundamentals.get("profit_margin")
    df["roe"] = fundamentals.get("roe")
    df["revenue_growth"] = fundamentals.get("revenue_growth")
    debt_to_equity = fundamentals.get("debt_to_equity")
    df["quality_low_leverage"] = -debt_to_equity if debt_to_equity is not None else np.nan
    return df


def build_panel(tickers_data: dict[str, pd.DataFrame], fundamentals_data: dict[str, dict]) -> pd.DataFrame:
    """
    Returns a long-format panel: one row per (date, ticker), columns =
    FACTOR_NAMES + ['forward_return_21d']. Rows with any NaN (early rolling-
    window warmup, or the last 21 days where forward return isn't knowable
    yet) are dropped — silently filling them would corrupt cross-sectional
    IC computation downstream.
    """
    frames = []
    for ticker, ohlcv in tickers_data.items():
        tech = compute_technical_factors(ohlcv)
        full = attach_fundamental_factors(tech, fundamentals_data[ticker])
        full["ticker"] = ticker
        full["date"] = full.index
        frames.append(full)

    panel = pd.concat(frames, ignore_index=True)
    before = len(panel)
    panel = panel.dropna(subset=FACTOR_NAMES + ["forward_return_21d"])
    after = len(panel)
    print(f"[factors.py] Panel: {after}/{before} rows kept after dropping warmup/lookahead-incomplete rows.")

    return panel[["date", "ticker"] + FACTOR_NAMES + ["forward_return_21d"]].reset_index(drop=True)
