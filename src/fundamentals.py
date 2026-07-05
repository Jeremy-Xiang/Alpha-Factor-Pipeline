"""
fundamentals.py — Per-ticker fundamentals snapshot for the value/quality
factors.

IMPORTANT SIMPLIFICATION, stated honestly rather than buried: these are
treated as a single static snapshot per ticker for the entire backtest
window, not a true point-in-time history. Real fundamentals change
quarterly and are reported with a lag (you don't know Q1's numbers until
several weeks into Q2) — a rigorous implementation needs point-in-time
fundamentals with reporting-lag awareness to avoid lookahead bias on the
value/quality factors specifically. This pipeline doesn't have that, and
says so in the IC results too: see README for how the value/quality
factors' IC numbers should be read with that caveat in mind, versus the
momentum/technical factors (computed purely from price/volume history),
which don't have this problem at all since they're naturally point-in-time
correct on every date.
"""

from __future__ import annotations

import numpy as np

from .seed import stable_seed


def get_fundamentals_snapshot(ticker: str) -> dict:
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info
        if not info or ("trailingPE" not in info and "forwardPE" not in info):
            raise RuntimeError("yfinance returned no usable fundamentals")
        return {
            "ticker": ticker,
            "source": "yfinance",
            "pe_ratio": info.get("trailingPE"),
            "pb_ratio": info.get("priceToBook"),
            "profit_margin": info.get("profitMargins"),
            "revenue_growth": info.get("revenueGrowth"),
            "debt_to_equity": info.get("debtToEquity"),
            "roe": info.get("returnOnEquity"),
            "fcf_yield": None,  # not reliably available from yfinance.info; left for a real data source
        }
    except Exception as exc:  # noqa: BLE001
        print(f"[fundamentals.py] Live fetch failed for {ticker} ({exc}). Using synthetic fallback.")
        return _synthetic_fundamentals(ticker)


def _synthetic_fundamentals(ticker: str) -> dict:
    # Salted differently from the price-history seed (stable_seed(ticker) alone) —
    # using the SAME seed here would make the first draw in this function
    # (pe_ratio) a deterministic linear rescaling of the first draw in
    # _synthetic_ohlcv (drift), since both start from the same fresh
    # np.random.default_rng(seed) and rng.uniform(a, b) = a + (b-a)*u for the
    # same raw u. That's not a hypothetical risk: it measured as an exact
    # 1.0 correlation between drift and pe_ratio across tickers before this
    # fix. Any "value factor" IC computed against that data would be a
    # measurement of the generator's seeding scheme, not of anything
    # resembling real value investing.
    rng = np.random.default_rng(stable_seed(f"{ticker}::fundamentals"))
    return {
        "ticker": ticker,
        "source": "synthetic (offline fallback — not real fundamentals)",
        "pe_ratio": round(float(rng.uniform(8, 45)), 1),
        "pb_ratio": round(float(rng.uniform(0.8, 12)), 2),
        "profit_margin": round(float(rng.uniform(-0.05, 0.35)), 3),
        "revenue_growth": round(float(rng.uniform(-0.10, 0.40)), 3),
        "debt_to_equity": round(float(rng.uniform(10, 250)), 1),
        "roe": round(float(rng.uniform(-0.05, 0.35)), 3),
        "fcf_yield": round(float(rng.uniform(-0.02, 0.10)), 3),
    }


def load_fundamentals_universe(tickers: list[str]) -> dict[str, dict]:
    return {t: get_fundamentals_snapshot(t) for t in tickers}
