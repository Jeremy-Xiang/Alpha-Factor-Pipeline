"""
backtest.py — Turns walk-forward predictions into a long-short portfolio
and evaluates it honestly: gross AND net of a simple transaction cost
assumption, decomposed by leg (long vs. short), against an equal-weight
benchmark over the exact same periods.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import WalkForwardPrediction

ROUND_TRIP_COST_BPS = 10  # 0.10% per round trip — a reasonable simple assumption for liquid large caps


@dataclass
class BacktestResult:
    dates: list
    gross_returns: list[float]
    net_returns: list[float]
    long_leg_returns: list[float]
    short_leg_returns: list[float]
    benchmark_returns: list[float]


def run_long_short_backtest(
    predictions: list[WalkForwardPrediction], long_short_frac: float = 0.3
) -> BacktestResult:
    dates, gross, net, long_legs, short_legs, benchmarks = [], [], [], [], [], []

    for pred in predictions:
        df = pd.DataFrame({"ticker": pred.tickers, "predicted": pred.predicted_return, "actual": pred.actual_return})
        df = df.sort_values("predicted", ascending=False).reset_index(drop=True)

        n = len(df)
        n_leg = max(1, int(round(n * long_short_frac)))

        long_leg = df.iloc[:n_leg]
        short_leg = df.iloc[-n_leg:]

        long_return = long_leg["actual"].mean()
        short_return = short_leg["actual"].mean()
        gross_return = long_return - short_return  # long-short: profit when longs beat shorts

        # Full turnover assumed each rebalance (holdings fully reset) — both
        # legs trade, so cost applies to both sides of the book.
        cost = (ROUND_TRIP_COST_BPS / 10_000) * 2
        net_return = gross_return - cost

        dates.append(pred.date)
        gross.append(gross_return)
        net.append(net_return)
        long_legs.append(long_return)
        short_legs.append(short_return)
        benchmarks.append(df["actual"].mean())  # equal-weight, the whole cross-section

    return BacktestResult(
        dates=dates,
        gross_returns=gross,
        net_returns=net,
        long_leg_returns=long_legs,
        short_leg_returns=short_legs,
        benchmark_returns=benchmarks,
    )


def performance_summary(returns: list[float], periods_per_year: float = 12.0) -> dict:
    r = np.array(returns)
    if len(r) == 0:
        return {"annualized_return": np.nan, "annualized_vol": np.nan, "sharpe": np.nan, "max_drawdown": np.nan, "hit_rate": np.nan}

    cumulative = np.cumprod(1 + r)
    n_periods = len(r)
    annualized_return = cumulative[-1] ** (periods_per_year / n_periods) - 1
    annualized_vol = r.std() * np.sqrt(periods_per_year)
    sharpe = (r.mean() * periods_per_year) / annualized_vol if annualized_vol > 1e-9 else 0.0

    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    hit_rate = (r > 0).mean()

    return {
        "annualized_return": annualized_return,
        "annualized_vol": annualized_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "hit_rate": hit_rate,
    }
