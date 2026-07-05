"""
run_pipeline.py — Runs the full pipeline end to end: load universe -> build
factor panel -> IC analysis & factor selection -> walk-forward model ->
long-short backtest -> performance summary + plots.

Usage:
    python run_pipeline.py --tickers AAPL,MSFT,... --period 3y
    python run_pipeline.py --basket   # built-in 25-ticker sector-diverse universe
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import performance_summary, run_long_short_backtest
from src.data import load_universe
from src.factors import FACTOR_NAMES, build_panel
from src.fundamentals import load_fundamentals_universe
from src.ic_analysis import compute_ic_series, get_rebalance_dates, select_factors, summarize_ic
from src.model import walk_forward_predict

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")

DEFAULT_BASKET = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
    "JPM", "BAC", "GS", "XOM", "CVX",
    "JNJ", "PG", "KO", "PEP", "TSLA", "RIVN", "WMT", "COST",
    "UNH", "HD", "DIS", "CSCO", "INTC", "VZ",
]


def plot_ic_summary(ic_summary: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["tab:green" if v >= 0 else "tab:red" for v in ic_summary["ic_ir"]]
    ax.barh(ic_summary.index, ic_summary["ic_ir"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(0.15, color="gray", linestyle="--", linewidth=1, label="selection threshold")
    ax.axvline(-0.15, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("IC Information Ratio (mean IC / std IC)")
    ax.set_title("Factor quality — IC Information Ratio by factor")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_equity_curves(result, path: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    dates = result.dates

    for label, returns, style in [
        ("Long-short (net of costs)", result.net_returns, "tab:blue"),
        ("Long-short (gross)", result.gross_returns, "tab:cyan"),
        ("Long leg only", result.long_leg_returns, "tab:green"),
        ("Short leg only", result.short_leg_returns, "tab:red"),
        ("Equal-weight benchmark", result.benchmark_returns, "black"),
    ]:
        cumulative = np.cumprod(1 + np.array(returns))
        ax.plot(dates, cumulative, label=label, color=style, linewidth=1.6 if "net" in label.lower() or "benchmark" in label.lower() else 1.0)

    ax.set_ylabel("Cumulative return (x initial)")
    ax.set_title("Walk-forward out-of-sample performance")
    ax.legend(loc="upper left", fontsize=9)
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", default=None)
    parser.add_argument("--tickers-file", default=None)
    parser.add_argument("--basket", action="store_true")
    parser.add_argument("--period", default="3y")
    parser.add_argument("--min-train-periods", type=int, default=8)
    parser.add_argument("--ic-threshold", type=float, default=0.15)
    args = parser.parse_args()

    os.makedirs(PLOTS_DIR, exist_ok=True)

    if args.tickers_file:
        with open(args.tickers_file) as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = DEFAULT_BASKET

    print(f"Loading {len(tickers)} tickers...")
    ohlcv = load_universe(tickers, period=args.period)
    fundamentals = load_fundamentals_universe(list(ohlcv.keys()))
    print(f"Got usable history for {len(ohlcv)}/{len(tickers)} tickers.\n")

    panel = build_panel(ohlcv, fundamentals)
    rebalance_dates = get_rebalance_dates(panel)
    print(f"{len(rebalance_dates)} rebalance dates (every 21 trading days).\n")

    ic_series = compute_ic_series(panel, rebalance_dates)
    ic_summary = summarize_ic(ic_series)
    print("=== Factor IC summary (sorted by |IC information ratio|) ===")
    print(ic_summary.round(3).to_string())

    selected = select_factors(ic_summary, min_abs_ic_ir=args.ic_threshold)
    print(f"\nSelected {len(selected)}/{len(FACTOR_NAMES)} factors (|IC_IR| >= {args.ic_threshold}): {selected}")

    if len(selected) == 0:
        print("\nNo factors cleared the threshold — lowering it or using a larger universe would be the "
              "next step, not forcing a model on factors with no measured edge.")
        return

    print(f"\nRunning walk-forward model (expanding window, min {args.min_train_periods} training periods)...")
    predictions = walk_forward_predict(panel, rebalance_dates, selected, min_train_periods=args.min_train_periods)
    print(f"{len(predictions)} out-of-sample periods evaluated.\n")

    result = run_long_short_backtest(predictions)

    print("=== Performance summary ===")
    for label, returns in [
        ("Long-short (gross)", result.gross_returns),
        ("Long-short (net of costs)", result.net_returns),
        ("Long leg only", result.long_leg_returns),
        ("Short leg only", result.short_leg_returns),
        ("Equal-weight benchmark", result.benchmark_returns),
    ]:
        s = performance_summary(returns)
        print(f"{label:30} ann.return={s['annualized_return']:+.3f}  ann.vol={s['annualized_vol']:.3f}  "
              f"Sharpe={s['sharpe']:+.2f}  maxDD={s['max_drawdown']:.3f}  hit_rate={s['hit_rate']:.2f}")

    corr = np.corrcoef(result.gross_returns, result.benchmark_returns)[0, 1]
    print(f"\nCorrelation of long-short strategy to benchmark: {corr:.3f}")
    print("(Market-neutral strategies should be judged on this kind of diversification value, not just on "
          "whether their standalone Sharpe beats a long-only benchmark's — a benchmark dominated by broad "
          "positive drift can easily out-Sharpe a genuinely market-neutral book with weak alpha, and that's "
          "not the long-short strategy 'failing.' See README for the full discussion.)")

    p1 = os.path.join(PLOTS_DIR, "ic_summary.png")
    p2 = os.path.join(PLOTS_DIR, "equity_curves.png")
    plot_ic_summary(ic_summary, p1)
    plot_equity_curves(result, p2)
    print(f"\nSaved: {p1}\nSaved: {p2}")


if __name__ == "__main__":
    main()
