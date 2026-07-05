"""
ic_analysis.py — Information coefficient analysis: for each rebalance
date, rank-correlate each factor's cross-sectional values against the
realized forward return, then summarize each factor's IC time series into
a single quality measure (mean IC, IC volatility, IC information ratio,
hit rate).

This step exists specifically so the model-building step doesn't just
throw all 17 factors at a gradient booster and hope. A factor with
consistently near-zero or sign-flipping IC is noise; including it just
gives the model more ways to overfit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .factors import FACTOR_NAMES


def get_rebalance_dates(panel: pd.DataFrame, spacing_days: int = 21) -> list:
    """
    Every `spacing_days`-th unique trading date, so consecutive rebalance
    periods don't share any of their forward-return window — overlapping
    windows would make consecutive IC observations artificially correlated.
    """
    all_dates = sorted(panel["date"].unique())
    return all_dates[::spacing_days]


def compute_ic_series(panel: pd.DataFrame, rebalance_dates: list) -> pd.DataFrame:
    """Returns a DataFrame indexed by rebalance date, one column per factor, values = that date's cross-sectional Spearman IC."""
    rows = []
    for date in rebalance_dates:
        cross_section = panel[panel["date"] == date]
        if len(cross_section) < 8:  # too few names for a meaningful cross-sectional correlation
            continue

        ic_row = {"date": date, "n_tickers": len(cross_section)}
        forward_returns = cross_section["forward_return_21d"]
        for factor in FACTOR_NAMES:
            ic, _ = spearmanr(cross_section[factor], forward_returns)
            ic_row[factor] = ic
        rows.append(ic_row)

    return pd.DataFrame(rows).set_index("date")


def summarize_ic(ic_series: pd.DataFrame) -> pd.DataFrame:
    summary = []
    for factor in FACTOR_NAMES:
        series = ic_series[factor].dropna()
        mean_ic, std_ic = series.mean(), series.std()
        ic_ir = mean_ic / std_ic if std_ic > 1e-9 else 0.0
        hit_rate = (np.sign(series) == np.sign(mean_ic)).mean() if len(series) else np.nan
        summary.append(
            {
                "factor": factor,
                "mean_ic": mean_ic,
                "std_ic": std_ic,
                "ic_ir": ic_ir,
                "hit_rate": hit_rate,
                "n_periods": len(series),
            }
        )
    return pd.DataFrame(summary).set_index("factor").sort_values("ic_ir", key=abs, ascending=False)


def select_factors(ic_summary: pd.DataFrame, min_abs_ic_ir: float = 0.15) -> list[str]:
    """
    Keep factors whose IC information ratio clears a threshold — i.e. the
    factor's average predictive direction is large relative to how much it
    bounces around period to period, not just "happened to average above
    zero." This is a real filter: see README for the actual factors this
    drops on the sample universe, not just the ones it keeps.
    """
    selected = ic_summary[ic_summary["ic_ir"].abs() >= min_abs_ic_ir].index.tolist()
    return selected
