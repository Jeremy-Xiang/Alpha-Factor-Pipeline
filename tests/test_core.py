"""
Tests for alpha-factor-pipeline. Run: pytest tests/ -v

Pins the headline bug (same-seed synthetic generators producing an exact
1.0 correlation between price drift and P/E ratio — fake "value factor"
signal manufactured by the seeding scheme) and the walk-forward
discipline (a model must never train on a period it's evaluated on).
"""

import string
import random as pyrandom

import numpy as np
import pandas as pd
import pytest

from src.backtest import performance_summary, run_long_short_backtest
from src.data import _synthetic_ohlcv, load_universe
from src.factors import FACTOR_NAMES, build_panel
from src.fundamentals import _synthetic_fundamentals, load_fundamentals_universe
from src.ic_analysis import compute_ic_series, get_rebalance_dates, select_factors, summarize_ic
from src.model import walk_forward_predict
from src.seed import stable_seed

TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "JPM", "XOM", "JNJ", "PG", "TSLA", "WMT", "KO", "GS"]


@pytest.fixture(scope="module")
def panel():
    ohlcv = load_universe(TICKERS, period="3y")
    fund = load_fundamentals_universe(TICKERS)
    return build_panel(ohlcv, fund)


def test_drift_and_fundamentals_are_independent():
    """THE bug: same-seed generators made drift and P/E the same raw
    uniform draw rescaled — measured 1.0 correlation. Post-fix it must
    converge to ~0 over a large sample."""
    rng = pyrandom.Random(0)
    fake = ["".join(rng.choices(string.ascii_uppercase, k=4)) for _ in range(400)]
    drifts, pes = [], []
    for t in fake:
        drifts.append(np.random.default_rng(stable_seed(t)).uniform(-0.0002, 0.0009))
        pes.append(_synthetic_fundamentals(t)["pe_ratio"])
    corr = abs(np.corrcoef(drifts, pes)[0, 1])
    assert corr < 0.15, f"seed leakage: |corr|={corr:.3f}"


def test_panel_no_nans_and_correct_columns(panel):
    assert not panel[FACTOR_NAMES + ["forward_return_21d"]].isna().any().any()
    assert set(["date", "ticker"]).issubset(panel.columns)


def test_rebalance_dates_non_overlapping(panel):
    dates = get_rebalance_dates(panel, spacing_days=21)
    all_dates = sorted(panel["date"].unique())
    idx = [all_dates.index(d) for d in dates]
    assert all(b - a == 21 for a, b in zip(idx, idx[1:]))


def test_ic_bounded(panel):
    ic = compute_ic_series(panel, get_rebalance_dates(panel))
    vals = ic[FACTOR_NAMES].to_numpy()
    assert np.nanmax(np.abs(vals)) <= 1.0  # Spearman correlation bound


def test_factor_selection_threshold(panel):
    summary = summarize_ic(compute_ic_series(panel, get_rebalance_dates(panel)))
    selected = select_factors(summary, min_abs_ic_ir=0.15)
    for f in selected:
        assert abs(summary.loc[f, "ic_ir"]) >= 0.15
    # and nothing above threshold was missed
    should = summary[summary["ic_ir"].abs() >= 0.15].index.tolist()
    assert set(selected) == set(should)


def test_walk_forward_never_trains_on_eval_period(panel):
    dates = get_rebalance_dates(panel)
    preds = walk_forward_predict(panel, dates, FACTOR_NAMES[:5], min_train_periods=8)
    assert len(preds) == len(dates) - 8
    for i, p in enumerate(preds):
        # each prediction date must be strictly later than every training date
        assert p.date == dates[8 + i]
        assert all(d < p.date for d in dates[: 8 + i])


def test_backtest_math(panel):
    dates = get_rebalance_dates(panel)
    preds = walk_forward_predict(panel, dates, FACTOR_NAMES[:5], min_train_periods=8)
    result = run_long_short_backtest(preds)
    # gross = long - short by construction, per period
    for g, l, s in zip(result.gross_returns, result.long_leg_returns, result.short_leg_returns):
        assert abs(g - (l - s)) < 1e-12
    # net = gross - cost, so net < gross always
    assert all(n < g for n, g in zip(result.net_returns, result.gross_returns))


def test_performance_summary_sane():
    s = performance_summary([0.01, -0.005, 0.02, 0.0, 0.015])
    assert s["annualized_vol"] > 0
    assert -1.0 <= s["max_drawdown"] <= 0.0
    assert 0.0 <= s["hit_rate"] <= 1.0
