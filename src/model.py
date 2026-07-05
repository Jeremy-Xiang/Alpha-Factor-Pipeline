"""
model.py — Walk-forward gradient boosting on the IC-selected factors.

"Walk-forward" here means: at each rebalance date, train ONLY on data from
strictly earlier rebalance dates (an expanding window), then predict that
date's cross-section. The model never sees a given period's data until
it's time to evaluate on it — the standard discipline for any return-
prediction backtest, and the same discipline stock-forecast-bench's
backtest engine uses for single-ticker forecasting, here applied
cross-sectionally instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass
class WalkForwardPrediction:
    date: object
    tickers: list[str]
    predicted_return: list[float]
    actual_return: list[float]


def walk_forward_predict(
    panel: pd.DataFrame,
    rebalance_dates: list,
    selected_factors: list[str],
    min_train_periods: int = 8,
    random_state: int = 42,
) -> list[WalkForwardPrediction]:
    if len(selected_factors) == 0:
        raise ValueError("No factors selected — check the IC threshold in ic_analysis.select_factors().")

    predictions = []

    for i in range(min_train_periods, len(rebalance_dates)):
        train_dates = rebalance_dates[:i]
        test_date = rebalance_dates[i]

        train_data = panel[panel["date"].isin(train_dates)]
        test_data = panel[panel["date"] == test_date]
        if len(test_data) < 8:
            continue

        model = HistGradientBoostingRegressor(max_iter=150, max_depth=4, random_state=random_state)
        model.fit(train_data[selected_factors], train_data["forward_return_21d"])

        predicted = model.predict(test_data[selected_factors])

        predictions.append(
            WalkForwardPrediction(
                date=test_date,
                tickers=test_data["ticker"].tolist(),
                predicted_return=predicted.tolist(),
                actual_return=test_data["forward_return_21d"].tolist(),
            )
        )

    return predictions
