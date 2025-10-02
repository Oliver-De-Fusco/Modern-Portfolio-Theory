import numpy as np
import pandas as pd
import typing

PORTFOLIO_LIMIT = 100_000
RISK_FREE = 0

# Generate portfolios
def create_portfolio(mean_returns :np.array, weight: list[float], co_variance_table: np.array, days=252) -> dict:
    """
    Calculates portfolio returns and variance based on weights, covariance is pre-calculated to reduce computation
    """

    expected_return = np.sum((weight * mean_returns)) * days
    std = np.sqrt(np.dot(weight, np.dot(co_variance_table, weight.T))) * np.sqrt(days)

    return {"return":expected_return, "std": std}