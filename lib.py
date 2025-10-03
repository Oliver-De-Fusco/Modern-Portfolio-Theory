import numpy as np
import pandas as pd
import typing

PORTFOLIO_LIMIT = 100_000
RISK_FREE = 0

# Generate portfolios
def create_portfolio(mean_returns: np.array, weights: list[float], co_variance_table: np.array) -> list[float, float]:
    """
    Calculates portfolio returns and variance based on weights, covariance is pre-calculated to reduce computation.

    Return, std
    """

    expected_return = np.sum(weights * mean_returns)
    std = np.sqrt(np.dot(weights.T, np.dot(co_variance_table, weights)))

    return expected_return, std