import scipy.stats as stats
import numpy as np


def MARR(returns, confidence_level=0.95):
    """
    Calculate the Minimum Acceptable Rate of Return (MARR) for a given set of returns.

    Parameters:
    returns (array-like): A list or array of returns.
    confidence_level (float): The confidence level for the MARR calculation (default is 0.95).

    Returns:
    float: The calculated MARR value.
    """
    # Convert returns to a numpy array
    returns = np.array(returns)

    # Calculate the mean and standard deviation of the returns
    mean_return = np.mean(returns)
    std_dev = np.std(returns)

    # Calculate the z-score for the given confidence level
    z_score = stats.norm.ppf(confidence_level)

    # Calculate MARR
    marr = mean_return - z_score * std_dev

    return marr

def VaR(returns, confidence_level=0.95):
    """
    Calculate the Value at Risk (VaR) for a given set of returns.

    Parameters:
    returns (array-like): A list or array of returns.
    confidence_level (float): The confidence level for the VaR calculation (default is 0.95).

    Returns:
    float: The calculated VaR value.
    """
    # Convert returns to a numpy array
    returns = np.array(returns)

    # Calculate the percentile for the given confidence level
    var_percentile = np.percentile(returns, (1 - confidence_level) * 100)

    return var_percentile

def CVaR(returns, confidence_level=0.95):
    """
    Calculate the Conditional Value at Risk (CVaR) for a given set of returns.

    Parameters:
    returns (array-like): A list or array of returns.
    confidence_level (float): The confidence level for the CVaR calculation (default is 0.95).

    Returns:
    float: The calculated CVaR value.
    """
    # Convert returns to a numpy array
    returns = np.array(returns)

    # Calculate the Value at Risk (VaR)
    var_value = VaR(returns, confidence_level)

    # Calculate the Conditional Value at Risk (CVaR)
    cvar_value = np.mean(returns[returns <= var_value])

    return cvar_value

def MDD(returns):
    """
    Calculate the Maximum Drawdown (MDD) for a given set of returns.

    Parameters:
    returns (array-like): A list or array of simple (not log) returns per period.

    Returns:
    float: The calculated MDD value, as a negative fraction (-0.25 is a 25% fall from a peak).
    """
    # Convert returns to a numpy array
    returns = np.array(returns)

    # Growth of 1 invested, starting from that 1 so a fall in the first period counts too
    wealth = np.concatenate(([1.0], np.cumprod(1 + returns)))

    # Calculate the running maximum
    running_max = np.maximum.accumulate(wealth)

    # Calculate the drawdown
    drawdown = wealth / running_max - 1

    # Return the maximum drawdown
    return np.min(drawdown)


def RSI(prices, period=14):
    """
    Calculate the Relative Strength Index (RSI) for a given set of prices.

    Parameters:
    prices (array-like): A list or array of prices.
    period (int): The number of periods to use for the RSI calculation (default is 14).

    Returns:
    float: The latest RSI value (0-100), or NaN if there are no more than `period` prices.
    """
    rsi = RSISeries(prices, period)
    return rsi[-1] if len(rsi) else np.nan


def RSISeries(prices, period=14):
    """
    Calculate the Relative Strength Index (RSI) at every price, e.g. to chart it.

    Parameters:
    prices (array-like): A list or array of prices.
    period (int): The number of periods to use for the RSI calculation (default is 14).

    Returns:
    np.ndarray: RSI values (0-100), one per price, NaN for the first `period` prices.
    """
    # Convert prices to a numpy array
    prices = np.array(prices, dtype=float)
    rsi = np.full(len(prices), np.nan)
    if len(prices) <= period:
        return rsi

    # Calculate price changes
    delta = np.diff(prices)

    # Separate gains and losses
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)

    # Wilder's smoothing: start from the simple average of the first `period` changes, then
    # fold in each later change with weight 1/period
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    rsi[period] = _rsi_from_averages(avg_gain, avg_loss)
    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rsi[i + 1] = _rsi_from_averages(avg_gain, avg_loss)

    return rsi


def _rsi_from_averages(avg_gain, avg_loss):
    # Only gains is maximum strength; no movement at all is neutral
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0

    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def SharpeRatio(returns, risk_free_rate=0.0):
    """
    Calculate the Sharpe Ratio for a given set of returns.

    Parameters:
    returns (array-like): A list or array of returns.
    risk_free_rate (float): The risk-free rate of return (default is 0.0).

    Returns:
    float: The calculated Sharpe Ratio value.
    """
    # Convert returns to a numpy array
    returns = np.array(returns)

    # Calculate excess returns
    excess_returns = returns - risk_free_rate

    # Calculate the mean and standard deviation of excess returns
    mean_excess_return = np.mean(excess_returns)
    std_dev_excess_return = np.std(excess_returns)

    # Calculate Sharpe Ratio
    sharpe_ratio = mean_excess_return / std_dev_excess_return if std_dev_excess_return != 0 else 0

    return sharpe_ratio

def SortinoRatio(returns, risk_free_rate=0.0):
    """
    Calculate the Sortino Ratio for a given set of returns.

    Parameters:
    returns (array-like): A list or array of returns.
    risk_free_rate (float): The risk-free rate of return (default is 0.0).

    Returns:
    float: The calculated Sortino Ratio value.
    """
    # Convert returns to a numpy array
    returns = np.array(returns)

    # Calculate excess returns
    excess_returns = returns - risk_free_rate

    # Calculate the mean of excess returns
    mean_excess_return = np.mean(excess_returns)

    # Calculate downside deviation: the root-mean-square shortfall below the risk-free rate,
    # over every period (a period above it is a shortfall of zero)
    downside_returns = np.minimum(excess_returns, 0)
    downside_deviation = np.sqrt(np.mean(downside_returns ** 2))

    # Calculate Sortino Ratio
    sortino_ratio = mean_excess_return / downside_deviation if downside_deviation != 0 else 0

    return sortino_ratio


def CAGR(returns, periods_per_year=252):
    """
    Calculate the Compound Annual Growth Rate (CAGR) for a given set of returns.

    Parameters:
    returns (array-like): A list or array of simple (not log) returns per period.
    periods_per_year (int): The number of periods in a year (default is 252 trading days).

    Returns:
    float: The calculated CAGR value, e.g. 0.12 for 12% a year, or NaN for no returns.
    """
    # Convert returns to a numpy array
    returns = np.array(returns)
    if len(returns) == 0:
        return np.nan

    # Growth of 1 invested over the whole period, spread evenly over its years
    growth = np.prod(1 + returns)
    years = len(returns) / periods_per_year

    return growth ** (1 / years) - 1


def CalmarRatio(returns, periods_per_year=252):
    """
    Calculate the Calmar Ratio: the CAGR divided by the size of the maximum drawdown.

    Parameters:
    returns (array-like): A list or array of simple (not log) returns per period.
    periods_per_year (int): The number of periods in a year (default is 252 trading days).

    Returns:
    float: The calculated Calmar Ratio value, or NaN if there was no drawdown.
    """
    max_drawdown = MDD(returns)
    if max_drawdown == 0:
        return np.nan

    return CAGR(returns, periods_per_year) / abs(max_drawdown)


def Beta(returns, market_returns):
    """
    Calculate the Beta of a set of returns against a market index's returns.

    Parameters:
    returns (array-like): A list or array of returns.
    market_returns (array-like): The market's returns over the same periods.

    Returns:
    float: The calculated Beta value (1 moves with the market, above 1 amplifies its moves,
    below 1 dampens them), or NaN if the market didn't move.
    """
    # Convert returns to numpy arrays
    returns = np.array(returns)
    market_returns = np.array(market_returns)

    # Calculate the variance of the market and its covariance with the returns
    market_variance = np.var(market_returns)
    if market_variance == 0:
        return np.nan
    covariance = np.mean((returns - returns.mean()) * (market_returns - market_returns.mean()))

    return covariance / market_variance


def Alpha(returns, market_returns, risk_free_rate=0.0):
    """
    Calculate Jensen's Alpha: the average return per period beyond what the market's return and
    the Beta explain (CAPM).

    Parameters:
    returns (array-like): A list or array of returns.
    market_returns (array-like): The market's returns over the same periods.
    risk_free_rate (float): The risk-free rate of return per period (default is 0.0).

    Returns:
    float: The calculated Alpha value per period.
    """
    # Convert returns to numpy arrays
    returns = np.array(returns)
    market_returns = np.array(market_returns)

    # Return expected from the market exposure alone, compared with the actual return
    beta = Beta(returns, market_returns)
    expected_excess = beta * np.mean(market_returns - risk_free_rate)

    return np.mean(returns - risk_free_rate) - expected_excess


def EMA(prices, span):
    """
    Calculate the Exponential Moving Average (EMA) at every price.

    Parameters:
    prices (array-like): A list or array of prices.
    span (int): The number of periods; each new price gets weight 2 / (span + 1).

    Returns:
    np.ndarray: EMA values, one per price, starting from the first price.
    """
    # Convert prices to a numpy array
    prices = np.array(prices, dtype=float)
    ema = np.empty_like(prices)
    if len(prices) == 0:
        return ema

    # Each value moves part of the way from the previous value towards the new price
    weight = 2 / (span + 1)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = weight * prices[i] + (1 - weight) * ema[i - 1]

    return ema


def MACD(prices, fast=12, slow=26, signal=9):
    """
    Calculate the Moving Average Convergence Divergence (MACD) at every price.

    Parameters:
    prices (array-like): A list or array of prices.
    fast (int): The span of the fast EMA (default is 12).
    slow (int): The span of the slow EMA (default is 26).
    signal (int): The span of the signal line's EMA of the MACD line (default is 9).

    Returns:
    (np.ndarray, np.ndarray, np.ndarray): The MACD line (fast EMA - slow EMA), the signal line
    and the histogram (MACD - signal), one value per price, NaN until there are enough prices.
    """
    # Convert prices to a numpy array
    prices = np.array(prices, dtype=float)

    # The MACD line only means something once the slow EMA has `slow` prices behind it
    macd = EMA(prices, fast) - EMA(prices, slow)
    macd[: slow - 1] = np.nan

    # The signal line smooths the MACD line, so it needs `signal` MACD values on top of that
    signal_line = np.full_like(macd, np.nan)
    if len(prices) >= slow:
        signal_line[slow - 1:] = EMA(macd[slow - 1:], signal)
        signal_line[: slow + signal - 2] = np.nan

    return macd, signal_line, macd - signal_line


def BollingerBands(prices, period=20, num_std=2.0):
    """
    Calculate Bollinger Bands at every price.

    Parameters:
    prices (array-like): A list or array of prices.
    period (int): The number of periods in the moving average (default is 20).
    num_std (float): How many standard deviations the bands sit from the average (default is 2).

    Returns:
    (np.ndarray, np.ndarray, np.ndarray): The middle band (simple moving average), the upper
    band and the lower band, one value per price, NaN for the first period - 1 prices.
    """
    # Convert prices to a numpy array
    prices = np.array(prices, dtype=float)
    middle = np.full(len(prices), np.nan)
    std_dev = np.full(len(prices), np.nan)

    # Mean and standard deviation of each window of `period` prices
    if len(prices) >= period:
        windows = np.lib.stride_tricks.sliding_window_view(prices, period)
        middle[period - 1:] = windows.mean(axis=1)
        std_dev[period - 1:] = windows.std(axis=1)

    return middle, middle + num_std * std_dev, middle - num_std * std_dev


class RiskMetrics:
    """The same measures grouped as static methods, e.g. RiskMetrics.VaR(returns)."""
    MARR = staticmethod(MARR)
    VaR = staticmethod(VaR)
    CVaR = staticmethod(CVaR)
    MDD = staticmethod(MDD)