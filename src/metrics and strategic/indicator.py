import scipy.stats as stats
import numpy as np

class RiskMetrics:
    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def MDD(returns):
        """
        Calculate the Maximum Drawdown (MDD) for a given set of returns.

        Parameters:
        returns (array-like): A list or array of returns.

        Returns:
        float: The calculated MDD value.
        """
        # Convert returns to a numpy array
        returns = np.array(returns)

        # Calculate the cumulative returns
        cum_returns = np.cumsum(returns)

        # Calculate the running maximum
        running_max = np.maximum.accumulate(cum_returns)

        # Calculate the drawdown
        drawdown = (cum_returns - running_max) / running_max

        # Return the maximum drawdown
        return np.min(drawdown)
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
    returns (array-like): A list or array of returns.

    Returns:
    float: The calculated MDD value.
    """
    # Convert returns to a numpy array
    returns = np.array(returns)

    # Calculate the cumulative returns
    cum_returns = np.cumsum(returns)

    # Calculate the running maximum
    running_max = np.maximum.accumulate(cum_returns)

    # Calculate the drawdown
    drawdown = (cum_returns - running_max) / running_max

    # Return the maximum drawdown
    return np.min(drawdown)


def RSI(prices, period=14):
    """
    Calculate the Relative Strength Index (RSI) for a given set of prices.

    Parameters:
    prices (array-like): A list or array of prices.
    period (int): The number of periods to use for the RSI calculation (default is 14).

    Returns:
    float: The calculated RSI value.
    """
    # Convert prices to a numpy array
    prices = np.array(prices)

    # Calculate price changes
    delta = np.diff(prices)

    # Separate gains and losses
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)

    # Calculate average gains and losses
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    # Calculate RS and RSI
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    return rsi

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

    # Calculate downside deviation
    downside_returns = excess_returns[excess_returns < 0]
    downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0

    # Calculate Sortino Ratio
    sortino_ratio = mean_excess_return / downside_deviation if downside_deviation != 0 else 0

    return sortino_ratio