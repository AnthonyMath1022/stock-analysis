"""Evaluate one stock's risk and return, or build a portfolio from several stocks.

The measures come from the analysis code in src/metrics and strategic: indicator.py for the risk
and return ratios, MarkowitzPortfolio.py for mean-variance optimisation."""
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# That folder's name has spaces, so it can't be imported as a package; put it on the path instead.
sys.path.insert(0, str(Path(__file__).parent / "src" / "metrics and strategic"))
import indicator  # noqa: E402
from MarkowitzPortfolio import MarkowitzPortfolio  # noqa: E402

TRADING_DAYS = 252
CONFIDENCE = 0.95    # value at risk looks at the worst 5% of days
RSI_PERIOD = 14
MACD_SPANS = (12, 26, 9)   # fast EMA, slow EMA, signal line
BOLLINGER = (20, 2.0)      # moving-average days, standard deviations to each band
MIN_DAYS = 60        # fewest daily returns a portfolio is optimised on
MAX_GAP = 10         # days of missing prices bridged (suspensions, another market's holidays)
PORTFOLIOS = ("Max Sharpe", "Min variance", "Equal weight")


@dataclass
class RiskSummary:
    """Return and risk measured from daily returns, as fractions (0.12 = 12%). Rates and ratios are
    annualised; VaR and CVaR are for a single day. NaN where there's too little data."""
    annual_return: float   # mean daily return × trading days
    cagr: float            # compound annual growth rate
    volatility: float      # annualised standard deviation of daily returns
    sharpe: float          # excess return over the risk-free rate per unit of volatility
    sortino: float         # like Sharpe, but only shortfalls below the risk-free rate count as risk
    calmar: float          # CAGR per unit of max drawdown
    max_drawdown: float    # worst fall from a peak, negative
    var: float             # daily return that only the worst 5% of days fall below, negative
    cvar: float            # average return on those worst 5% of days
    beta: float = np.nan   # sensitivity to the benchmark index's daily moves; stocks only
    alpha: float = np.nan  # annual return beyond what beta and the benchmark explain; stocks only


@dataclass
class Technicals:
    """Chart indicators for each day shown, one value per day. They're computed over a longer
    price history so they're already warmed up on the first day; NaN where it's still too short."""
    moving_averages: dict       # window in days -> simple moving average
    bollinger_upper: np.ndarray
    bollinger_lower: np.ndarray
    rsi: np.ndarray
    macd: np.ndarray
    macd_signal: np.ndarray
    macd_histogram: np.ndarray


def parse_rate(text: str) -> float:
    """
    A percentage as the user typed it, e.g. '4' or '3.5%' -> 0.035.

    Raises:
    ValueError: If it isn't a number.
    """
    try:
        return float(text.strip().rstrip("%")) / 100
    except ValueError:
        raise ValueError(f"'{text}' isn't a rate. Type a percentage such as 4 or 3.5.") from None


def evaluate_returns(daily, risk_free_rate: float) -> RiskSummary:
    """Risk summary of daily simple returns; risk_free_rate is annual, e.g. 0.04."""
    daily = np.asarray(daily, dtype=float)
    daily = daily[np.isfinite(daily)]
    if len(daily) < 2:
        return RiskSummary(**{f.name: np.nan for f in fields(RiskSummary)})
    rf_daily = risk_free_rate / TRADING_DAYS
    annualise = np.sqrt(TRADING_DAYS)
    return RiskSummary(
        annual_return=np.mean(daily) * TRADING_DAYS,
        cagr=indicator.CAGR(daily, TRADING_DAYS),
        volatility=np.std(daily) * annualise,   # population std, as indicator.SharpeRatio uses
        sharpe=indicator.SharpeRatio(daily, rf_daily) * annualise,
        sortino=indicator.SortinoRatio(daily, rf_daily) * annualise,
        calmar=indicator.CalmarRatio(daily, TRADING_DAYS),
        max_drawdown=indicator.MDD(daily),
        var=indicator.VaR(daily, CONFIDENCE),
        cvar=indicator.CVaR(daily, CONFIDENCE),
    )


def evaluate_stock(close: pd.Series, risk_free_rate: float,
                   benchmark: pd.Series | None = None) -> RiskSummary:
    """
    Risk summary of a stock from its daily closes, indexed by date. Given the closes of a
    benchmark index over the same period, it also measures the stock's beta and alpha against it.
    """
    close = close.dropna()
    summary = evaluate_returns(close.pct_change().dropna(), risk_free_rate)
    if benchmark is not None:
        # Only days both traded, so the stock's and the index's returns span the same dates.
        both = pd.concat([close, benchmark], axis=1, join="inner").dropna().pct_change().dropna()
        if len(both) >= 2:
            stock, market = both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy()
            rf_daily = risk_free_rate / TRADING_DAYS
            summary.beta = indicator.Beta(stock, market)
            summary.alpha = indicator.Alpha(stock, market, rf_daily) * TRADING_DAYS
    return summary


def technicals(history: pd.Series, days: int, moving_averages=(20, 50, 200)) -> Technicals:
    """Chart indicators for the last `days` closes of `history`, a longer run of daily closes."""
    prices = history.to_numpy(dtype=float)
    shown = slice(len(prices) - days, None)
    _middle, upper, lower = indicator.BollingerBands(prices, *BOLLINGER)
    macd, signal, histogram = indicator.MACD(prices, *MACD_SPANS)
    return Technicals(
        moving_averages={n: history.rolling(n).mean().to_numpy()[shown] for n in moving_averages},
        bollinger_upper=upper[shown],
        bollinger_lower=lower[shown],
        rsi=indicator.RSISeries(prices, RSI_PERIOD)[shown],
        macd=macd[shown],
        macd_signal=signal[shown],
        macd_histogram=histogram[shown],
    )


@dataclass(frozen=True)
class Holding:
    symbol: str   # Yahoo symbol, e.g. '1155.KL'
    name: str     # company name for display
    market: str   # a key of MARKETS


@dataclass
class PortfolioAnalysis:
    holdings: list[Holding]     # the holdings with prices, in the order given
    missing: list[Holding]      # holdings Yahoo had no prices for in the period, left out
    first_day: pd.Timestamp     # range of days every holding has prices for
    last_day: pd.Timestamp
    days: int                   # daily returns the optimisation used
    risk_free_rate: float
    weights: pd.DataFrame       # symbol index, one column per PORTFOLIOS entry, each summing to 1
    assets: pd.DataFrame        # symbol index, RiskSummary columns for each stock held alone
    portfolios: pd.DataFrame    # PORTFOLIOS index, RiskSummary columns
    frontier: pd.DataFrame      # volatility, return along the efficient frontier
    simulated: pd.DataFrame     # volatility, return, sharpe of random long-only portfolios


def _summaries(returns: dict, risk_free_rate: float) -> pd.DataFrame:
    """One row of RiskSummary columns (without the stock-only beta and alpha) per named daily
    return series."""
    rows = {name: asdict(evaluate_returns(r, risk_free_rate)) for name, r in returns.items()}
    return pd.DataFrame(rows).T.drop(columns=["beta", "alpha"]).astype(float)


def analyse_portfolio(holdings: list[Holding], start_date, end_date,
                      risk_free_rate: float) -> PortfolioAnalysis:
    """
    Find the long-only weights with the best Sharpe ratio and with the lowest volatility from the
    holdings' daily returns over a date range, and measure how each portfolio would have done.

    Raises:
    ValueError: If fewer than two holdings have prices, or they share too few trading days.
    RuntimeError: If the optimiser doesn't converge.
    """
    if len(holdings) < 2:
        raise ValueError("Add at least two stocks to build a portfolio.")
    symbols = [h.symbol for h in holdings]
    prices = yf.download(
        symbols,
        start=start_date,
        end=pd.to_datetime(end_date) + pd.Timedelta(days=1),  # yfinance's end is exclusive
        auto_adjust=True,
        progress=False,
    )
    if prices.empty:
        raise ValueError("No price data found for these stocks in that date range.")
    close = prices["Close"].reindex(columns=symbols).dropna(axis=1, how="all")
    held = [h for h in holdings if h.symbol in close.columns]
    missing = [h for h in holdings if h.symbol not in close.columns]
    if len(held) < 2:
        raise ValueError("Yahoo Finance has prices for fewer than two of these stocks in that "
                         "date range.")

    # Bridge short gaps, then keep only the days every stock has a price for.
    close = close.ffill(limit=MAX_GAP).dropna()
    daily = close.pct_change().dropna()
    if len(daily) < MIN_DAYS:
        raise ValueError(f"These stocks only share {len(daily)} trading days in that date range; "
                         f"at least {MIN_DAYS} are needed. Pick a longer period.")

    mean = daily.mean() * TRADING_DAYS
    # Population covariance, so the optimiser's volatilities match evaluate_returns exactly.
    cov = daily.cov(ddof=0) * TRADING_DAYS
    result = MarkowitzPortfolio().efficient_frontier(mean, cov, risk_free_rate)

    weights = pd.DataFrame({
        "Max Sharpe": result["max_sharpe_weights"],
        "Min variance": result["min_variance_weights"],
        "Equal weight": 1 / len(held),
    })[list(PORTFOLIOS)].clip(lower=0)
    weights /= weights.sum()   # the optimiser can leave weights a hair outside [0, 1]

    return PortfolioAnalysis(
        holdings=held,
        missing=missing,
        first_day=daily.index[0],
        last_day=daily.index[-1],
        days=len(daily),
        risk_free_rate=risk_free_rate,
        weights=weights,
        assets=_summaries({s: daily[s] for s in daily}, risk_free_rate),
        portfolios=_summaries({p: daily @ weights[p] for p in PORTFOLIOS}, risk_free_rate),
        frontier=result["efficient_frontier"],
        simulated=result["simulated_portfolios"],
    )
