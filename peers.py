"""Compare a stock against its peer group: the largest companies in the same industry, listed
on the same exchange."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
from yfinance import EquityQuery

from yfinnace import MARKETS, MINOR_UNITS, build_symbol

MAX_PEERS = 8
MIN_PEERS = 3        # fewer industry peers than this and the group widens to the whole sector
SCREEN_SIZE = 50     # candidates fetched before dropping cross-listings and extra share classes
TRADING_DAYS = 252

# Yahoo's screener spells sub-industries with an em dash, Ticker.info with " - ".
SCREENER_DASH = "—"


@dataclass
class PeerComparison:
    symbol: str                  # the stock being compared, e.g. '1155.KL'
    group: str                   # industry or sector name, e.g. 'Banks—Regional'
    basis: str                   # 'industry' or 'sector'
    currency: str                # ISO code all prices and market caps are in
    performance: pd.DataFrame    # Date index, one column per symbol, rebased to 100
    peer_median: pd.Series       # median of the peers' (not the target's) rebased lines
    table: pd.DataFrame          # one row per symbol, columns named in METRICS; target first

    @property
    def peers(self) -> list[str]:
        return [s for s in self.table.index if s != self.symbol]

    def median(self, column: str) -> float:
        return self.table.loc[self.peers, column].median()

    def rank(self, column: str, ascending: bool = False) -> tuple[int, int] | None:
        """Target's place among everyone with a value, 1 = highest (or lowest if ascending)."""
        values = self.table[column].dropna()
        if self.symbol not in values.index:
            return None
        return int(values.rank(ascending=ascending, method="min")[self.symbol]), len(values)


# column -> (heading, higher is better?) ; None means neither direction is "better"
METRICS = {
    "name": ("Company", None),
    "price": ("Last price", None),
    "return": ("Return", True),
    "volatility": ("Volatility", False),
    "max_drawdown": ("Max drawdown", True),   # stored negative, so higher = shallower
    "market_cap": ("Market cap", None),
    "pe": ("P/E", None),
    "pb": ("P/B", None),
    "dividend_yield": ("Div. yield", None),
}


def _group_query(field: str, value: str):
    try:
        return EquityQuery("eq", [field, value])
    except ValueError:   # the screener doesn't recognise this sector/industry name
        return None


def _screen(exchanges: tuple, field: str, value: str) -> list[dict]:
    by_group = _group_query(field, value)
    if by_group is None:
        return []
    query = EquityQuery("and", [EquityQuery("is-in", ["exchange", *exchanges]), by_group])
    return yf.screen(query, sortField="intradaymarketcap", sortAsc=False,
                     size=SCREEN_SIZE).get("quotes", [])


def _pick_peers(quotes: list[dict], target: dict, limit: int) -> list[dict]:
    """Largest companies first, skipping the target, lines quoted in another currency (London's
    foreign cross-listings) and second share classes (preference shares, B shares)."""
    seen = {target.get("longName") or target.get("shortName")}
    peers = []
    for q in quotes:
        name = q.get("longName") or q.get("shortName")
        if (q["symbol"] == target["symbol"] or q.get("currency") != target.get("currency")
                or not q.get("marketCap") or name in seen):
            continue
        seen.add(name)
        peers.append(q)
        if len(peers) == limit:
            break
    return peers


def find_peers(symbol: str, market: str, limit: int = MAX_PEERS) -> tuple[dict, list[dict], str, str]:
    """
    Look up the stock's industry and find the largest other companies in it on the same market.

    Returns:
    (target info, peer quotes, group name, 'industry' or 'sector')

    Raises:
    ValueError: If Yahoo has no sector for the stock or no peers can be found.
    """
    info = yf.Ticker(symbol).info
    info.setdefault("symbol", symbol)
    sector, industry = info.get("sector"), info.get("industry")
    if not sector:
        raise ValueError(f"Yahoo Finance has no sector for '{symbol}', so it has no peer group.")

    exchanges = MARKETS[market].exchanges or (info.get("exchange"),)
    if industry:
        industry = industry.replace(" - ", SCREENER_DASH)
        peers = _pick_peers(_screen(exchanges, "industry", industry), info, limit)
        if len(peers) >= MIN_PEERS:
            return info, peers, industry, "industry"
    peers = _pick_peers(_screen(exchanges, "sector", sector), info, limit)
    if not peers:
        raise ValueError(f"No other {sector} companies found on {market}.")
    return info, peers, sector, "sector"


def performance_metrics(close: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rebase each column to 100 at its first price and measure the path it took.

    Returns:
    (rebased prices, DataFrame indexed by column with return, volatility, max_drawdown as fractions)
    """
    close = close.ffill()                        # bridge a peer's missing days
    rebased = close / close.bfill().iloc[0] * 100
    daily = np.log(close).diff()
    metrics = pd.DataFrame({
        "return": rebased.iloc[-1] / 100 - 1,
        "volatility": daily.std() * np.sqrt(TRADING_DAYS),
        "max_drawdown": (rebased / rebased.cummax() - 1).min(),
    })
    return rebased, metrics


def get_peer_comparison(ticker: str, market: str, start_date, end_date,
                        limit: int = MAX_PEERS) -> PeerComparison:
    """
    Compare a stock with its peers over a date range: price performance from daily history,
    valuation from Yahoo's latest quote.

    Raises:
    ValueError: If the ticker is malformed, has no peers, or Yahoo returns no prices for it.
    """
    symbol = build_symbol(ticker, market)
    info, peers, group, basis = find_peers(symbol, market, limit)
    quotes = [info, *peers]
    symbols = [q["symbol"] for q in quotes]

    prices = yf.download(
        symbols,
        start=start_date,
        end=pd.to_datetime(end_date) + pd.Timedelta(days=1),  # yfinance's end is exclusive
        auto_adjust=True,
        progress=False,
    )
    close = prices["Close"].reindex(columns=symbols).dropna(axis=1, how="all")
    if symbol not in close.columns:
        raise ValueError(f"No price data for '{symbol}' in that date range.")

    currency = info.get("currency") or MARKETS[market].currency
    currency, divisor = MINOR_UNITS.get(currency, (currency, 1))

    rebased, metrics = performance_metrics(close)
    fundamentals = pd.DataFrame([{
        "symbol": q["symbol"],
        "name": q.get("longName") or q.get("shortName") or q["symbol"],
        "market_cap": q.get("marketCap"),      # already in major units, even in London
        "pe": q.get("trailingPE"),
        "pb": q.get("priceToBook"),
        "dividend_yield": q.get("dividendYield"),  # Yahoo gives this in percent
    } for q in quotes if q["symbol"] in close.columns]).set_index("symbol")
    fundamentals["price"] = close.ffill().iloc[-1] / divisor
    table = fundamentals.join(metrics)[list(METRICS)]
    table[table.columns[1:]] = table[table.columns[1:]].astype(float)

    peer_cols = [s for s in close.columns if s != symbol]
    return PeerComparison(
        symbol=symbol, group=group, basis=basis, currency=currency,
        performance=rebased,
        peer_median=rebased[peer_cols].median(axis=1),
        table=table,
    )
