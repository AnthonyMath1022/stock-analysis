import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA = Path(__file__).parent / "data"   # stock lists built by update_stock_lists.py


@dataclass(frozen=True)
class Market:
    suffix: str              # Yahoo Finance exchange suffix, "" for US
    currency: str            # fallback if Yahoo doesn't report one
    pattern: str             # regex a normalized ticker must match
    example: str             # shown to the user when their input is wrong
    pad: int = 0             # zero-pad numeric codes to this width (e.g. HK 700 -> 0700)
    dash_classes: bool = False  # Yahoo writes share classes with "-" (BRK.B -> BRK-B)
    popular: dict = field(default_factory=dict)  # company name -> ticker, for dropdowns
    exchanges: tuple = ()    # Yahoo screener exchange codes, for finding peers on this market
    aliases: dict = field(default_factory=dict)  # other names users type -> ticker
    benchmark: str = ""       # Yahoo symbol of the market's main index, for beta and alpha
    benchmark_name: str = ""  # short name of that index, e.g. 'KLCI'


def _read_stock_list(filename: str) -> pd.DataFrame | None:
    """A stock list from DATA, largest first, or None if it hasn't been built."""
    if not (DATA / filename).exists():
        return None
    # dtype=str keeps leading zeros (0318); keep_default_na stops short names like NA becoming NaN.
    return pd.read_csv(DATA / filename, dtype=str, keep_default_na=False)


def _bursa_stocks() -> tuple[dict, dict]:
    """
    Every Bursa stock as (dropdown label -> code, short name -> code), e.g.
    ({'MAYBANK · Malayan Banking Bhd.': '1155', ...}, {'MAYBANK': '1155', ...}).
    Falls back to a few large caps if the list hasn't been built.
    """
    df = _read_stock_list("bursa_stocks.csv")
    if df is None:
        return {"Maybank": "1155", "Public Bank": "1295", "CIMB": "1023",
                "Tenaga Nasional": "5347", "IHH Healthcare": "5225",
                "Petronas Chemicals": "5183"}, {}
    return (dict(zip(df["short_name"] + " · " + df["name"], df["code"])),
            dict(zip(df["short_name"], df["code"])))


def _sgx_stocks() -> dict:
    """Every SGX stock as company name -> code, e.g. {'DBS Group Holdings Ltd': 'D05', ...}.
    Falls back to a few large caps if the list hasn't been built."""
    df = _read_stock_list("sgx_stocks.csv")
    if df is None:
        return {"DBS": "D05", "OCBC": "O39", "UOB": "U11", "Singtel": "Z74",
                "CapitaLand Integrated Commercial Trust": "C38U"}
    return dict(zip(df["name"], df["code"]))


BURSA_POPULAR, BURSA_SHORT_NAMES = _bursa_stocks()


# Keyed by exchange, not country: China has two exchanges with different suffixes.
MARKETS = {
    "United States (NYSE/Nasdaq)": Market(
        suffix="", exchanges=("NMS", "NGM", "NCM", "NYQ", "ASE"),
        benchmark="^GSPC", benchmark_name="S&P 500",
        currency="USD", pattern=r"[A-Z]{1,5}(-[A-Z])?", example="AAPL, BRK.B",
        dash_classes=True,
        popular={"Apple": "AAPL", "Microsoft": "MSFT", "NVIDIA": "NVDA",
                 "Alphabet": "GOOGL", "Amazon": "AMZN", "Tesla": "TSLA"},
    ),
    "Malaysia (Bursa)": Market(
        suffix=".KL", exchanges=("KLS",),
        benchmark="^KLSE", benchmark_name="KLCI",
        # LEAP Market codes have five digits (03011).
        currency="MYR", pattern=r"\d{4}[A-Z]{0,2}|03\d{3}", example="1155 or MAYBANK",
        pad=4, popular=BURSA_POPULAR, aliases=BURSA_SHORT_NAMES,
    ),
    "Singapore (SGX)": Market(
        suffix=".SI", exchanges=("SES",),
        benchmark="^STI", benchmark_name="STI",
        currency="SGD", pattern=r"[A-Z0-9]{3,4}", example="D05 (DBS)",
        popular=_sgx_stocks(),
    ),
    "Hong Kong (HKEX)": Market(
        suffix=".HK", exchanges=("HKG",),
        benchmark="^HSI", benchmark_name="Hang Seng",
        currency="HKD", pattern=r"\d{4,5}", example="0700 (Tencent)",
        pad=4,
        popular={"Tencent": "0700", "Alibaba": "9988", "HSBC": "0005", "AIA": "1299",
                 "Meituan": "3690", "HKEX": "0388"},
    ),
    "China (Shanghai)": Market(
        suffix=".SS", exchanges=("SHH",),
        benchmark="000001.SS", benchmark_name="SSE Composite",
        currency="CNY", pattern=r"[569]\d{5}", example="600519 (Kweichow Moutai)",
        pad=6,
        popular={"Kweichow Moutai": "600519", "ICBC": "601398",
                 "China Merchants Bank": "600036", "Ping An Insurance": "601318"},
    ),
    "China (Shenzhen)": Market(
        suffix=".SZ", exchanges=("SHZ",),
        benchmark="399001.SZ", benchmark_name="SZSE Component",
        currency="CNY", pattern=r"[0-3]\d{5}", example="300750 (CATL)",
        pad=6,
        popular={"CATL": "300750", "BYD": "002594", "Wuliangye": "000858",
                 "Ping An Bank": "000001"},
    ),
    "Japan (Tokyo)": Market(
        suffix=".T", exchanges=("JPX",),
        benchmark="^N225", benchmark_name="Nikkei 225",
        currency="JPY", pattern=r"[0-9][0-9A-Z]{3}", example="7203 (Toyota)",
        popular={"Toyota": "7203", "Sony": "6758", "SoftBank Group": "9984",
                 "Mitsubishi UFJ": "8306", "Nintendo": "7974", "Keyence": "6861"},
    ),
    "United Kingdom (London)": Market(
        suffix=".L", exchanges=("LSE",),
        benchmark="^FTSE", benchmark_name="FTSE 100",
        currency="GBP", pattern=r"[A-Z0-9]{2,4}(-[A-Z])?", example="SHEL (Shell)",
        dash_classes=True,
        popular={"Shell": "SHEL", "AstraZeneca": "AZN", "HSBC": "HSBA",
                 "Unilever": "ULVR", "BP": "BP"},
    ),
    "India (NSE)": Market(
        suffix=".NS", exchanges=("NSI",),
        benchmark="^NSEI", benchmark_name="Nifty 50",
        currency="INR", pattern=r"[A-Z0-9&\-]{1,20}", example="RELIANCE",
        popular={"Reliance Industries": "RELIANCE", "TCS": "TCS", "HDFC Bank": "HDFCBANK",
                 "Infosys": "INFY"},
    ),
    "Australia (ASX)": Market(
        suffix=".AX", exchanges=("ASX",),
        benchmark="^AXJO", benchmark_name="ASX 200",
        currency="AUD", pattern=r"[A-Z0-9]{3,6}", example="CBA (CommBank)",
        popular={"Commonwealth Bank": "CBA", "BHP": "BHP", "CSL": "CSL",
                 "National Australia Bank": "NAB"},
    ),
    "Canada (Toronto)": Market(
        suffix=".TO", exchanges=("TOR",),
        benchmark="^GSPTSE", benchmark_name="TSX",
        currency="CAD", pattern=r"[A-Z]{1,5}(-[A-Z]{1,2})?", example="RY (Royal Bank)",
        dash_classes=True,
        popular={"Royal Bank of Canada": "RY", "TD Bank": "TD", "Shopify": "SHOP",
                 "Enbridge": "ENB"},
    ),
    "Germany (XETRA)": Market(
        suffix=".DE", exchanges=("GER",),
        benchmark="^GDAXI", benchmark_name="DAX",
        currency="EUR", pattern=r"[A-Z0-9]{2,6}", example="SAP",
        popular={"SAP": "SAP", "Siemens": "SIE", "Allianz": "ALV",
                 "Deutsche Telekom": "DTE"},
    ),
}

CURRENCY_SYMBOLS = {
    "USD": "$", "MYR": "RM", "SGD": "S$", "HKD": "HK$", "CNY": "¥", "JPY": "¥",
    "GBP": "£", "INR": "₹", "AUD": "A$", "CAD": "C$", "EUR": "€",
}

# Yahoo quotes some exchanges in minor units (London in pence). Convert to the major currency.
MINOR_UNITS = {"GBp": ("GBP", 100), "GBX": ("GBP", 100), "ZAc": ("ZAR", 100), "ILA": ("ILS", 100)}

PRICE_COLUMNS = ["Open", "High", "Low", "Close"]


def build_symbol(ticker: str, market: str) -> str:
    """
    Normalize what the user typed into a Yahoo symbol for the given market.

    Examples: ('700', 'Hong Kong (HKEX)') -> '0700.HK', ('brk.b', 'United States ...') -> 'BRK-B',
    ('1155.KL', 'Malaysia (Bursa)') -> '1155.KL', ('maybank', 'Malaysia (Bursa)') -> '1155.KL'.

    Raises:
    ValueError: If the ticker doesn't look like a valid code for that market.
    """
    m = MARKETS[market]
    ticker = ticker.strip().upper()
    if m.suffix and ticker.endswith(m.suffix.upper()):  # user already typed the suffix
        ticker = ticker[: -len(m.suffix)]
    ticker = m.aliases.get(ticker, ticker)
    if m.dash_classes:
        ticker = ticker.replace(".", "-")
    if m.pad and ticker.isdigit():
        ticker = ticker.zfill(m.pad)

    if not re.fullmatch(m.pattern, ticker):
        raise ValueError(f"'{ticker}' doesn't look like a {market} ticker. Example: {m.example}.")
    return f"{ticker}{m.suffix}"


def get_currency(symbol: str, market: str) -> str:
    """Currency Yahoo quotes the symbol in, falling back to the market's usual currency."""
    try:
        return yf.Ticker(symbol).fast_info["currency"] or MARKETS[market].currency
    except Exception:
        return MARKETS[market].currency


def get_stock_data(ticker: str, market: str, start_date, end_date) -> tuple[pd.DataFrame, str]:
    """
    Fetch daily historical prices from Yahoo Finance.

    Parameters:
    ticker (str): Ticker as the user typed it, e.g. 'AAPL', '1155', '700'.
    market (str): A key of MARKETS.
    start_date, end_date (str | date): Inclusive date range.

    Returns:
    (pd.DataFrame, str): Date, Open, High, Low, Close, Volume (Close is split/dividend
    adjusted), and the ISO currency code the prices are in, e.g. 'MYR'.

    Raises:
    ValueError: If the ticker is malformed or Yahoo returns no data.
    """
    symbol = build_symbol(ticker, market)
    df = yf.download(
        symbol,
        start=start_date,
        end=pd.to_datetime(end_date) + pd.Timedelta(days=1),  # yfinance's end is exclusive
        auto_adjust=True,
        multi_level_index=False,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"No data found for '{symbol}'. Check the ticker and date range.")

    currency = get_currency(symbol, market)
    if currency in MINOR_UNITS:
        currency, divisor = MINOR_UNITS[currency]
        df[PRICE_COLUMNS] = df[PRICE_COLUMNS] / divisor

    return df.reset_index(), currency


def get_benchmark(market: str, start_date, end_date) -> pd.Series | None:
    """
    Daily closes of the market's benchmark index over an inclusive date range, indexed by date.
    None if the market has no benchmark or Yahoo can't supply it, so callers can carry on without.
    """
    symbol = MARKETS[market].benchmark
    if not symbol:
        return None
    try:
        df = yf.download(
            symbol,
            start=start_date,
            end=pd.to_datetime(end_date) + pd.Timedelta(days=1),  # yfinance's end is exclusive
            auto_adjust=True,
            multi_level_index=False,
            progress=False,
        )
    except Exception:
        return None
    return None if df.empty else df["Close"]


def format_price(value: float, currency: str) -> str:
    """Format a price for display, e.g. (10.5, 'MYR') -> 'RM 10.50', (3000, 'JPY') -> '¥3,000'."""
    decimals = 0 if currency == "JPY" else 2
    return _with_currency(f"{value:,.{decimals}f}", currency)


def format_money_compact(value: float, currency: str) -> str:
    """Format a large amount such as a market cap, e.g. (1.22e11, 'MYR') -> 'RM 122.0B'."""
    return _with_currency(compact_number(value), currency)


def compact_number(value: float, _pos=None) -> str:
    """Shorten a large number, e.g. 1_250_000 -> '1.2M'. Doubles as a matplotlib tick formatter."""
    for divisor, unit in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= divisor:
            return f"{value / divisor:.1f}{unit}"
    return f"{value:.0f}"


def format_percent(value: float, decimals: int = 1, signed: bool = False) -> str:
    """Format a fraction as a percentage, e.g. 0.1234 -> '12.3%'; a missing value -> '—'."""
    if pd.isna(value):
        return "—"
    return f"{value * 100:{'+' if signed else ''}.{decimals}f}%"


def format_ratio(value: float) -> str:
    """Format a ratio such as a Sharpe ratio, e.g. 1.234 -> '1.23'; a missing value -> '—'."""
    return "—" if pd.isna(value) else f"{value:.2f}"


def _with_currency(amount: str, currency: str) -> str:
    symbol = CURRENCY_SYMBOLS.get(currency)
    if symbol is None:
        return f"{amount} {currency}"
    return f"{symbol} {amount}" if symbol.isalpha() else f"{symbol}{amount}"
