"""Rebuild the stock lists offered in the company dropdown: data/bursa_stocks.csv (Bursa Malaysia)
and data/sgx_stocks.csv (Singapore Exchange).

Both come from kjhq's Stock-Symbols-and-Metadata datasets on Hugging Face, which are meant to be
updated weekly. Each stock is checked against Yahoo Finance, which the app downloads prices from,
and ones Yahoo no longer lists (delisted, taken private, renamed) are left out. Rows come out
largest market cap first, then the stocks Yahoo's screener leaves out, which have no market cap.

The Malaysian dataset names stocks by their Bursa short name (MAYBANK), but Yahoo only knows Bursa
stocks by their numeric code (1155.KL). Yahoo's screener reports that same short name as a stock's
shortName, so it is used to look up each code, with Yahoo's search as a fallback. The Singapore
dataset already uses SGX codes (D05), so those are only checked for recent prices.

Run it again whenever you want fresh lists:  python update_stock_lists.py
"""
from pathlib import Path

import pandas as pd
import yfinance as yf
from yfinance import EquityQuery

DATASET = "https://huggingface.co/datasets/kjhq/{0}-Stock-Symbols-and-Metadata/resolve/main/{1}.csv"
DATA = Path(__file__).parent / "data"
PAGE_SIZE = 250   # most rows Yahoo's screener returns per request


def read_dataset(country: str) -> pd.DataFrame:
    """The Hugging Face dataset for a country: name, ticker, market, sector."""
    # keep_default_na stops tickers like NA becoming NaN.
    stocks = pd.read_csv(DATASET.format(country, country.lower()), dtype=str, keep_default_na=False)
    stocks["ticker"] = stocks["ticker"].str.strip().str.upper()
    return stocks


def yahoo_listings(exchange: str, suffix: str) -> pd.DataFrame:
    """Every stock Yahoo's screener has on an exchange: code, short_name, largest market cap first."""
    query = EquityQuery("is-in", ["exchange", exchange])
    quotes, offset = [], 0
    while True:
        page = yf.screen(query, offset=offset, size=PAGE_SIZE,
                         sortField="intradaymarketcap", sortAsc=False)
        quotes += page["quotes"]
        offset += PAGE_SIZE
        if offset >= page["total"] or not page["quotes"]:
            break
    return pd.DataFrame({
        "code": [q["symbol"].removesuffix(suffix) for q in quotes],
        "short_name": [q.get("shortName", "").strip().upper() for q in quotes],
    })


def search_bursa_code(short_name: str) -> str | None:
    """Code of the Bursa stock whose Yahoo short name is exactly this, via Yahoo's search."""
    for q in yf.Search(short_name, max_results=10, news_count=0).quotes:
        if q["symbol"].endswith(".KL") and q.get("shortname", "").strip().upper() == short_name:
            return q["symbol"].removesuffix(".KL")
    return None


def traded_recently(codes: list[str], suffix: str) -> list[str]:
    """The codes Yahoo has prices for in the last three months."""
    if not codes:
        return []
    symbols = [code + suffix for code in codes]
    close = yf.download(symbols, period="3mo", progress=False)["Close"]
    return [code for code, s in zip(codes, symbols) if s in close and close[s].notna().any()]


def save(filename: str, stocks: pd.DataFrame, missing: list[str]):
    DATA.mkdir(exist_ok=True)
    stocks.to_csv(DATA / filename, index=False)
    print(f"Wrote {len(stocks)} of {len(stocks) + len(missing)} stocks to {DATA / filename}.")
    if missing:
        print(f"  Not on Yahoo Finance ({len(missing)}, likely delisted, taken private or renamed): "
              + ", ".join(missing))


def build_bursa():
    stocks = read_dataset("Malaysia").rename(columns={"ticker": "short_name"})
    listed = yahoo_listings("KLS", ".KL").drop_duplicates("short_name")
    unscreened = stocks.loc[~stocks["short_name"].isin(listed["short_name"]), "short_name"]
    found = {name: code for name in unscreened if (code := search_bursa_code(name))}
    listed = pd.concat([listed, pd.DataFrame({"short_name": list(found), "code": list(found.values())})])

    # Yahoo's screener and search each miss a few stocks at random, so keep last build's code for
    # any stock still not found, as long as it has recent prices.
    if (DATA / "bursa_stocks.csv").exists():
        last = pd.read_csv(DATA / "bursa_stocks.csv", dtype=str, keep_default_na=False)
        last = last[last["short_name"].isin(stocks["short_name"])
                    & ~last["short_name"].isin(listed["short_name"])]
        alive = traded_recently(last["code"].tolist(), ".KL")
        listed = pd.concat([listed, last.loc[last["code"].isin(alive), ["code", "short_name"]]])

    # Inner join in Yahoo's order, so the list keeps its largest-first sort.
    matched = listed.merge(stocks, on="short_name")[["code", "short_name", "name", "sector"]]
    save("bursa_stocks.csv", matched,
         stocks.loc[~stocks["short_name"].isin(matched["short_name"]), "short_name"].tolist())


def build_sgx():
    stocks = read_dataset("Singapore").rename(columns={"ticker": "code"})
    listed = yahoo_listings("SES", ".SI")[["code"]]
    unscreened = stocks.loc[~stocks["code"].isin(listed["code"]), "code"].tolist()
    listed = pd.concat([listed, pd.DataFrame({"code": traded_recently(unscreened, ".SI")})])

    matched = listed.merge(stocks, on="code")[["code", "name", "sector"]]
    save("sgx_stocks.csv", matched, stocks.loc[~stocks["code"].isin(matched["code"]), "code"].tolist())


if __name__ == "__main__":
    build_bursa()
    build_sgx()
