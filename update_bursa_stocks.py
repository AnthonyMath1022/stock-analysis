"""Rebuild data/bursa_stocks.csv, the list of Bursa Malaysia stocks offered in the company dropdown.

The list comes from the kjhq/Malaysia-Stock-Symbols-and-Metadata dataset on Hugging Face, which is
updated weekly. It names stocks by their Bursa short name (MAYBANK), but Yahoo Finance only knows
Bursa stocks by their numeric code (1155.KL). Yahoo's screener reports that same short name as a
stock's shortName, so it is used to look up each code, with Yahoo's search as a fallback for the
few listed stocks the screener leaves out. Rows come out largest market cap first.

Run it again whenever you want a fresh list:  python update_bursa_stocks.py
"""
from pathlib import Path

import pandas as pd
import yfinance as yf
from yfinance import EquityQuery

SOURCE = ("https://huggingface.co/datasets/kjhq/Malaysia-Stock-Symbols-and-Metadata"
          "/resolve/main/malaysia.csv")
OUTPUT = Path(__file__).parent / "data" / "bursa_stocks.csv"
PAGE_SIZE = 250   # most rows Yahoo's screener returns per request


def yahoo_codes() -> pd.DataFrame:
    """Every stock Yahoo lists on Bursa: short_name, code, largest market cap first."""
    query = EquityQuery("is-in", ["exchange", "KLS"])
    quotes, offset = [], 0
    while True:
        page = yf.screen(query, offset=offset, size=PAGE_SIZE,
                         sortField="intradaymarketcap", sortAsc=False)
        quotes += page["quotes"]
        offset += PAGE_SIZE
        if offset >= page["total"] or not page["quotes"]:
            break
    return pd.DataFrame({
        "short_name": [q.get("shortName", "").strip().upper() for q in quotes],
        "code": [q["symbol"].removesuffix(".KL") for q in quotes],
    }).drop_duplicates("short_name")


def search_code(short_name: str) -> str | None:
    """Code of the Bursa stock whose Yahoo short name is exactly this, via Yahoo's search."""
    for q in yf.Search(short_name, max_results=10, news_count=0).quotes:
        if q["symbol"].endswith(".KL") and q.get("shortname", "").strip().upper() == short_name:
            return q["symbol"].removesuffix(".KL")
    return None


def main():
    stocks = pd.read_csv(SOURCE, dtype=str, keep_default_na=False)
    stocks["short_name"] = stocks["ticker"].str.strip().str.upper()
    codes = yahoo_codes()

    unscreened = stocks.loc[~stocks["short_name"].isin(codes["short_name"]), "short_name"]
    found = {name: code for name in unscreened if (code := search_code(name))}
    codes = pd.concat([codes, pd.DataFrame({"short_name": list(found), "code": list(found.values())})])

    # Inner join in Yahoo's order, so the list keeps its largest-first sort.
    matched = codes.merge(stocks, on="short_name")[["code", "short_name", "name", "sector"]]
    OUTPUT.parent.mkdir(exist_ok=True)
    matched.to_csv(OUTPUT, index=False)

    missing = stocks.loc[~stocks["short_name"].isin(matched["short_name"]), "short_name"]
    print(f"Wrote {len(matched)} of {len(stocks)} stocks to {OUTPUT}.")
    if len(missing):
        print(f"No Yahoo code found for {len(missing)} (likely delisted or suspended): "
              + ", ".join(missing))


if __name__ == "__main__":
    main()
