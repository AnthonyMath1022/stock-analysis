"""Fetch a stock's recent news, score each headline with FinBERT and have a local Hugging Face
LLM write a short analysis of it. Both models are downloaded once and then run offline.

    py -3.11 agent.py 1155.KL           # Maybank
    py -3.11 agent.py AAPL --no-llm     # FinBERT scores only
"""
import email.utils
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from functools import lru_cache
from urllib.parse import urljoin

import pandas as pd
import requests
import torch
import yfinance as yf
from finvizfinance.quote import finvizfinance

# Without Developer Mode, Windows can't symlink the model cache; it copies instead, which is fine.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
from transformers import (AutoModelForCausalLM, AutoTokenizer, TextStreamer,  # noqa: E402
                          pipeline)

FINBERT_MODEL = "ProsusAI/finbert"
# ~4 GB on the GPU. Qwen/Qwen3-4B-Instruct-2507 writes better but needs ~8 GB on its own.
LLM_MODEL = "Qwen/Qwen3-1.7B"
MAX_NEW_TOKENS = 500

MAX_HEADLINES = 30   # newest headlines kept, and sent to the LLM
NEWS_DAYS = 30       # how far back Google News searches
POSITIVE_ABOVE = 0.15   # average FinBERT score (-1..+1) above this reads as positive overall
NEGATIVE_BELOW = -0.15

GOOGLE_NEWS = "https://news.google.com/rss/search"
# Yahoo suffix -> country of the English Google News edition to search. FinBERT only reads
# English, so markets without an English edition (China, Japan, Germany) use the US one.
NEWS_EDITIONS = {"": "US", ".KL": "MY", ".SI": "SG", ".HK": "HK", ".L": "GB", ".NS": "IN",
                 ".AX": "AU", ".TO": "CA"}
# Words a company name is cut at to get a search term: 'DBS Group Holdings Ltd' -> 'DBS'.
NAME_ENDINGS = {"bhd", "berhad", "ltd", "limited", "inc", "corp", "corporation", "co", "plc",
                "holdings", "holding", "group", "ag", "se", "nv", "sa", "tbk", "ord"}
# Quote and forecast pages Google News lists as articles; they carry no news and score neutral.
QUOTE_PAGE = re.compile(r"stock price|price, chart|stock forecast|estimates [–-] |^\$\S+ \(",
                        re.IGNORECASE)


@dataclass
class NewsAnalysis:
    symbol: str              # e.g. '1155.KL'
    name: str                # e.g. 'Malayan Banking Berhad'
    headlines: pd.DataFrame  # newest first: date, title, source, link, sentiment, positive,
                             # negative, neutral, score (= positive - negative, -1..+1)
    summary: str | None = None   # the local LLM's write-up, once summarize() has run
    messages: list = field(default_factory=list)   # the conversation with the LLM so far

    @property
    def score(self) -> float:
        return float(self.headlines["score"].mean()) if len(self.headlines) else 0.0

    @property
    def sentiment(self) -> str:
        """'positive', 'negative' or 'neutral', from the average headline score."""
        if self.score > POSITIVE_ABOVE:
            return "positive"
        if self.score < NEGATIVE_BELOW:
            return "negative"
        return "neutral"

    @property
    def counts(self) -> dict:
        """Headlines per FinBERT label, e.g. {'positive': 12, 'negative': 5, 'neutral': 13}."""
        counts = self.headlines["sentiment"].value_counts()
        return {label: int(counts.get(label, 0)) for label in ("positive", "negative", "neutral")}

    def daily(self) -> pd.Series:
        """Average headline score per day, oldest first."""
        return self.headlines.groupby(self.headlines["date"].dt.date)["score"].mean()

    def ask(self, question: str, on_text=None) -> str:
        """
        The local LLM's answer to a question about these headlines. Earlier questions and
        answers are remembered, so follow-ups work. If on_text is given, it's called with each
        piece of the answer as it's generated.
        """
        messages = self.messages or [{"role": "system",
                                      "content": f"{ANALYST_PROMPT}\n\n{_news_context(self)}"}]
        messages = messages + [{"role": "user", "content": question}]
        answer = ask_llm(messages, on_text)
        self.messages = messages + [{"role": "assistant", "content": answer}]
        return answer

    def summarize(self, on_text=None) -> str:
        """The LLM's analysis of the headlines: overall view, key themes, risks and catalysts."""
        self.summary = self.ask(ANALYSIS_REQUEST, on_text)
        return self.summary


# ---------------------------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------------------------

def _search_term(name: str) -> str:
    """'Malayan Banking Berhad' -> 'Malayan Banking', 'SHELL PLC ORD EUR0.07' -> 'SHELL'."""
    words = []
    for word in name.split():
        if word.strip(".,").lower() in NAME_ENDINGS:
            break
        words.append(word)
    return " ".join(words)


def _company_names(symbol: str) -> list[str]:
    """Yahoo's long and short name, e.g. ['Malayan Banking Berhad', 'MAYBANK']."""
    try:
        info = yf.Ticker(symbol).info
    except Exception:
        return []
    return [n for n in (info.get("longName"), info.get("shortName")) if n]


def _finviz_news(symbol: str) -> pd.DataFrame:
    """Headlines finviz has collected for a US ticker over the last few days."""
    df = finvizfinance(symbol).ticker_news()
    return pd.DataFrame({"date": pd.to_datetime(df["Date"]), "title": df["Title"].str.strip(),
                         "source": df["Source"],
                         # finviz's own articles have site-relative links: /news/407468/...
                         "link": df["Link"].map(lambda link: urljoin("https://finviz.com", link))})


def _google_news(terms: list[str], country: str) -> pd.DataFrame:
    """Headlines from Google News mentioning any of the terms in the last NEWS_DAYS days."""
    query = " OR ".join(f'"{t}"' for t in terms) + f" when:{NEWS_DAYS}d"
    params = {"q": query, "hl": f"en-{country}", "gl": country, "ceid": f"{country}:en"}
    response = requests.get(GOOGLE_NEWS, params=params, timeout=15)
    response.raise_for_status()

    rows = []
    for item in ET.fromstring(response.content).iterfind("./channel/item"):
        source = item.findtext("source") or ""
        title = item.findtext("title") or ""
        if source and title.endswith(f" - {source}"):   # Google appends the source to titles
            title = title[: -len(source) - 3]
        published = email.utils.parsedate_to_datetime(item.findtext("pubDate"))
        rows.append({"date": published.replace(tzinfo=None), "title": title.strip(),
                     "source": source, "link": item.findtext("link")})
    return pd.DataFrame(rows, columns=["date", "title", "source", "link"])


def fetch_news(symbol: str, name: str | None = None) -> pd.DataFrame:
    """
    The newest MAX_HEADLINES headlines about a stock, newest first, with columns date, title,
    source and link. US stocks come from finviz, everything else (or a US stock finviz doesn't
    know) from Google News, searched by company name.

    Args:
    symbol: Yahoo symbol, e.g. 'AAPL', '1155.KL', '0700.HK'.
    name: Company name to search for. Yahoo's names are used if it isn't given.
    """
    suffix = re.search(r"\.[A-Z]+$", symbol)
    suffix = suffix.group() if suffix else ""

    news = pd.DataFrame()
    if not suffix:
        try:
            news = _finviz_news(symbol)
        except Exception:
            pass   # not on finviz (or finviz is down): fall back to Google News
    if news.empty:
        names = ([name] if name else []) + _company_names(symbol)
        terms = list({_search_term(n).lower(): _search_term(n) for n in names
                      if _search_term(n)}.values())
        news = _google_news(terms or [symbol], NEWS_EDITIONS.get(suffix, "US"))

    news = news[(news["title"] != "") & ~news["title"].str.contains(QUOTE_PAGE)]
    news = news.loc[~news["title"].str.lower().duplicated()]   # same story, several outlets
    return news.sort_values("date", ascending=False).head(MAX_HEADLINES).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _finbert():
    return pipeline("text-classification", model=FINBERT_MODEL, top_k=None,
                    device=0 if torch.cuda.is_available() else -1)


@lru_cache(maxsize=1)
def _llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, dtype="auto", device_map="auto")
    return tokenizer, model


def score_headlines(titles: list[str]) -> pd.DataFrame:
    """
    FinBERT's probabilities for each headline, one row per title, with columns sentiment
    (the most likely label), positive, negative, neutral and score (positive - negative).
    """
    if not titles:
        return pd.DataFrame(columns=["sentiment", "positive", "negative", "neutral", "score"])
    results = _finbert()(titles, batch_size=16, truncation=True)
    probs = pd.DataFrame([{r["label"]: r["score"] for r in result} for result in results])
    probs = probs[["positive", "negative", "neutral"]]
    probs.insert(0, "sentiment", probs.idxmax(axis=1))
    probs["score"] = probs["positive"] - probs["negative"]
    return probs


class _Streamer(TextStreamer):
    """Hands each finished piece of the reply to a function as it's generated."""

    def __init__(self, tokenizer, on_text):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.on_text = on_text

    def on_finalized_text(self, text: str, stream_end: bool = False):
        if text:
            self.on_text(text)


def ask_llm(messages: list[dict], on_text=None, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """
    The local LLM's next reply in a conversation of {'role': ..., 'content': ...} messages.
    If on_text is given, it's called with each piece of the reply as it's generated.
    """
    tokenizer, model = _llm()
    # enable_thinking=False skips Qwen3's hidden reasoning; other models' templates ignore it.
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                           enable_thinking=False, return_tensors="pt",
                                           return_dict=True).to(model.device)
    # Qwen3's recommended settings for non-thinking mode.
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                            temperature=0.7, top_p=0.8, top_k=20,
                            streamer=_Streamer(tokenizer, on_text) if on_text else None)
    reply = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(reply, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------------------------

ANALYST_PROMPT = (
    "You are an equity research analyst. You are given recent news headlines about one listed "
    "company, each tagged with a FinBERT sentiment score from -1 (negative) to +1 (positive). "
    "Base everything you say only on these headlines: do not invent figures, dates or events "
    "that are not in them, and if they don't answer a question, say so. Ignore headlines that "
    "are about other companies or only mention this one in passing. Answer follow-up questions "
    "in a few sentences."
)
ANALYSIS_REQUEST = (
    "Write a short analysis under these headings:\n"
    "Overall view: Bullish, Bearish or Mixed, with a one-sentence reason.\n"
    "Key themes: 3-5 bullet points.\n"
    "Risks: bullet points.\n"
    "Possible catalysts: bullet points.\n"
    "Keep it under 250 words."
)


def _news_context(analysis: NewsAnalysis) -> str:
    """The company, FinBERT's verdict and every headline, for the LLM's system prompt."""
    counts = analysis.counts
    lines = [f"{row.date:%Y-%m-%d} | {row.score:+.2f} | {row.title} ({row.source})"
             for row in analysis.headlines.itertuples()]
    return (
        f"Company: {analysis.name} ({analysis.symbol})\n"
        f"FinBERT: {counts['positive']} positive, {counts['negative']} negative, "
        f"{counts['neutral']} neutral headlines; average score {analysis.score:+.2f} "
        f"({analysis.sentiment}).\n\n"
        "Headlines, newest first (date | score | title (source)):\n" + "\n".join(lines)
    )


def analyze_news(symbol: str, name: str | None = None, summarize: bool = True) -> NewsAnalysis:
    """
    Fetch a stock's recent headlines, score them with FinBERT and, if summarize, have the
    local LLM write an analysis of them. The first call downloads the models.

    Args:
    symbol: Yahoo symbol, e.g. 'AAPL', '1155.KL'.
    name: Company name, used to search for news and in the LLM prompt.
    summarize: False to skip the (slower) LLM and return FinBERT scores only.

    Raises:
    ValueError: If no news was found for the stock.
    """
    news = fetch_news(symbol, name)
    if news.empty:
        raise ValueError(f"No recent news found for {symbol}.")
    headlines = pd.concat([news, score_headlines(news["title"].tolist())], axis=1)

    names = _company_names(symbol) if name is None else [name]
    analysis = NewsAnalysis(symbol, names[0] if names else symbol, headlines)
    if summarize:
        analysis.summarize()
    return analysis


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score a stock's recent news with FinBERT and "
                                                 "summarize it with a local LLM.")
    parser.add_argument("symbol", help="Yahoo symbol, e.g. AAPL, 1155.KL, D05.SI, 0700.HK")
    parser.add_argument("--name", help="company name to search news for (default: Yahoo's)")
    parser.add_argument("--no-llm", action="store_true", help="only score headlines with FinBERT")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")   # headlines have curly quotes; cp1252 can't print them

    result = analyze_news(args.symbol.upper(), args.name, summarize=not args.no_llm)
    counts = result.counts
    print(f"\n{result.name} ({result.symbol}): {result.sentiment.upper()} "
          f"(average {result.score:+.2f}; {counts['positive']} positive, "
          f"{counts['negative']} negative, {counts['neutral']} neutral)\n")
    for row in result.headlines.itertuples():
        print(f"{row.date:%Y-%m-%d}  {row.sentiment:<8} {row.score:+.2f}  {row.title} "
              f"({row.source})")
    if result.summary:
        print(f"\n--- {LLM_MODEL} ---\n{result.summary}")
