"""Desktop UI for browsing stock prices: pick a market, a company and a date range, then plot."""
import re
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from tkinter import messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator

from evaluation import (BOLLINGER, MACD_SPANS, RSI_PERIOD, Holding, evaluate_stock, parse_rate,
                        technicals)
from news_panel import WIDTH as NEWS_WIDTH
from news_panel import NewsPanel
from peer_window import PeerWindow
from peers import get_peer_comparison
from portfolio_window import PortfolioWindow
from yfinnace import (MARKETS, build_symbol, compact_number, format_percent, format_price,
                      format_ratio, get_benchmark, get_stock_data)

# Preset ranges as days back from today; None means "since 1 January".
PERIODS = {"1M": 30, "3M": 91, "6M": 182, "YTD": None, "1Y": 365, "5Y": 1826, "Max": 365 * 60}
# Extra calendar days downloaded before the start, so indicators are warmed up on the first day
# shown (a 200-day average needs about 290 calendar days).
WARMUP_DAYS = 400
MOVING_AVERAGES = (20, 50, 200)
MA_COLORS = {20: "#f5a623", 50: "#4a90e2", 200: "#9b59b6"}
UP, DOWN, LINE = "#26a69a", "#ef5350", "#1f77b4"
BAND, RSI_LINE, MACD_LINE, SIGNAL_LINE = "#898781", "#6250d6", "#2a78d6", "#eb6834"


def fit_range(series, visible, pad=0.05):
    """(low, high) covering the series over the visible days with a margin; None if all NaN."""
    values = np.concatenate([s[visible] for s in series])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    low, high = values.min(), values.max()
    margin = (high - low) * pad or abs(high) * 0.01 or 1
    return low - margin, high + margin


def format_small(value: float) -> str:
    """Four significant figures for indicator readings whose scale follows the price."""
    return "—" if np.isnan(value) else f"{value:.4g}"


def ticker_from_choice(text: str) -> str:
    """'Tencent (0700)' -> '0700'; anything the user typed themselves is passed through."""
    match = re.fullmatch(r".*\(([^()]+)\)\s*", text)
    return (match.group(1) if match else text).strip()


class StockApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stock Analysis")
        # Up to 800 px tall but clear of the taskbar, and at the top so the toolbar isn't cut off.
        height = min(800, self.winfo_screenheight() - 120)
        self.geometry(f"1200x{height}+{max((self.winfo_screenwidth() - 1200) // 2, 0)}+0")
        self.minsize(900, 600)

        self.executor = ThreadPoolExecutor(max_workers=1)
        self.data = None       # last fetched DataFrame, just the chosen period
        self.technicals = None  # chart indicators for each day of self.data
        self.benchmark = None  # the market index's daily closes over the period, if Yahoo has them
        self.benchmark_name = ""
        self.currency = None
        self.symbol = None
        self.choices = []      # every dropdown entry for the current market
        self.portfolio = None  # PortfolioWindow, once "Add to portfolio" opens it
        self.news_widened = None  # pixels the window grew by to fit the news panel, while it's open

        # Mouse interaction state, rebuilt on every redraw.
        self.ax = None           # price axis
        self.axes = []           # price axis, then the panels below it
        self.fits = []           # (panel axis, visible slice -> y limits) refitted when zooming
        self.price_series = []   # arrays the price axis must fit when zooming
        self.drag = None         # (press x in pixels, xlim at press) while panning
        self.background = None   # canvas snapshot the crosshair is blitted onto

        self.market = tk.StringVar(value=next(iter(MARKETS)))
        self.company = tk.StringVar()
        self.period = tk.StringVar(value="1Y")
        self.start = tk.StringVar()
        self.end = tk.StringVar()
        self.chart_type = tk.StringVar(value="Line")
        self.show_volume = tk.BooleanVar(value=True)
        self.show_ma = {n: tk.BooleanVar(value=n in (20, 50)) for n in MOVING_AVERAGES}
        self.show_bollinger = tk.BooleanVar(value=False)
        self.show_rsi = tk.BooleanVar(value=False)
        self.show_macd = tk.BooleanVar(value=False)
        self.risk_free = tk.StringVar(value="4")   # percent a year, for Sharpe and Sortino
        self.status = tk.StringVar(value="Pick a company and press Plot.")

        self._build_controls()
        self._build_chart()
        self._on_market_change()
        self._on_period_change()

    # ---------- layout ----------

    def _build_controls(self):
        panel = ttk.Frame(self, padding=12)
        panel.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(panel, text="Market").pack(anchor="w")
        market_box = ttk.Combobox(panel, textvariable=self.market, values=list(MARKETS),
                                  state="readonly", width=30)
        market_box.pack(fill=tk.X, pady=(0, 10))
        market_box.bind("<<ComboboxSelected>>", lambda _e: self._on_market_change())

        ttk.Label(panel, text="Company (type to search, ↓ to list)").pack(anchor="w")
        # Let the dropdown list spill past the box so long company names aren't cut off.
        ttk.Style(self).configure("Company.TCombobox", postoffset=(0, 0, 160, 0))
        self.company_box = ttk.Combobox(panel, textvariable=self.company, width=30,
                                        style="Company.TCombobox")
        self.company_box.pack(fill=tk.X)
        self.company_box.bind("<Return>", lambda _e: self.fetch_and_plot())
        self.company_box.bind("<KeyRelease>", lambda _e: self._filter_companies())
        self.company_box.bind("<<ComboboxSelected>>", lambda _e: self._filter_companies())
        self.example = ttk.Label(panel, foreground="gray")
        self.example.pack(anchor="w", pady=(0, 10))

        ttk.Label(panel, text="Period").pack(anchor="w")
        periods = ttk.Frame(panel)
        periods.pack(fill=tk.X, pady=(0, 6))
        for i, name in enumerate(PERIODS):
            ttk.Radiobutton(periods, text=name, value=name, variable=self.period,
                            style="Toolbutton", command=self._on_period_change,
                            ).grid(row=i // 4, column=i % 4, sticky="ew", padx=1, pady=1)
        for col in range(4):
            periods.columnconfigure(col, weight=1)

        dates = ttk.Frame(panel)
        dates.pack(fill=tk.X, pady=(0, 10))
        for row, (label, var) in enumerate((("Start", self.start), ("End", self.end))):
            ttk.Label(dates, text=label, width=6).grid(row=row, column=0, sticky="w")
            entry = ttk.Entry(dates, textvariable=var, width=12)
            entry.grid(row=row, column=1, sticky="ew", pady=1)
            entry.bind("<Key>", lambda _e: self.period.set(""))  # typing a date means custom
        ttk.Label(dates, text="YYYY-MM-DD", foreground="gray").grid(row=0, column=2, padx=4)
        dates.columnconfigure(1, weight=1)

        ttk.Label(panel, text="Chart").pack(anchor="w")
        chart_types = ttk.Frame(panel)
        chart_types.pack(fill=tk.X, pady=(0, 6))
        for name in ("Line", "Candlestick"):
            ttk.Radiobutton(chart_types, text=name, value=name, variable=self.chart_type,
                            style="Toolbutton", command=self.redraw,
                            ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        # Two columns: lines drawn over the price, and panels stacked under it.
        toggles = ttk.Frame(panel)
        toggles.pack(fill=tk.X)
        overlays = [(f"MA {n}", var) for n, var in self.show_ma.items()]
        overlays.append(("Bollinger bands", self.show_bollinger))
        panels = [("Volume", self.show_volume), (f"RSI {RSI_PERIOD}", self.show_rsi),
                  ("MACD", self.show_macd)]
        for column, (heading, items) in enumerate((("On the price", overlays),
                                                   ("Panels below", panels))):
            ttk.Label(toggles, text=heading, foreground="gray").grid(row=0, column=column,
                                                                     sticky="w")
            for row, (text, var) in enumerate(items, start=1):
                ttk.Checkbutton(toggles, text=text, variable=var, command=self.redraw
                                ).grid(row=row, column=column, sticky="w")
            toggles.columnconfigure(column, weight=1)

        rate = ttk.Frame(panel)
        rate.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(rate, text="Risk-free rate").pack(side=tk.LEFT)
        ttk.Label(rate, text="% a year", foreground="gray").pack(side=tk.RIGHT)
        rate_entry = ttk.Entry(rate, textvariable=self.risk_free, width=6, justify="right")
        rate_entry.pack(side=tk.RIGHT, padx=4)
        for event in ("<Return>", "<FocusOut>"):
            rate_entry.bind(event, lambda _e: self._update_stats())

        self.plot_button = ttk.Button(panel, text="Plot", command=self.fetch_and_plot)
        self.plot_button.pack(fill=tk.X, pady=(16, 4), ipady=4)
        self.peers_button = ttk.Button(panel, text="Compare with peers", command=self.compare_peers)
        self.peers_button.pack(fill=tk.X, pady=(0, 4), ipady=4)
        ttk.Button(panel, text="Add to portfolio", command=self.add_to_portfolio
                   ).pack(fill=tk.X, pady=(0, 4), ipady=4)
        ttk.Button(panel, text="Analyze news", command=self.analyze_news
                   ).pack(fill=tk.X, pady=(0, 6), ipady=4)
        ttk.Label(panel, textvariable=self.status, wraplength=230, foreground="gray"
                  ).pack(anchor="w")

    def _build_chart(self):
        # The chart, with the news panel beside it while it's open; the divider between drags.
        self.panes = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.panes.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = ttk.Frame(self.panes, padding=(0, 12, 12, 12))
        self.panes.add(right, weight=1)
        self.news = NewsPanel(self.panes, on_close=self.close_news)

        self.stats = ttk.Frame(right)
        self.stats.pack(fill=tk.X)
        self.risk_stats = ttk.Frame(right)   # risk measures over the loaded period
        self.risk_stats.pack(fill=tk.X, pady=(4, 6))

        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.toolbar = NavigationToolbar2Tk(self.canvas, right)
        self.toolbar.update()
        ttk.Label(right, foreground="gray",
                  text="Scroll to zoom  ·  drag to pan  ·  double-click to reset"
                  ).pack(side=tk.BOTTOM, anchor="w")
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        for event, handler in (("scroll_event", self._on_scroll),
                               ("button_press_event", self._on_press),
                               ("button_release_event", self._on_release),
                               ("motion_notify_event", self._on_motion),
                               ("figure_leave_event", lambda _e: self._clear_crosshair()),
                               ("draw_event", self._on_draw)):
            self.canvas.mpl_connect(event, handler)

    # ---------- control callbacks ----------

    def _on_market_change(self):
        m = MARKETS[self.market.get()]
        self.choices = [f"{name} ({ticker})" for name, ticker in m.popular.items()]
        self.company_box["values"] = self.choices
        self.company.set(self.choices[0] if self.choices else "")
        self.example.config(text=f"e.g. {m.example}")

    def _filter_companies(self):
        """Narrow the dropdown to companies containing the typed text; a picked one shows them all."""
        text = self.company.get()
        if text in self.choices:
            matches = self.choices
        else:
            matches = [c for c in self.choices if text.strip().lower() in c.lower()]
        self.company_box["values"] = matches

    def _on_period_change(self):
        days = PERIODS.get(self.period.get())
        today = date.today()
        start = date(today.year, 1, 1) if days is None else today - timedelta(days=days)
        self.start.set(start.isoformat())
        self.end.set(today.isoformat())

    # ---------- data ----------

    def _read_inputs(self):
        """(market, ticker, start, end, symbol) from the controls, or None after showing an error."""
        market, ticker = self.market.get(), ticker_from_choice(self.company.get())
        try:
            start, end = date.fromisoformat(self.start.get()), date.fromisoformat(self.end.get())
            if start > end:
                raise ValueError("Start date must be before end date.")
            symbol = build_symbol(ticker, market)
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e))
            return None
        return market, ticker, start, end, symbol

    def fetch_and_plot(self):
        inputs = self._read_inputs()
        if inputs is None:
            return
        market, ticker, start, end, symbol = inputs

        def download():
            history, currency = get_stock_data(ticker, market,
                                               start - timedelta(days=WARMUP_DAYS), end)
            return history, currency, get_benchmark(market, start, end)

        self._run_in_background(f"Downloading {symbol}…", download, (),
                                lambda result: self._show_data(result, symbol, start, market))

    def _show_data(self, result, symbol, start, market):
        history, currency, benchmark = result
        # The warm-up days only feed the indicators; everything else covers the chosen period.
        data = history[history["Date"].dt.date >= start].reset_index(drop=True)
        if data.empty:
            self.status.set("Download failed.")
            messagebox.showerror("Could not load data",
                                 f"No data found for '{symbol}' in that date range.")
            return
        self.data, self.currency, self.benchmark = data, currency, benchmark
        self.technicals = technicals(history["Close"], len(data), MOVING_AVERAGES)
        self.benchmark_name = MARKETS[market].benchmark_name
        self.symbol = f"{self.company.get()} · {symbol}"
        status = f"Loaded {len(data)} trading days for {symbol}."
        if benchmark is None and self.benchmark_name:
            status += f" No {self.benchmark_name} data, so no beta or alpha."
        self.status.set(status)
        self.redraw()

    def _company_name(self, symbol):
        """'Tencent (0700)' -> 'Tencent'; the symbol if the user typed only a ticker."""
        return re.sub(r"\s*\([^()]*\)\s*$", "", self.company.get()).strip() or symbol

    def compare_peers(self):
        inputs = self._read_inputs()
        if inputs is None:
            return
        market, ticker, start, end, symbol = inputs
        name = self._company_name(symbol)
        period = f"{start:%d %b %Y} – {end:%d %b %Y}"

        def show(comparison):
            self.status.set(f"Compared {symbol} with {len(comparison.peers)} peers.")
            PeerWindow(self, comparison, name, market, period, on_open=self.open_peer)

        self._run_in_background(f"Finding peers of {symbol}…", get_peer_comparison,
                                (ticker, market, start, end), show)

    def open_peer(self, symbol, name):
        """Load a peer from the comparison window into the main chart."""
        suffix = MARKETS[self.market.get()].suffix
        ticker = symbol[: -len(suffix)] if suffix and symbol.endswith(suffix) else symbol
        self.company.set(f"{name} ({ticker})")
        self.lift()
        self.fetch_and_plot()

    def add_to_portfolio(self):
        """Add the chosen company to the portfolio window, opening it the first time."""
        market = self.market.get()
        try:
            symbol = build_symbol(ticker_from_choice(self.company.get()), market)
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e))
            return
        if self.portfolio is None or not self.portfolio.winfo_exists():
            self.portfolio = PortfolioWindow(self, self.risk_free.get())
        if self.portfolio.add(Holding(symbol, self._company_name(symbol), market)):
            n = len(self.portfolio.holdings)
            self.status.set(f"Added {symbol} to the portfolio ({n} stock{'s' if n != 1 else ''}).")
        else:
            self.status.set(f"{symbol} is already in the portfolio.")

    def analyze_news(self):
        """Open the news panel beside the chart and have the agent analyze the chosen company."""
        try:
            symbol = build_symbol(ticker_from_choice(self.company.get()), self.market.get())
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e))
            return
        # Search the news by the dropdown's company name ('MAYBANK · Malayan Banking Bhd.' ->
        # 'Malayan Banking Bhd.'); for a typed ticker, let the agent look the name up.
        name = None
        if self.company.get() in self.choices:
            name = self._company_name(symbol).split(" · ")[-1]
        if self.news_widened is None:
            self._open_news()
        self.news.analyze(symbol, name)

    def _open_news(self):
        self.panes.add(self.news, weight=0)
        # Grow the window to make room, as far as the screen allows, rather than squeeze the chart.
        self.news_widened = 0
        if self.state() == "normal":   # leave a maximized window alone
            width = self.winfo_width()
            self._set_width(min(width + NEWS_WIDTH, self.winfo_screenwidth()))
            self.news_widened = self.winfo_width() - width
        self.update_idletasks()
        self.panes.sashpos(0, self.panes.winfo_width() - NEWS_WIDTH)

    def close_news(self):
        self.panes.forget(self.news)
        if self.news_widened and self.state() == "normal":
            self._set_width(max(self.winfo_width() - self.news_widened, 900))
        self.news_widened = None

    def _set_width(self, width):
        """Resize the window, keeping its height, and shift it left if it would run off screen."""
        height, x, y = map(int, re.fullmatch(r"\d+x(\d+)\+(-?\d+)\+(-?\d+)",
                                             self.geometry()).groups())
        x = max(min(x, self.winfo_screenwidth() - width), 0)
        self.geometry(f"{width}x{height}+{x}+{y}")
        self.update_idletasks()

    def _run_in_background(self, message, fn, args, on_success):
        self.plot_button.state(["disabled"])
        self.peers_button.state(["disabled"])
        self.status.set(message)
        self._wait_for(self.executor.submit(fn, *args), on_success)

    def _wait_for(self, future, on_success):
        # Tk widgets must only be touched from the main thread, so poll instead of using a callback.
        if not future.done():
            self.after(100, self._wait_for, future, on_success)
            return
        self.plot_button.state(["!disabled"])
        self.peers_button.state(["!disabled"])
        try:
            result = future.result()
        except Exception as e:
            self.status.set("Download failed.")
            messagebox.showerror("Could not load data", str(e))
            return
        on_success(result)

    # ---------- drawing ----------

    def redraw(self):
        if self.data is None:
            return
        df, currency, tech = self.data, self.currency, self.technicals
        x = np.arange(len(df))

        # The price on top, then one shorter panel for each indicator switched on.
        panels = [name for name, var in (("Volume", self.show_volume), ("RSI", self.show_rsi),
                                         ("MACD", self.show_macd)) if var.get()]
        self.figure.clear()
        grid = self.figure.add_gridspec(1 + len(panels), 1, hspace=0.06,
                                        height_ratios=[3] + [1] * len(panels))
        ax = self.figure.add_subplot(grid[0])
        lower = {name: self.figure.add_subplot(grid[i + 1], sharex=ax)
                 for i, name in enumerate(panels)}
        self.axes, self.fits = [ax, *lower.values()], []

        up = (df["Close"] >= df["Open"]).to_numpy()
        colors = np.where(up, UP, DOWN)
        if self.chart_type.get() == "Candlestick":
            ax.vlines(x, df["Low"], df["High"], colors=colors, linewidth=0.8)
            body_low = df[["Open", "Close"]].min(axis=1)
            body = (df["Close"] - df["Open"]).abs().replace(0, 1e-9)  # keep doji visible
            ax.bar(x, body, bottom=body_low, width=0.6, color=colors, linewidth=0)
            self.price_series = [df["Low"].to_numpy(), df["High"].to_numpy()]
        else:
            ax.plot(x, df["Close"], color=LINE, linewidth=1.4, label="Close")
            self.price_series = [df["Close"].to_numpy()]

        for n, var in self.show_ma.items():
            ma = tech.moving_averages[n]
            if var.get() and np.isfinite(ma).any():
                ax.plot(x, ma, color=MA_COLORS[n], linewidth=1.1, label=f"MA {n}")
                self.price_series.append(ma)

        if self.show_bollinger.get() and np.isfinite(tech.bollinger_upper).any():
            period, width = BOLLINGER
            ax.fill_between(x, tech.bollinger_lower, tech.bollinger_upper, color=BAND,
                            alpha=0.12, linewidth=0, label=f"Bollinger {period}, {width:g}σ")
            for band in (tech.bollinger_upper, tech.bollinger_lower):
                ax.plot(x, band, color=BAND, linewidth=0.8)
            self.price_series += [tech.bollinger_upper, tech.bollinger_lower]

        ax.set_title(self.symbol, loc="left", fontsize=12, fontweight="bold")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: format_price(v, currency)))
        ax.grid(alpha=0.3)
        entries = len(ax.get_legend_handles_labels()[0])
        if entries:
            # A translucent backing keeps the names readable over lines; past three, go two wide.
            ax.legend(loc="upper left", fontsize=9, ncol=1 if entries <= 3 else 2,
                      framealpha=0.8, edgecolor="none")

        # Plot against trading-day index so weekends and holidays don't leave gaps.
        dates = df["Date"]

        def date_label(value, _pos):
            i = int(round(value))
            return dates.iloc[i].strftime("%d %b %Y") if 0 <= i < len(dates) else ""

        for axis in self.axes[:-1]:
            axis.tick_params(labelbottom=False)
        self.axes[-1].xaxis.set_major_locator(MaxNLocator(8, integer=True))
        self.axes[-1].xaxis.set_major_formatter(FuncFormatter(date_label))

        if "Volume" in lower:
            volume = df["Volume"].to_numpy()
            lower["Volume"].bar(x, volume, width=0.8, color=colors, alpha=0.6, linewidth=0)
            lower["Volume"].yaxis.set_major_formatter(FuncFormatter(compact_number))
            self.fits.append((lower["Volume"], lambda v: (0, max(volume[v].max(), 1) * 1.1)))
        if "RSI" in lower:
            rsi_ax = lower["RSI"]
            rsi_ax.axhspan(30, 70, color=RSI_LINE, alpha=0.06, linewidth=0)
            for level in (30, 70):   # oversold / overbought
                rsi_ax.axhline(level, color="gray", linewidth=0.7, alpha=0.6)
            rsi_ax.plot(x, tech.rsi, color=RSI_LINE, linewidth=1.1)
            rsi_ax.set_ylim(0, 100)
            rsi_ax.set_yticks([30, 70])
        if "MACD" in lower:
            macd_ax = lower["MACD"]
            fast, slow, signal = MACD_SPANS
            histogram = tech.macd_histogram
            macd_ax.axhline(0, color="gray", linewidth=0.7, alpha=0.6)
            macd_ax.bar(x, histogram, width=0.8, color=np.where(histogram >= 0, UP, DOWN),
                        alpha=0.5, linewidth=0, label="Histogram")
            macd_ax.plot(x, tech.macd, color=MACD_LINE, linewidth=1.1,
                         label=f"MACD {fast}, {slow}")
            macd_ax.plot(x, tech.macd_signal, color=SIGNAL_LINE, linewidth=1.1,
                         label=f"Signal {signal}")
            macd_ax.yaxis.set_major_locator(MaxNLocator(3))
            macd_ax.legend(loc="upper left", ncol=3, fontsize=8, framealpha=0.8, edgecolor="none",
                           borderaxespad=0.2, borderpad=0.2, handlelength=1.5)
            lines = [tech.macd, tech.macd_signal, histogram]
            self.fits.append((macd_ax, lambda v: fit_range(lines, v, pad=0.15)))
        for name, panel in lower.items():
            panel.grid(alpha=0.3)
            if name != "MACD":   # the MACD legend already names its panel
                panel.text(0.005, 0.95, f"RSI {RSI_PERIOD}" if name == "RSI" else name,
                           transform=panel.transAxes, ha="left", va="top", fontsize=8,
                           color="gray", bbox=dict(facecolor="white", alpha=0.8,
                                                   edgecolor="none", pad=1))

        # Crosshair artists are "animated" so normal draws skip them; they're blitted on hover.
        cross = dict(color="gray", linewidth=0.8, linestyle="--", animated=True)
        self.cross_v = [a.axvline(len(df) - 1, **cross) for a in self.axes]
        self.cross_h = ax.axhline(df["Close"].iloc[-1], **cross)
        self.info = ax.text(0.99, 0.97, "", transform=ax.transAxes, ha="right", va="top",
                            fontsize=9, family="monospace", animated=True,
                            bbox=dict(boxstyle="round", facecolor="white",
                                      edgecolor="lightgray", alpha=0.9))

        self.ax, self.background, self.drag = ax, None, None
        self.figure.subplots_adjust(left=0.1, right=0.98, top=0.93, bottom=0.08)
        self._set_xlim(-1, len(df))
        self.toolbar.update()        # make the toolbar's Home button return to this view
        self.toolbar.push_current()
        self._update_stats()

    # ---------- mouse interaction ----------

    def _set_xlim(self, lo, hi):
        """Show trading days lo..hi, kept inside the data, and refit the y axes to them."""
        n = len(self.data)
        width = min(max(hi - lo, 10), n + 1)   # at least 10 bars, at most everything
        lo = max(-1, min(lo, n - width))
        hi = lo + width
        self.ax.set_xlim(lo, hi)               # the panels follow through sharex

        visible = slice(max(int(np.ceil(lo)), 0), min(int(np.floor(hi)), n - 1) + 1)
        self.ax.set_ylim(*fit_range(self.price_series, visible))
        for axis, fit in self.fits:
            limits = fit(visible)
            if limits is not None:   # None while an indicator is still warming up
                axis.set_ylim(*limits)
        self.canvas.draw_idle()

    def _in_chart(self, event):
        return self.ax is not None and event.inaxes in self.axes

    def _on_scroll(self, event):
        if not self._in_chart(event):
            return
        lo, hi = self.ax.get_xlim()
        factor = 0.8 if event.button == "up" else 1.25   # wheel up zooms in
        x = event.xdata
        self._set_xlim(x - (x - lo) * factor, x + (hi - x) * factor)

    def _on_press(self, event):
        # Leave clicks alone while the toolbar's own zoom/pan tool is active.
        if not self._in_chart(event) or self.toolbar.mode:
            return
        if event.dblclick:
            self._set_xlim(-1, len(self.data))
        elif event.button == 1:
            self.drag = (event.x, self.ax.get_xlim())
            self.canvas.get_tk_widget().config(cursor="fleur")
            self._clear_crosshair()

    def _on_release(self, _event):
        if self.drag is not None:
            self.drag = None
            self.canvas.get_tk_widget().config(cursor="")

    def _on_motion(self, event):
        if self.drag is not None:
            press_x, (lo, hi) = self.drag
            shift = (event.x - press_x) * (hi - lo) / self.ax.bbox.width
            self._set_xlim(lo - shift, hi - shift)
        elif self._in_chart(event) and event.xdata is not None:
            self._draw_crosshair(event)
        else:
            self._clear_crosshair()

    def _on_draw(self, _event):
        self.background = self.canvas.copy_from_bbox(self.figure.bbox)

    def _clear_crosshair(self):
        if self.background is not None:
            self.canvas.restore_region(self.background)
            self.canvas.blit(self.figure.bbox)

    def _draw_crosshair(self, event):
        if self.background is None:
            return
        df, currency = self.data, self.currency
        i = int(np.clip(round(event.xdata), 0, len(df) - 1))
        row = df.iloc[i]
        prev_close = df["Close"].iloc[i - 1] if i > 0 else row["Open"]
        pct = (row["Close"] - prev_close) / prev_close * 100
        lines = [
            row["Date"].strftime("%a %d %b %Y"),
            f"Open   {format_price(row['Open'], currency)}",
            f"High   {format_price(row['High'], currency)}",
            f"Low    {format_price(row['Low'], currency)}",
            f"Close  {format_price(row['Close'], currency)} ({pct:+.2f}%)",
            f"Volume {compact_number(row['Volume'])}",
        ]
        tech = self.technicals
        if self.show_rsi.get():
            lines.append(f"RSI    {format_small(tech.rsi[i])}")
        if self.show_macd.get():
            lines.append(f"MACD   {format_small(tech.macd[i])}  "
                         f"signal {format_small(tech.macd_signal[i])}")
        self.info.set_text("\n".join(lines))

        self.canvas.restore_region(self.background)
        for line in self.cross_v:
            line.set_xdata([i, i])
            line.axes.draw_artist(line)
        if event.inaxes is self.ax:
            self.cross_h.set_ydata([event.ydata, event.ydata])
            self.ax.draw_artist(self.cross_h)
        self.ax.draw_artist(self.info)
        self.canvas.blit(self.figure.bbox)

    def _update_stats(self):
        if self.data is None:
            return
        for frame in (self.stats, self.risk_stats):
            for child in frame.winfo_children():
                child.destroy()
        close, currency = self.data["Close"], self.currency
        change = close.iloc[-1] - close.iloc[0]
        pct = change / close.iloc[0] * 100
        items = [
            ("Last close", format_price(close.iloc[-1], currency), None),
            ("Change", f"{'+' if change >= 0 else '−'}{format_price(abs(change), currency)} "
                       f"({pct:+.2f}%)", UP if change >= 0 else DOWN),
            ("Period high", format_price(self.data["High"].max(), currency), None),
            ("Period low", format_price(self.data["Low"].min(), currency), None),
        ]
        for label, value, color in items:
            self._tile(self.stats, label, value, color).pack(side=tk.LEFT)

        try:
            rate = parse_rate(self.risk_free.get())
        except ValueError as e:
            rate = np.nan   # Sharpe, Sortino and alpha show as "—"
            self.status.set(str(e))
        risk = evaluate_stock(self.data.set_index("Date")["Close"], rate, self.benchmark)
        rsi = self.technicals.rsi[-1]
        rsi_text = "—" if np.isnan(rsi) else f"{rsi:.0f} · " + (
            "overbought" if rsi >= 70 else "oversold" if rsi <= 30 else "neutral")
        vs = f" vs {self.benchmark_name}" if self.benchmark_name else ""
        # Returns and return-for-risk on top, risk underneath, both in matching columns.
        rows = [
            [("CAGR", format_percent(risk.cagr, signed=True)),
             ("Alpha" + vs, format_percent(risk.alpha, signed=True)),
             ("Sharpe ratio", format_ratio(risk.sharpe)),
             ("Sortino ratio", format_ratio(risk.sortino)),
             ("Calmar ratio", format_ratio(risk.calmar)),
             (f"RSI {RSI_PERIOD}", rsi_text)],
            [("Volatility p.a.", format_percent(risk.volatility)),
             ("Beta" + vs, format_ratio(risk.beta)),
             ("Max drawdown", format_percent(risk.max_drawdown)),
             ("VaR 95% · 1 day", format_percent(risk.var, 2)),
             ("CVaR 95% · 1 day", format_percent(risk.cvar, 2))],
        ]
        for r, tiles in enumerate(rows):
            for c, (label, value) in enumerate(tiles):
                self._tile(self.risk_stats, label, value, size=11
                           ).grid(row=r, column=c, sticky="w", pady=(0, 4))

    @staticmethod
    def _tile(parent, label, value, color=None, size=13):
        """A stat tile (grey label over a bold value) for the caller to pack or grid."""
        box = ttk.Frame(parent, padding=(0, 0, 24 if size >= 13 else 20, 0))
        ttk.Label(box, text=label, foreground="gray").pack(anchor="w")
        ttk.Label(box, text=value, font=("Segoe UI", size, "bold"),
                  foreground=color or "").pack(anchor="w")
        return box


if __name__ == "__main__":
    app = StockApp()
    app.protocol("WM_DELETE_WINDOW", lambda: (app.executor.shutdown(wait=False),
                                              app.news.executor.shutdown(wait=False,
                                                                         cancel_futures=True),
                                              app.destroy()))
    app.mainloop()
