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

from peer_window import PeerWindow
from peers import get_peer_comparison
from yfinnace import MARKETS, build_symbol, compact_number, format_price, get_stock_data

# Preset ranges as days back from today; None means "since 1 January".
PERIODS = {"1M": 30, "3M": 91, "6M": 182, "YTD": None, "1Y": 365, "5Y": 1826, "Max": 365 * 60}
MOVING_AVERAGES = (20, 50, 200)
MA_COLORS = {20: "#f5a623", 50: "#4a90e2", 200: "#9b59b6"}
UP, DOWN, LINE = "#26a69a", "#ef5350", "#1f77b4"


def ticker_from_choice(text: str) -> str:
    """'Tencent (0700)' -> '0700'; anything the user typed themselves is passed through."""
    match = re.fullmatch(r".*\(([^()]+)\)\s*", text)
    return (match.group(1) if match else text).strip()


class StockApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stock Analysis")
        self.geometry("1200x720")
        self.minsize(900, 560)

        self.executor = ThreadPoolExecutor(max_workers=1)
        self.data = None       # last fetched DataFrame
        self.currency = None
        self.symbol = None
        self.choices = []      # every dropdown entry for the current market

        # Mouse interaction state, rebuilt on every redraw.
        self.ax = self.vol_ax = None
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

        ttk.Checkbutton(panel, text="Volume", variable=self.show_volume,
                        command=self.redraw).pack(anchor="w")
        for n, var in self.show_ma.items():
            ttk.Checkbutton(panel, text=f"{n}-day moving average", variable=var,
                            command=self.redraw).pack(anchor="w")

        self.plot_button = ttk.Button(panel, text="Plot", command=self.fetch_and_plot)
        self.plot_button.pack(fill=tk.X, pady=(16, 4), ipady=4)
        self.peers_button = ttk.Button(panel, text="Compare with peers", command=self.compare_peers)
        self.peers_button.pack(fill=tk.X, pady=(0, 6), ipady=4)
        ttk.Label(panel, textvariable=self.status, wraplength=230, foreground="gray"
                  ).pack(anchor="w")

    def _build_chart(self):
        right = ttk.Frame(self, padding=(0, 12, 12, 12))
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.stats = ttk.Frame(right)
        self.stats.pack(fill=tk.X, pady=(0, 6))

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
        self._run_in_background(f"Downloading {symbol}…", get_stock_data,
                                (ticker, market, start, end),
                                lambda result: self._show_data(result, symbol))

    def _show_data(self, result, symbol):
        self.data, self.currency = result
        self.symbol = f"{self.company.get()} · {symbol}"
        self.status.set(f"Loaded {len(self.data)} trading days for {symbol}.")
        self.redraw()

    def compare_peers(self):
        inputs = self._read_inputs()
        if inputs is None:
            return
        market, ticker, start, end, symbol = inputs
        name = re.sub(r"\s*\([^()]*\)\s*$", "", self.company.get()).strip() or symbol
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
        df, currency = self.data, self.currency
        x = np.arange(len(df))

        self.figure.clear()
        if self.show_volume.get():
            grid = self.figure.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
            ax = self.figure.add_subplot(grid[0])
            vol_ax = self.figure.add_subplot(grid[1], sharex=ax)
        else:
            ax, vol_ax = self.figure.add_subplot(111), None

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
            if var.get() and len(df) > n:
                ma = df["Close"].rolling(n).mean()
                ax.plot(x, ma, color=MA_COLORS[n], linewidth=1.1, label=f"MA {n}")
                self.price_series.append(ma.to_numpy())

        ax.set_title(self.symbol, loc="left", fontsize=12, fontweight="bold")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: format_price(v, currency)))
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper left", fontsize=9, frameon=False)

        # Plot against trading-day index so weekends and holidays don't leave gaps.
        dates = df["Date"]

        def date_label(value, _pos):
            i = int(round(value))
            return dates.iloc[i].strftime("%d %b %Y") if 0 <= i < len(dates) else ""

        bottom_ax = vol_ax or ax
        bottom_ax.xaxis.set_major_locator(MaxNLocator(8, integer=True))
        bottom_ax.xaxis.set_major_formatter(FuncFormatter(date_label))

        if vol_ax is not None:
            ax.tick_params(labelbottom=False)
            vol_ax.bar(x, df["Volume"], width=0.8, color=colors, alpha=0.6, linewidth=0)
            vol_ax.yaxis.set_major_formatter(FuncFormatter(compact_number))
            vol_ax.set_ylabel("Volume", fontsize=9)
            vol_ax.grid(alpha=0.3)

        # Crosshair artists are "animated" so normal draws skip them; they're blitted on hover.
        cross = dict(color="gray", linewidth=0.8, linestyle="--", animated=True)
        self.cross_v = [a.axvline(len(df) - 1, **cross) for a in (ax, vol_ax) if a is not None]
        self.cross_h = ax.axhline(df["Close"].iloc[-1], **cross)
        self.info = ax.text(0.99, 0.97, "", transform=ax.transAxes, ha="right", va="top",
                            fontsize=9, family="monospace", animated=True,
                            bbox=dict(boxstyle="round", facecolor="white",
                                      edgecolor="lightgray", alpha=0.9))

        self.ax, self.vol_ax, self.background, self.drag = ax, vol_ax, None, None
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
        self.ax.set_xlim(lo, hi)               # volume axis follows through sharex

        visible = slice(max(int(np.ceil(lo)), 0), min(int(np.floor(hi)), n - 1) + 1)
        prices = np.concatenate([s[visible] for s in self.price_series])
        low, high = np.nanmin(prices), np.nanmax(prices)
        pad = (high - low) * 0.05 or abs(high) * 0.01 or 1
        self.ax.set_ylim(low - pad, high + pad)
        if self.vol_ax is not None:
            self.vol_ax.set_ylim(0, max(self.data["Volume"].iloc[visible].max(), 1) * 1.1)
        self.canvas.draw_idle()

    def _in_chart(self, event):
        return self.ax is not None and event.inaxes in (self.ax, self.vol_ax)

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
        self.info.set_text("\n".join([
            row["Date"].strftime("%a %d %b %Y"),
            f"Open   {format_price(row['Open'], currency)}",
            f"High   {format_price(row['High'], currency)}",
            f"Low    {format_price(row['Low'], currency)}",
            f"Close  {format_price(row['Close'], currency)} ({pct:+.2f}%)",
            f"Volume {compact_number(row['Volume'])}",
        ]))

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
        for child in self.stats.winfo_children():
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
            box = ttk.Frame(self.stats, padding=(0, 0, 24, 0))
            box.pack(side=tk.LEFT)
            ttk.Label(box, text=label, foreground="gray").pack(anchor="w")
            ttk.Label(box, text=value, font=("Segoe UI", 13, "bold"),
                      foreground=color or "").pack(anchor="w")


if __name__ == "__main__":
    app = StockApp()
    app.protocol("WM_DELETE_WINDOW", lambda: (app.executor.shutdown(wait=False), app.destroy()))
    app.mainloop()
