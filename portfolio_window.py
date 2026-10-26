"""Window for building a portfolio: list the holdings, then see the Markowitz efficient frontier,
the optimal weights and how each portfolio would have done over the period."""
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from tkinter import messagebox, ttk

import numpy as np
from matplotlib import patheffects
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from evaluation import PORTFOLIOS, Holding, PortfolioAnalysis, analyse_portfolio, parse_rate
from peer_window import short_name
from yfinnace import format_percent, format_ratio

# Categorical slots 1-3 for the frontier and the two optimised portfolios; the rest is ink and grey.
FRONTIER, MAX_SHARPE, MIN_VARIANCE = "#2a78d6", "#eb6834", "#1baf7a"
EQUAL, STOCK, CLOUD = "#0b0b0b", "#52514e", "#c3c2b7"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
HALO = [patheffects.withStroke(linewidth=3, foreground="white")]   # keeps labels legible on dots

PERIODS = {"1Y": 365, "3Y": 1096, "5Y": 1826, "10Y": 3652}
# portfolio -> (matplotlib marker, colour, marker size, glyph tying its table row to the chart)
MARKERS = {"Max Sharpe": ("*", MAX_SHARPE, 17, "★"),
           "Min variance": ("D", MIN_VARIANCE, 9, "◆"),
           "Equal weight": ("s", EQUAL, 8, "■")}
LABEL_STOCKS = 10    # name every stock on the chart up to this many; past that, hover to identify
HIT_RADIUS = 14      # pixels from a point that still count as hovering it

# column -> (heading, formatter) for the portfolio table
PORTFOLIO_COLUMNS = {
    "annual_return": ("Return p.a.", lambda v: format_percent(v, signed=True)),
    "volatility": ("Volatility", format_percent),
    "sharpe": ("Sharpe", format_ratio),
    "sortino": ("Sortino", format_ratio),
    "max_drawdown": ("Max drawdown", format_percent),
    "var": ("VaR 95% · 1 day", lambda v: format_percent(v, 2)),
    "cvar": ("CVaR 95% · 1 day", lambda v: format_percent(v, 2)),
}
HOLDING_COLUMNS = {
    "name": "Company",
    "symbol": "Symbol",
    "annual_return": "Return p.a.",
    "volatility": "Volatility",
    "sharpe": "Sharpe",
    "Max Sharpe": "Max-Sharpe weight",
    "Min variance": "Min-variance weight",
}


def chart_label(name: str) -> str:
    """Bursa company names read 'SHORT · Long name'; the short part is enough on the chart."""
    return short_name(name.split(" · ")[0], 18)


class PortfolioWindow(tk.Toplevel):
    def __init__(self, master, risk_free_rate: str):
        super().__init__(master)
        self.title("Portfolio")
        self.geometry("1250x820")
        self.minsize(1000, 680)

        self.executor = ThreadPoolExecutor(max_workers=1)
        self.poll = None                 # pending after() id while optimising
        self.holdings: list[Holding] = []
        self.analysis: PortfolioAnalysis | None = None
        self.selected = None             # symbol or portfolio picked in a table, kept highlighted
        self.highlighted = None          # point the chart currently rings
        self.ax = None
        self.points = []                 # (volatility, return, key) of every named chart point

        self.period = tk.StringVar(value="3Y")
        self.start = tk.StringVar()
        self.end = tk.StringVar()
        self.risk_free = tk.StringVar(value=risk_free_rate)
        self.status = tk.StringVar()
        self.subtitle = tk.StringVar()
        self.note = tk.StringVar()

        self._build_controls()
        self._build_results()
        self._on_period_change()
        self._holdings_changed()
        self.protocol("WM_DELETE_WINDOW", self._close)

    # ---------- layout ----------

    def _build_controls(self):
        panel = ttk.Frame(self, padding=12)
        panel.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(panel, text="Period").pack(anchor="w")
        periods = ttk.Frame(panel)
        periods.pack(fill=tk.X, pady=(0, 6))
        for i, name in enumerate(PERIODS):
            ttk.Radiobutton(periods, text=name, value=name, variable=self.period,
                            style="Toolbutton", command=self._on_period_change,
                            ).grid(row=0, column=i, sticky="ew", padx=1)
            periods.columnconfigure(i, weight=1)

        dates = ttk.Frame(panel)
        dates.pack(fill=tk.X, pady=(0, 10))
        for row, (label, var) in enumerate((("Start", self.start), ("End", self.end))):
            ttk.Label(dates, text=label, width=6).grid(row=row, column=0, sticky="w")
            entry = ttk.Entry(dates, textvariable=var, width=12)
            entry.grid(row=row, column=1, sticky="ew", pady=1)
            entry.bind("<Key>", lambda _e: self.period.set(""))  # typing a date means custom
        ttk.Label(dates, text="YYYY-MM-DD", foreground="gray").grid(row=0, column=2, padx=4)
        dates.columnconfigure(1, weight=1)

        rate = ttk.Frame(panel)
        rate.pack(fill=tk.X)
        ttk.Label(rate, text="Risk-free rate").pack(side=tk.LEFT)
        ttk.Label(rate, text="% a year", foreground="gray").pack(side=tk.RIGHT)
        rate_entry = ttk.Entry(rate, textvariable=self.risk_free, width=6, justify="right")
        rate_entry.pack(side=tk.RIGHT, padx=4)
        rate_entry.bind("<Return>", lambda _e: self.optimise())

        self.optimise_button = ttk.Button(panel, text="Optimise", command=self.optimise)
        self.optimise_button.pack(fill=tk.X, pady=(16, 6), ipady=4)
        ttk.Label(panel, textvariable=self.status, wraplength=230, foreground="gray"
                  ).pack(anchor="w")

        ttk.Label(panel, wraplength=230, foreground="gray", justify="left",
                  text="Add stocks from the main window with “Add to portfolio”.\n\n"
                       "Weights are long-only (no short selling) and add up to 100%. "
                       "Max Sharpe has the best return per unit of risk; Min variance has the "
                       "lowest volatility."
                  ).pack(side=tk.BOTTOM, anchor="w")

    def _build_results(self):
        right = ttk.Frame(self, padding=(0, 12, 12, 12))
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(right, text="Portfolio", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(right, textvariable=self.subtitle, foreground="gray").pack(anchor="w")
        ttk.Label(right, textvariable=self.note, foreground="gray").pack(anchor="w")

        columns = ["name", *PORTFOLIO_COLUMNS]
        self.portfolio_tree = ttk.Treeview(right, columns=columns, show="headings", height=3,
                                           selectmode="browse")
        self.portfolio_tree.heading("name", text="Portfolio", anchor="w")
        self.portfolio_tree.column("name", width=150, anchor="w")
        for col, (heading, _fmt) in PORTFOLIO_COLUMNS.items():
            self.portfolio_tree.heading(col, text=heading, anchor="e")
            self.portfolio_tree.column(col, width=105, anchor="e")
        self.portfolio_tree.pack(fill=tk.X, pady=(8, 4))
        self.portfolio_tree.bind("<<TreeviewSelect>>",
                                 lambda _e: self._on_select(self.portfolio_tree))

        ttk.Label(right, foreground="gray",
                  text="Weights are fitted to this period's prices: they show what would have "
                       "worked, not a forecast. No trading costs or rebalancing are modelled."
                  ).pack(side=tk.BOTTOM, anchor="w")

        panes = ttk.PanedWindow(right, orient=tk.VERTICAL)
        panes.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        chart = ttk.Frame(panes)
        self.figure = Figure(figsize=(8, 4.5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("figure_leave_event", lambda _e: self._highlight(self.selected))
        panes.add(chart, weight=3)

        table = ttk.Frame(panes)
        buttons = ttk.Frame(table)
        buttons.pack(fill=tk.X, pady=(6, 4))
        ttk.Label(buttons, text="Holdings", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Clear", command=self.clear).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="Remove selected", command=self.remove_selected
                   ).pack(side=tk.RIGHT, padx=4)

        self.holdings_tree = ttk.Treeview(table, columns=list(HOLDING_COLUMNS), show="headings",
                                          height=6, selectmode="extended")
        for col, heading in HOLDING_COLUMNS.items():
            left = col in ("name", "symbol")
            self.holdings_tree.heading(col, text=heading, anchor="w" if left else "e")
            self.holdings_tree.column(col, width=230 if col == "name" else 110,
                                      anchor="w" if left else "e", stretch=col == "name")
        scroll = ttk.Scrollbar(table, orient=tk.VERTICAL, command=self.holdings_tree.yview)
        self.holdings_tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.holdings_tree.pack(fill=tk.BOTH, expand=True)
        self.holdings_tree.bind("<<TreeviewSelect>>",
                                lambda _e: self._on_select(self.holdings_tree))
        self.holdings_tree.bind("<Delete>", lambda _e: self.remove_selected())
        panes.add(table, weight=2)

    # ---------- holdings ----------

    def add(self, holding: Holding) -> bool:
        """Add a holding; False if it's already in the portfolio."""
        if any(h.symbol == holding.symbol for h in self.holdings):
            return False
        self.holdings.append(holding)
        self._holdings_changed()
        return True

    def remove_selected(self):
        picked = set(self.holdings_tree.selection())
        if picked:
            self.holdings = [h for h in self.holdings if h.symbol not in picked]
            self._holdings_changed()

    def clear(self):
        self.holdings = []
        self._holdings_changed()

    def _holdings_changed(self):
        """Old results no longer match the holdings, so drop them until the next Optimise."""
        self.analysis = None
        self.selected = None
        n = len(self.holdings)
        self.status.set("Add at least two stocks." if n < 2 else
                        f"{n} stocks. Press Optimise to find the best weights.")
        self.subtitle.set(f"{n} stock{'s' if n != 1 else ''}")
        self.note.set("")
        self._fill_tables()
        self._draw_chart()

    # ---------- optimising ----------

    def _on_period_change(self):
        days = PERIODS.get(self.period.get())
        if days is not None:
            today = date.today()
            self.start.set((today - timedelta(days=days)).isoformat())
            self.end.set(today.isoformat())

    def optimise(self):
        if self.poll is not None:     # already running
            return
        if len(self.holdings) < 2:
            messagebox.showinfo("Portfolio", "Add at least two stocks from the main window with "
                                "“Add to portfolio” first.", parent=self)
            return
        try:
            start, end = date.fromisoformat(self.start.get()), date.fromisoformat(self.end.get())
            if start >= end:
                raise ValueError("Start date must be before end date.")
            rate = parse_rate(self.risk_free.get())
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e), parent=self)
            return

        holdings = list(self.holdings)
        self.optimise_button.state(["disabled"])
        self.status.set(f"Downloading prices for {len(holdings)} stocks and optimising…")
        future = self.executor.submit(analyse_portfolio, holdings, start, end, rate)
        self._wait_for(future, holdings)

    def _wait_for(self, future, holdings):
        # Tk widgets must only be touched from the main thread, so poll instead of using a callback.
        if not future.done():
            self.poll = self.after(100, self._wait_for, future, holdings)
            return
        self.poll = None
        self.optimise_button.state(["!disabled"])
        try:
            analysis = future.result()
        except Exception as e:
            self.status.set("Optimisation failed.")
            messagebox.showerror("Could not build portfolio", str(e), parent=self)
            return
        if holdings != self.holdings:
            self.status.set("The holdings changed while optimising. Press Optimise again.")
            return
        self._show(analysis)

    def _show(self, a: PortfolioAnalysis):
        self.analysis = a
        self.subtitle.set(f"{len(a.holdings)} stocks  ·  {a.days} trading days, "
                          f"{a.first_day:%d %b %Y} – {a.last_day:%d %b %Y}  ·  "
                          f"risk-free rate {format_percent(a.risk_free_rate)}")
        notes = []
        if a.missing:
            notes.append(f"No prices for {', '.join(h.symbol for h in a.missing)} in this "
                         "period, so left out.")
        if len({h.market for h in a.holdings}) > 1:
            notes.append("Returns are measured in each stock's own currency.")
        self.note.set("  ".join(notes))
        best = a.portfolios.loc["Max Sharpe"]
        self.status.set(f"Best Sharpe ratio {format_ratio(best.sharpe)}: "
                        f"{format_percent(best.annual_return, signed=True)} a year at "
                        f"{format_percent(best.volatility)} volatility.")
        self._fill_tables()
        self._draw_chart()

    def _close(self):
        if self.poll is not None:
            self.after_cancel(self.poll)
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.destroy()

    # ---------- tables ----------

    def _fill_tables(self):
        a = self.analysis
        self.portfolio_tree.delete(*self.portfolio_tree.get_children())
        if a is not None:
            for name in PORTFOLIOS:
                row = a.portfolios.loc[name]
                self.portfolio_tree.insert("", tk.END, iid=name, values=[
                    f"{MARKERS[name][3]}  {name}",
                    *[fmt(row[col]) for col, (_h, fmt) in PORTFOLIO_COLUMNS.items()]])

        self.holdings_tree.delete(*self.holdings_tree.get_children())
        for h in self.holdings:
            values = [h.name, h.symbol] + ["—"] * (len(HOLDING_COLUMNS) - 2)
            if a is not None and h.symbol in a.assets.index:
                row, weights = a.assets.loc[h.symbol], a.weights.loc[h.symbol]
                values[2:] = [format_percent(row.annual_return, signed=True),
                              format_percent(row.volatility), format_ratio(row.sharpe),
                              format_percent(weights["Max Sharpe"]),
                              format_percent(weights["Min variance"])]
            self.holdings_tree.insert("", tk.END, iid=h.symbol, values=values)

    def _on_select(self, tree):
        selection = tree.selection()
        if not selection:
            return
        # One highlight at a time: picking in one table clears the other.
        other = self.holdings_tree if tree is self.portfolio_tree else self.portfolio_tree
        other.selection_set(())
        self.selected = selection[0] if any(p[2] == selection[0] for p in self.points) else None
        self._highlight(self.selected)

    # ---------- chart ----------

    def _draw_chart(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.ax, self.points, self.highlighted = ax, [], None
        a = self.analysis
        if a is None:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "Add at least two stocks, then press Optimise."
                    if len(self.holdings) < 2 else "Press Optimise to find the best weights.",
                    ha="center", va="center", fontsize=11, color=MUTED, transform=ax.transAxes)
            self.canvas.draw_idle()
            return

        sim, frontier, assets, ports = a.simulated, a.frontier, a.assets, a.portfolios
        ax.scatter(sim["volatility"], sim["return"], s=5, color=CLOUD, alpha=0.5,
                   linewidths=0, zorder=1, rasterized=True)
        ax.plot(frontier["volatility"], frontier["return"], color=FRONTIER, linewidth=2,
                solid_capstyle="round", zorder=3)

        names = {h.symbol: h.name for h in a.holdings}
        ax.scatter(assets["volatility"], assets["annual_return"], s=40, color=STOCK,
                   edgecolors="white", linewidths=1.5, zorder=4)
        for symbol, row in assets.iterrows():
            self.points.append((row.volatility, row.annual_return, symbol))
            if len(assets) <= LABEL_STOCKS:
                ax.annotate(chart_label(names[symbol]), (row.volatility, row.annual_return),
                            xytext=(7, 3), textcoords="offset points", fontsize=8.5,
                            color=STOCK, path_effects=HALO, zorder=4)

        # Optimised portfolios sit on the frontier, whose upper left is always empty: label there.
        offsets = {"Max Sharpe": ((-10, 7), "right"), "Min variance": ((-10, 7), "right"),
                   "Equal weight": ((9, -12), "left")}
        for name in PORTFOLIOS:
            marker, color, size, _glyph = MARKERS[name]
            row = ports.loc[name]
            ax.plot(row.volatility, row.annual_return, marker=marker, color=color,
                    markersize=size, markeredgecolor="white", markeredgewidth=1.5,
                    linestyle="none", zorder=5)
            xytext, ha = offsets[name]
            ax.annotate(name, (row.volatility, row.annual_return), xytext=xytext, ha=ha,
                        textcoords="offset points", fontsize=9, fontweight="bold", color=INK,
                        path_effects=HALO, zorder=5)
            self.points.append((row.volatility, row.annual_return, name))

        xs = np.concatenate([sim["volatility"], frontier["volatility"], assets["volatility"]])
        ys = np.concatenate([sim["return"], frontier["return"], assets["annual_return"]])
        xpad, ypad = np.ptp(xs) * 0.1 or 0.01, np.ptp(ys) * 0.1 or 0.01
        # Extra room on the left for the "Min variance" label beside the leftmost point.
        xlim, ylim = (xs.min() - 2 * xpad, xs.max() + xpad), (ys.min() - ypad, ys.max() + ypad)

        # Capital market line: mixes of the risk-free asset and the max-Sharpe portfolio.
        sharpe = ports.loc["Max Sharpe", "sharpe"]
        ax.plot(xlim, [a.risk_free_rate + sharpe * x for x in xlim], color=MUTED, linewidth=1,
                linestyle="--", zorder=2)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        legend = [
            Line2D([], [], color=FRONTIER, linewidth=2, label="Efficient frontier"),
            Line2D([], [], color=MUTED, linewidth=1, linestyle="--", label="Capital market line"),
            *[Line2D([], [], marker=MARKERS[p][0], color=MARKERS[p][1], linestyle="none",
                     markersize=MARKERS[p][2] * 0.75, label=p) for p in PORTFOLIOS],
            Line2D([], [], marker="o", color=STOCK, linestyle="none", markersize=6,
                   label="Single stock"),
            Line2D([], [], marker="o", color=CLOUD, linestyle="none", markersize=5,
                   label="Random mix"),
        ]
        ax.legend(handles=legend, loc="lower left", bbox_to_anchor=(0, 1), ncol=4, fontsize=9,
                  frameon=False, borderaxespad=0.3, columnspacing=1.6)

        ax.set_xlabel("Volatility (annualised)", fontsize=9, color="#52514e")
        ax.set_ylabel("Return (annualised)", fontsize=9, color="#52514e")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelcolor="#52514e", labelsize=9)

        self.ring, = ax.plot([], [], "o", markersize=20, markerfacecolor="none",
                             markeredgecolor=INK, markeredgewidth=1.2, zorder=6, visible=False)
        self.tooltip = ax.text(0.02, 0.97, "", transform=ax.transAxes, ha="left", va="top",
                               fontsize=9, family="monospace", visible=False, zorder=10,
                               bbox=dict(boxstyle="round", facecolor="white",
                                         edgecolor="lightgray", alpha=0.95))
        self.figure.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.11)
        self._highlight(self.selected)
        self.canvas.draw_idle()

    def _tooltip_lines(self, key):
        a = self.analysis
        if key in PORTFOLIOS:
            title, row = key, a.portfolios.loc[key]
            weights = a.weights[key].sort_values(ascending=False)
            extra = ["Largest weights"] + [f"  {s:<15}{format_percent(w):>7}"
                                           for s, w in weights[weights >= 0.005].head(3).items()]
        else:
            name = next(h.name for h in a.holdings if h.symbol == key)
            title, row = f"{short_name(name, 24)} · {key}", a.assets.loc[key]
            extra = [f"{'Max-Sharpe weight':<17}{format_percent(a.weights.loc[key, 'Max Sharpe']):>7}"]
        return [title,
                f"{'Return p.a.':<17}{format_percent(row.annual_return, signed=True):>7}",
                f"{'Volatility':<17}{format_percent(row.volatility):>7}",
                f"{'Sharpe':<17}{format_ratio(row.sharpe):>7}",
                *extra]

    def _highlight(self, key):
        """Ring a chart point and show its details; None clears."""
        if self.analysis is None or key == self.highlighted:
            return
        self.highlighted = key
        point = next((p for p in self.points if p[2] == key), None)
        if point is None:
            self.ring.set_visible(False)
            self.tooltip.set_visible(False)
        else:
            self.ring.set_data([point[0]], [point[1]])
            self.ring.set_visible(True)
            self.tooltip.set_text("\n".join(self._tooltip_lines(key)))
            self.tooltip.set_visible(True)
        self.canvas.draw_idle()

    def _on_motion(self, event):
        if self.analysis is None or event.inaxes is not self.ax or not self.points:
            self._highlight(self.selected)
            return
        xy = self.ax.transData.transform([(p[0], p[1]) for p in self.points])
        distance = np.hypot(xy[:, 0] - event.x, xy[:, 1] - event.y)
        nearest = int(distance.argmin())
        self._highlight(self.points[nearest][2] if distance[nearest] <= HIT_RADIUS
                        else self.selected)
