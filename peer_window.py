"""Window comparing a stock with its peer group: a rebased performance chart and a sortable table."""
import math
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

from peers import METRICS, PeerComparison
from yfinnace import format_money_compact, format_price

# Focus + context: the stock and one picked peer in colour, the rest of the group as grey context.
TARGET, PICKED, PEER, MEDIAN = "#2a78d6", "#eb6834", "#c3c2b7", "#52514e"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
UP, DOWN = "#26a69a", "#ef5350"
MEDIAN_ROW = "__median__"


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def short_name(name: str, width: int = 22) -> str:
    return name if len(name) <= width else name[: width - 1].rstrip() + "…"


class PeerWindow(tk.Toplevel):
    def __init__(self, master, comparison: PeerComparison, title: str, market: str,
                 period: str, on_open=None):
        super().__init__(master)
        self.cmp = comparison
        self.on_open = on_open     # called with (symbol, company name) on double-click
        self.picked = None         # peer highlighted in the chart, from a click on a row or line
        self.sort = ("market_cap", True)   # (column, descending)

        self.title(f"Peers · {comparison.symbol}")
        self.geometry("1150x800")
        self.minsize(900, 600)

        self._build_header(title, market, period)
        self._build_body()
        self._fill_table()
        self._draw_chart()

    # ---------- layout ----------

    def _build_header(self, title, market, period):
        c = self.cmp
        top = ttk.Frame(self, padding=(12, 12, 12, 0))
        top.pack(fill=tk.X)
        ttk.Label(top, text=f"{title} vs. peers", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(top, foreground="gray",
                  text=f"{c.group} on {market}  ·  {len(c.peers)} largest peers by market cap  ·  {period}"
                  ).pack(anchor="w")
        if c.basis == "sector":
            ttk.Label(top, foreground="gray",
                      text="Too few companies in the same industry on this exchange, so peers "
                           "are drawn from the whole sector.").pack(anchor="w")

        stats = ttk.Frame(top)
        stats.pack(fill=tk.X, pady=(10, 0))
        for label, value, note, color in self._summary():
            box = ttk.Frame(stats, padding=(0, 0, 28, 0))
            box.pack(side=tk.LEFT)
            ttk.Label(box, text=label, foreground="gray").pack(anchor="w")
            ttk.Label(box, text=value, font=("Segoe UI", 13, "bold"),
                      foreground=color or "").pack(anchor="w")
            ttk.Label(box, text=note, foreground="gray").pack(anchor="w")

    def _summary(self):
        """Headline tiles: the stock's value, where it ranks, and the peer median."""
        c, row = self.cmp, self.cmp.table.loc[self.cmp.symbol]

        def placing(column, best_word, ascending=False):
            rank = c.rank(column, ascending)
            if rank is None:
                return "not ranked"
            place, n = rank
            return f"{best_word} of {n}" if place == 1 else f"{ordinal(place)} {best_word} of {n}"

        ret = row["return"]
        tiles = [
            ("Return", self._fmt("return", ret),
             f"{placing('return', 'highest')}  ·  median {self._fmt('return', c.median('return'))}",
             None if math.isnan(ret) else UP if ret >= 0 else DOWN),
            ("Volatility", self._fmt("volatility", row["volatility"]),
             f"{placing('volatility', 'lowest', ascending=True)}  ·  "
             f"median {self._fmt('volatility', c.median('volatility'))}", None),
            ("Max drawdown", self._fmt("max_drawdown", row["max_drawdown"]),
             f"{placing('max_drawdown', 'shallowest')}  ·  "
             f"median {self._fmt('max_drawdown', c.median('max_drawdown'))}", None),
        ]
        for column in ("market_cap", "pe", "dividend_yield"):
            tiles.append((METRICS[column][0], self._fmt(column, row[column]),
                          f"median {self._fmt(column, c.median(column))}", None))
        return tiles

    def _build_body(self):
        panes = ttk.PanedWindow(self, orient=tk.VERTICAL)
        panes.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        chart = ttk.Frame(panes)
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("figure_leave_event", lambda _e: self._hide_tooltip())
        self.canvas.mpl_connect("button_press_event", self._on_chart_click)
        panes.add(chart, weight=3)

        table = ttk.Frame(panes)
        columns = list(METRICS)
        self.tree = ttk.Treeview(table, columns=columns, show="headings", selectmode="browse")
        for col in columns:
            self.tree.heading(col, text=METRICS[col][0], command=lambda c=col: self._sort_by(c))
            self.tree.column(col, width=230 if col == "name" else 95,
                             anchor="w" if col == "name" else "e", stretch=col == "name")
        scroll = ttk.Scrollbar(table, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)

        bold = tkfont.nametofont("TkDefaultFont").copy()
        bold.configure(weight="bold")
        italic = tkfont.nametofont("TkDefaultFont").copy()
        italic.configure(slant="italic")
        self._fonts = (bold, italic)   # keep references alive
        self.tree.tag_configure("target", font=bold, background="#e8f1fc")
        self.tree.tag_configure("median", font=italic, foreground="#52514e", background="#f4f4f2")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Double-1>", self._on_double_click)
        panes.add(table, weight=2)

        ttk.Label(self, foreground="gray", padding=(12, 0, 12, 8),
                  text="Click a row or line to highlight it  ·  double-click a row to open its chart  "
                       "·  click a column heading to sort").pack(anchor="w")

    # ---------- table ----------

    def _fmt(self, column, value):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "—"
        currency = self.cmp.currency
        return {
            "name": lambda v: v,
            "price": lambda v: format_price(v, currency),
            "return": lambda v: f"{v * 100:+.1f}%",
            "volatility": lambda v: f"{v * 100:.1f}%",
            "max_drawdown": lambda v: f"{v * 100:.1f}%",
            "market_cap": lambda v: format_money_compact(v, currency),
            "pe": lambda v: f"{v:.1f}",
            "pb": lambda v: f"{v:.2f}",
            "dividend_yield": lambda v: f"{v:.2f}%",
        }[column](value)

    def _fill_table(self):
        self.tree.delete(*self.tree.get_children())
        column, descending = self.sort
        table = self.cmp.table
        key = table[column].str.lower() if column == "name" else table[column]
        order = key.sort_values(ascending=not descending, na_position="last").index
        for symbol in order:
            row = table.loc[symbol]
            self.tree.insert("", tk.END, iid=symbol,
                             values=[self._fmt(c, row[c]) for c in METRICS],
                             tags=("target",) if symbol == self.cmp.symbol else ())
        # The median row always sits at the bottom, whatever the sort.
        medians = ["Peer median"] + [self._fmt(c, self.cmp.median(c)) for c in list(METRICS)[1:]]
        self.tree.insert("", tk.END, iid=MEDIAN_ROW, values=medians, tags=("median",))

        for col in METRICS:
            arrow = (" ▼" if descending else " ▲") if col == column else ""
            self.tree.heading(col, text=METRICS[col][0] + arrow)
        if self.picked:
            self.tree.selection_set(self.picked)

    def _sort_by(self, column):
        current, descending = self.sort
        # A new column starts with the best (or largest) first; clicking again flips it.
        self.sort = (column, not descending) if column == current else (
            column, METRICS[column][1] is not False and column != "name")
        self._fill_table()

    def _on_select(self, _event):
        selection = self.tree.selection()
        symbol = selection[0] if selection else None
        picked = symbol if symbol in self.cmp.peers else None
        if picked != self.picked:
            self.picked = picked
            self._draw_chart()

    def _on_double_click(self, event):
        symbol = self.tree.identify_row(event.y)
        if symbol and symbol != MEDIAN_ROW and self.on_open:
            self.on_open(symbol, self.cmp.table.loc[symbol, "name"])

    # ---------- chart ----------

    def _draw_chart(self):
        c = self.cmp
        perf = c.performance
        x = np.arange(len(perf))
        names = c.table["name"]

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.axhline(100, color=MUTED, linewidth=0.8)
        for symbol in c.peers:
            if symbol != self.picked:
                ax.plot(x, perf[symbol], color=PEER, linewidth=1, zorder=1)
        ax.plot(x, c.peer_median, color=MEDIAN, linewidth=1.5, linestyle="--", zorder=2)
        if self.picked:
            ax.plot(x, perf[self.picked], color=PICKED, linewidth=2, zorder=3)
        ax.plot(x, perf[c.symbol], color=TARGET, linewidth=2, zorder=4)

        # Direct labels at the right edge, nudged apart so they don't overlap.
        ends = [(perf[c.symbol].iloc[-1], short_name(names[c.symbol]), TARGET),
                (c.peer_median.iloc[-1], "Peer median", MEDIAN)]
        if self.picked:
            ends.append((perf[self.picked].iloc[-1], short_name(names[self.picked]), PICKED))
        lo, hi = np.nanmin(perf.to_numpy()), np.nanmax(perf.to_numpy())
        gap, last_y = (hi - lo) * 0.05, -np.inf
        for y, label, color in sorted(ends):
            last_y = max(y, last_y + gap)   # the dot stays on the line; only the text moves
            ax.plot(x[-1], y, "o", color=color, markersize=5, zorder=5)
            ax.annotate(f"{label}  {y - 100:+.1f}%", (x[-1], last_y), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=9, color=INK,
                        annotation_clip=False)

        legend = [Line2D([], [], color=TARGET, linewidth=2, label=short_name(names[c.symbol], 30)),
                  Line2D([], [], color=MEDIAN, linewidth=1.5, linestyle="--", label="Peer median"),
                  Line2D([], [], color=PEER, linewidth=1, label="Other peers")]
        if self.picked:
            legend.insert(1, Line2D([], [], color=PICKED, linewidth=2,
                                    label=short_name(names[self.picked], 30)))
        ax.legend(handles=legend, loc="lower left", bbox_to_anchor=(0, 1), ncol=len(legend),
                  fontsize=9, frameon=False, borderaxespad=0.2)

        ax.set_title("Growth of 100 over the period", loc="left", fontsize=11, color=INK, pad=26)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelcolor="#52514e", labelsize=9)

        dates = perf.index

        def date_label(value, _pos):
            i = int(round(value))
            return dates[i].strftime("%d %b %Y") if 0 <= i < len(dates) else ""

        ax.xaxis.set_major_locator(MaxNLocator(7, integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(date_label))
        ax.set_xlim(-1, len(perf))

        self.tooltip = ax.text(0.99, 0.03, "", transform=ax.transAxes, ha="right", va="bottom",
                               fontsize=9, family="monospace", visible=False, zorder=10,
                               bbox=dict(boxstyle="round", facecolor="white",
                                         edgecolor="lightgray", alpha=0.95))
        self.hover_dot, = ax.plot([], [], "o", markersize=7, markeredgecolor="white",
                                  markeredgewidth=1.5, zorder=6, visible=False)
        self.ax = ax
        self.figure.subplots_adjust(left=0.06, right=0.8, top=0.86, bottom=0.1)
        self.canvas.draw_idle()

    def _nearest(self, event):
        """(row index, symbol) of the line closest to the cursor, or None outside the plot."""
        if event.inaxes is not self.ax or event.xdata is None:
            return None
        perf = self.cmp.performance
        i = int(np.clip(round(event.xdata), 0, len(perf) - 1))
        values = perf.iloc[i].dropna()
        if values.empty:
            return None
        return i, (values - event.ydata).abs().idxmin()

    def _on_motion(self, event):
        hit = self._nearest(event)
        if hit is None:
            self._hide_tooltip()
            return
        i, symbol = hit
        c = self.cmp
        perf = c.performance
        color = TARGET if symbol == c.symbol else PICKED if symbol == self.picked else MEDIAN

        lines = [perf.index[i].strftime("%a %d %b %Y"),
                 f"{short_name(c.table.loc[symbol, 'name'], 24):<24} {perf[symbol].iloc[i] - 100:+7.1f}%"]
        if symbol != c.symbol:
            lines.append(f"{short_name(c.table.loc[c.symbol, 'name'], 24):<24} "
                         f"{perf[c.symbol].iloc[i] - 100:+7.1f}%")
        lines.append(f"{'Peer median':<24} {c.peer_median.iloc[i] - 100:+7.1f}%")
        self.tooltip.set_text("\n".join(lines))
        self.tooltip.set_visible(True)
        self.hover_dot.set_data([i], [perf[symbol].iloc[i]])
        self.hover_dot.set_color(color)
        self.hover_dot.set_visible(True)
        self.canvas.draw_idle()

    def _hide_tooltip(self):
        if getattr(self, "tooltip", None) is not None and self.tooltip.get_visible():
            self.tooltip.set_visible(False)
            self.hover_dot.set_visible(False)
            self.canvas.draw_idle()

    def _on_chart_click(self, event):
        hit = self._nearest(event)
        if hit is None:
            return
        symbol = hit[1]
        if symbol in self.cmp.peers:
            self.tree.selection_set(symbol)   # fires _on_select, which redraws
            self.tree.see(symbol)
