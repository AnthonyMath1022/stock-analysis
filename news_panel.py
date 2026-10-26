"""Side panel where the news agent talks: FinBERT's read of a stock's recent headlines, a local
LLM's analysis of them, and its answers to follow-up questions about the same headlines."""
import queue
import re
import tkinter as tk
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from tkinter import ttk

UP, DOWN, MUTED, ASK = "#26a69a", "#ef5350", "#898781", "#2a78d6"
SENTIMENT_COLORS = {"positive": UP, "negative": DOWN, "neutral": MUTED}
WIDTH = 380   # starting width in pixels; the divider beside the chart drags


class NewsPanel(ttk.Frame):
    def __init__(self, master, on_close):
        super().__init__(master, padding=(8, 12, 12, 12))
        self.analysis = None   # agent.NewsAnalysis for the stock shown, once FinBERT has scored it
        self.busy = False
        # The models are slow and use the GPU, so they get their own thread, one request at a
        # time, and never hold up chart downloads.
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.calls = queue.Queue()   # (function, args) the worker wants run on the Tk thread
        self.question = tk.StringVar()
        self.status = tk.StringVar()
        self._build(on_close)

    # ---------- layout ----------

    def _build(self, on_close):
        header = ttk.Frame(self)
        header.pack(fill=tk.X)
        ttk.Label(header, text="News analyst", font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT)
        ttk.Button(header, text="✕", width=3, command=on_close).pack(side=tk.RIGHT)
        self.subject = ttk.Label(self, foreground="gray",
                                 text="FinBERT sentiment and a local LLM's read of the news")
        self.subject.pack(anchor="w")

        scores = ttk.Frame(self)
        scores.pack(fill=tk.X, pady=(8, 6))
        self.sentiment = ttk.Label(scores, font=("Segoe UI", 13, "bold"))
        self.sentiment.pack(anchor="w")
        self.breakdown = ttk.Label(scores, foreground="gray")
        self.breakdown.pack(anchor="w")

        # Packed before the tabs so it keeps its place at the bottom however short the panel is.
        ask = ttk.Frame(self)
        ask.pack(side=tk.BOTTOM, fill=tk.X, pady=(6, 0))
        ttk.Label(ask, textvariable=self.status, foreground="gray", wraplength=WIDTH - 20
                  ).pack(anchor="w", pady=(0, 4))
        self.ask_button = ttk.Button(ask, text="Ask", command=self.ask, state="disabled")
        self.ask_button.pack(side=tk.RIGHT, padx=(4, 0))
        entry = ttk.Entry(ask, textvariable=self.question)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)
        entry.bind("<Return>", lambda _e: self.ask())

        tabs = ttk.Notebook(self)
        tabs.pack(fill=tk.BOTH, expand=True)
        tabs.add(self._build_chat(tabs), text="Analysis")
        tabs.add(self._build_headlines(tabs), text="Headlines")

    def _build_chat(self, parent):
        frame = ttk.Frame(parent)
        self.chat = tk.Text(frame, wrap="word", width=40, height=10, relief="flat",
                            font=("Segoe UI", 10), padx=8, pady=6, state="disabled",
                            cursor="arrow", background="white")
        scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.chat.yview)
        self.chat.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat.pack(fill=tk.BOTH, expand=True)
        # A disabled Text doesn't take focus on click, which would stop Ctrl+C copying.
        self.chat.bind("<1>", lambda _e: self.chat.focus_set())

        self.chat.tag_configure("speaker", font=("Segoe UI", 9, "bold"), foreground=MUTED,
                                spacing1=10)
        self.chat.tag_configure("question", foreground=ASK)
        self.chat.tag_configure("bold", font=("Segoe UI", 10, "bold"))
        self.chat.tag_configure("bullet", lmargin1=6, lmargin2=20)
        self.chat.tag_configure("muted", foreground=MUTED)
        self.chat.tag_configure("error", foreground=DOWN)
        return frame

    def _build_headlines(self, parent):
        frame = ttk.Frame(parent)
        # Headlines are cut off in the table, so the picked one is shown here in full.
        self.picked = ttk.Label(frame, text="Double-click a headline to open the article.",
                                foreground="gray", wraplength=WIDTH - 30)
        self.picked.pack(side=tk.BOTTOM, anchor="w", pady=(4, 0))
        self.tree = ttk.Treeview(frame, columns=("date", "score", "title"), show="headings",
                                 height=10, selectmode="browse")
        for col, heading, width, anchor in (("date", "Date", 56, "w"), ("score", "Score", 50, "e"),
                                            ("title", "Headline", 260, "w")):
            self.tree.heading(col, text=heading, anchor=anchor)
            self.tree.column(col, width=width, anchor=anchor, stretch=col == "title")
        for label, color in SENTIMENT_COLORS.items():
            if label != "neutral":
                self.tree.tag_configure(label, foreground=color)
        scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Double-1>", self._open_article)
        return frame

    # ---------- the agent ----------

    def analyze(self, symbol, name=None):
        """Start over with a stock: fetch and score its news, then stream the LLM's analysis."""
        if self.busy:
            self.status.set("Still working; try again once this answer is finished.")
            return
        self.analysis = None
        self._clear()
        self.subject.config(text=f"{name or symbol} · {symbol}")
        self.status.set("Fetching news and scoring it with FinBERT… "
                        "(the first run also downloads the models, about 4.4 GB)")

        def work():
            import agent   # deferred: loading torch and transformers takes a few seconds

            analysis = agent.analyze_news(symbol, name, summarize=False)
            self._later(self._show_scores, analysis)
            self._later(self._start_reply)
            return analysis.summarize(on_text=lambda text: self._later(self._stream, text))

        self._run(work)

    def ask(self):
        """Send the typed question to the agent, which answers from the same headlines."""
        question = self.question.get().strip()
        if not question or self.analysis is None or self.busy:
            return
        self.question.set("")
        self._write("You\n", "speaker")
        self._write(question + "\n", "question")
        self._start_reply()
        analysis = self.analysis
        self._run(lambda: analysis.ask(question,
                                       on_text=lambda text: self._later(self._stream, text)))

    def _run(self, work):
        """Run work() on the agent's thread, then show the reply it returns."""
        self.busy = True
        self.ask_button.state(["disabled"])
        self._poll(self.executor.submit(work))

    def _later(self, fn, *args):
        """From the worker thread: have fn(*args) run on the Tk thread."""
        self.calls.put((fn, args))

    def _poll(self, future):
        # Tk widgets must only be touched from the main thread, so the worker queues its updates
        # and this drains them.
        while not self.calls.empty():
            fn, args = self.calls.get()
            fn(*args)
        if not future.done():
            self.after(50, self._poll, future)
            return
        self.busy = False
        if self.analysis is not None:
            self.ask_button.state(["!disabled"])
        try:
            reply = future.result()
        except Exception as e:
            self.status.set("")
            self._write(f"\n{e}\n", "error")
            return
        self._finish_reply(reply)

    # ---------- showing results ----------

    def _clear(self):
        self.chat.configure(state="normal")
        self.chat.delete("1.0", tk.END)
        self.chat.configure(state="disabled")
        self.tree.delete(*self.tree.get_children())
        self.sentiment.config(text="")
        self.breakdown.config(text="")

    def _show_scores(self, analysis):
        self.analysis = analysis
        counts = analysis.counts
        self.sentiment.config(text=f"{analysis.sentiment.capitalize()} news  "
                                   f"{analysis.score:+.2f}",
                              foreground=SENTIMENT_COLORS[analysis.sentiment])
        self.breakdown.config(text=f"{counts['positive']} positive · {counts['negative']} negative"
                                   f" · {counts['neutral']} neutral headlines (FinBERT)")
        for i, row in analysis.headlines.iterrows():
            self.tree.insert("", tk.END, iid=str(i), tags=(row["sentiment"],),
                             values=(f"{row['date']:%d %b}", f"{row['score']:+.2f}",
                                     f"{row['title']} — {row['source']}"))

    def _start_reply(self):
        self._write("Analyst\n", "speaker")
        # The reply streams in as plain text after this mark, then is redrawn with formatting.
        self.chat.mark_set("reply", "end-1c")
        self.chat.mark_gravity("reply", tk.LEFT)
        self.status.set("Thinking…")

    def _stream(self, text):
        if self.chat.compare("reply", "==", "end-1c"):   # nothing written yet
            text = text.lstrip()
        self._write(text)

    def _finish_reply(self, reply):
        self.chat.configure(state="normal")
        self.chat.delete("reply", "end-1c")
        self._insert_markdown(reply)
        self.chat.configure(state="disabled")
        self.chat.see(tk.END)
        self.status.set("Ask a follow-up question about this news below.")

    def _insert_markdown(self, text):
        """The small part of Markdown the model uses: **bold**, # headings and - bullets."""
        for line in text.strip().splitlines():
            line = line.rstrip()
            heading = re.fullmatch(r"#+\s*(.*)", line)
            if heading:
                line = f"**{heading.group(1)}**"
            bullet = re.fullmatch(r"\s*[-*•]\s+(.*)", line)
            tags = ("bullet",) if bullet else ()
            if bullet:
                line = "•  " + bullet.group(1)
            # re.split with a group puts the bold parts at the odd positions.
            for i, part in enumerate(re.split(r"\*\*(.+?)\*\*", line)):
                if part:
                    self.chat.insert(tk.END, part, tags + (("bold",) if i % 2 else ()))
            self.chat.insert(tk.END, "\n", tags)

    def _write(self, text, *tags):
        self.chat.configure(state="normal")
        self.chat.insert(tk.END, text, tags)
        self.chat.configure(state="disabled")
        self.chat.see(tk.END)

    def _selected_headline(self):
        selected = self.tree.selection()
        if selected and self.analysis is not None:
            return self.analysis.headlines.loc[int(selected[0])]
        return None

    def _on_select(self, _event):
        row = self._selected_headline()
        if row is not None:
            self.picked.config(text=f"{row['title']} — {row['source']}, {row['date']:%d %b %Y}. "
                                    "Double-click to open.")

    def _open_article(self, _event):
        row = self._selected_headline()
        if row is not None:
            webbrowser.open(row["link"])
