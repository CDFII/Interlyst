"""

"""
# Imports
import numpy, tkinter

from tkinter import filedialog, messagebox, ttk

from data.generate import random_scenario
from interface.toolbox import METHODS, Recommender, Recommendation, design_summary, signal_state
from interface.renderer import build_scenario, build_timeline, render_png
from sim.baselines import fixed_design
from sim.delay_model import APPROACHES, TURNS, VALID_MOVES, Scenario

# Config
APP_NAME = "Interlyst"
CANVAS_SIZE = 560
TIMELINE_HEIGHT = 58

SCORE_COLORS = {
    "A":    "#1f9d3a",
    "B":    "#3fae3a",
    "C":    "#a3b320",
    "D":    "#e08a00",
    "E":    "#d9531e",
    "F":    "#c62828"
}

PRESETS = {
    "Quiet 3-Leg junction (night)":     ("3leg", 40, 2, False,  [[0, 0, 0],         [20, 80, 0],        [30, 0, 20],        [0, 90, 15]]),
    "Rush-hour crossroads":             ("4leg", 50, 3, True,   [[60, 500, 70],     [50, 350, 60],      [70, 540, 80],      [40, 330, 60]]),
    "Heavy left turns":                 ("4leg", 50, 3, False,  [[220, 450, 80],    [120, 380, 90],     [250, 500, 70],     [100, 350, 80]]),
    "Fast road, 3-Leg (70km/h)":        ("3leg", 70, 3, False,  [[0, 0, 0],         [120, 650, 0],      [160, 0, 140],      [0, 600, 100]]),
    "Overloaded crossroads":            ("4leg", 50, 2, False,  [[150, 800, 100],   [120, 700, 100],    [150, 820, 100],    [120, 720, 100]])
}

# -- METHODS --
class App:
    # Init
    def __init__(self, root: tkinter.Tk, recommender: Recommender | None = None):
        self.root = root
        root.title(APP_NAME)
        root.iconbitmap("icon.ico")

        # App properties
        self.engine = recommender or Recommender()
        self.rec = None
        self.time = 1.00
        self.playing = True

        # Scenario settings
        self.layout         = tkinter.StringVar(value="4leg")
        self.speed          = tkinter.StringVar(value="50")
        self.budget         = tkinter.StringVar(value="3")
        self.pedestrian     = tkinter.BooleanVar(value=False)
        self.method         = tkinter.StringVar(value="ai" if self.engine.has_ai else "optimizer")
        self.preset         = tkinter.StringVar(value="Heavy left turns")
        self.animSpeed      = tkinter.StringVar(value="4x")
        self.flow_vars      = {(a, t): tkinter.StringVar(value="0") for a in range(4) for t in range(3)}
        self.flow_entries   = {}

        self._build_ui()
        self.load_preset("Heavy left turns")
        self._tick()

    # Builder
    def _build_ui(self):
        input_padding = {
            "padx": 6,
            "pady": 3
        }

        # Panel containers
        left    = ttk.Frame(self.root)
        mid     = ttk.Frame(self.root)
        right   = ttk.Frame(self.root)

        left.grid(row=0, column=0, sticky="n", padx=8, pady=8)
        mid.grid(row=0, column=1, sticky="n", pady=8)
        right.grid(row=0, column=2, sticky="n", padx=8, pady=8)

        # Setting inputs
        # Junction settings
        ttk.Label(left, text="1. Junction", font=("Helvetica", 11, "bold")).grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(left, text="Speed limit (km/h)").grid(row=2, column=0, columnspan=2, sticky="w", **input_padding)
        ttk.Label(left, text="Max lanes per approach").grid(row=3, column=0, columnspan=2, sticky="w", **input_padding)
        ttk.Label(left, text="Layout").grid(row=1, column=0, columnspan=2, sticky="w", **input_padding)
        
        ttk.Combobox(left, textvariable=self.layout, values=["4leg", "3leg"], width=6, state="readonly").grid(row=1, column=2, columnspan=2, sticky="w")
        ttk.Combobox(left, textvariable=self.speed, values=["30", "40", "50", "60", "70"], width=6, state="readonly").grid(row=2, column=2, columnspan=2, sticky="w")
        ttk.Combobox(left, textvariable=self.budget, values=["2", "3", "4"], width=6, state="readonly").grid(row=3, column=2, columnspan=2, sticky="w")

        ttk.Checkbutton(left, text="Pedestrian crossings", variable=self.pedestrian).grid(row=4, column=0, columnspan=4, sticky="w", **input_padding)

        # Traffic flow settings
        ttk.Label(left, text="2. Traffic flow (vehicles / hour)", font=("Helvetica", 11, "bold")).grid(row=5, column=0, columnspan=4, sticky="w", pady=(10, 0))
        ttk.Label(left, text="Examples").grid(row=11, column=0, sticky="w", padx=6, pady=(8, 0))

        for index, name in enumerate(("left", "through", "right")):
            ttk.Label(left, text=name).grid(row=6, column=1 + index)

        for index1, approach in enumerate(APPROACHES):
            ttk.Label(left, text=f"{approach} approach").grid(row=7 + index1, column=0, sticky="w", padx=6)

            for index2 in range(3):
                entry = ttk.Entry(left, width=8, justify="center", textvariable=self.flow_vars[(index1, index2)])
                entry.grid(row=7 + index1, column=1 + index2, padx=2, pady=1)
                self.flow_entries[(index1, index2)] = entry

        # > Scenario selection
        preset_box = ttk.Combobox(left, textvariable=self.preset, values=list(PRESETS), width=24, state="readonly")
        preset_box.grid(row=11, column=1, columnspan=3, sticky="w", pady=(8, 0))
        preset_box.bind("<<ComboboxSelected>>", lambda _e: self.load_preset(self.preset.get()))

        ttk.Button(left, text="Random Scenario", command=self.random_scenario).grid(row=12, column=0, columnspan=4, sticky="we", padx=6, pady=3)

        # Model selection
        model_note = (f"AI model loaded: {self.engine.has_ai}" if self.engine.has_ai else "No trained model found")
        if self.engine.load_error:
            model_note += f"\n{self.engine.load_error}"

        ttk.Label(left, text="3. Model selection", font=("Helvetica", 11, "bold")).grid(row=13, column=0, columnspan=4, sticky="w", pady=(10, 0))
        ttk.Label(left, text=model_note, wraplength=280, foreground="#555555").grid(row=18, column=0, columnspan=4, sticky="we", padx=6, pady=8)

        for index, (key, label) in enumerate(METHODS.items()):
            ttk.Radiobutton(left, text=label, variable=self.method, value=key, state="normal" if (key != "ai" or self.engine.has_ai) else "disabled").grid(
                row=14 + index, column=0, columnspan=4, sticky="w", padx=6
            )

        ttk.Button(left, text="Recommend design", command=self.recommend).grid(row=18, column=0, columnspan=4, sticky="we", padx=6, pady=8)

        self.status = ttk.Label(left, text="", wraplength=280, foreground="#b00020")
        self.status.grid(row=20, column=0, columnspan=4, sticky="w", padx=6)

        # Main canvas
        ttk.Label(mid, text="Animation Speed").grid(row=2, column=1)

        ttk.Combobox(mid, textvariable=self.animSpeed, values=["1x", "4x", "10x"], width=5, state="readonly").grid(row=2, column=2)

        ttk.Button(mid, text="Export Image (.png)", command=self.export_png).grid(row=2, column=3, padx=6)

        self.canvas = tkinter.Canvas(mid, width=CANVAS_SIZE, height=CANVAS_SIZE, highlightthickness=0, bg="white")
        self.canvas.grid(row=0, column=0, columnspan=4)
        
        self.timeline = tkinter.Canvas(mid, width=CANVAS_SIZE, height=TIMELINE_HEIGHT, highlightthickness=0, bg="white")
        self.timeline.grid(row=1, column=0, columnspan=4)

        self.play_button = ttk.Button(mid, text="Pause", command=self.toggle_play)
        self.play_button.grid(row=2, column=0, pady=4)

        # Results
        ttk.Label(right, text="Score", font=("Helvetica", 11, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(right, text="Comparison", font=("Helvetica", 11, "bold")).grid(row=4, column=0, sticky="w")
        ttk.Label(right, text="Recommend design", font=("Helvetica", 11, "bold")).grid(row=6, column=0, sticky="w", pady=(6, 0))

        self.score_label = ttk.Label(right, text="-", font=("Helvetica", 34, "bold"))
        self.stats_label = ttk.Label(right, text="Press 'Recommend design'", wraplength=360)

        self.score_label.grid(row=1, column=0, sticky="w")
        self.stats_label.grid(row=1, column=1, sticky="w")

        result_columns = ("group", "veh/h", "capacity", "v/c", "delay (s)")
        self.tree = ttk.Treeview(right, columns=result_columns, show="headings", height=8)
        self.tree.tag_configure("hot", background="#ffd9d9")
        self.tree.grid(row=3, column=0, columnspan=2, sticky="we", pady=6)

        for c, w in zip(result_columns, (70, 60, 70, 55, 65)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="e")

        self.compare_text   = tkinter.Text(right, width=60, height=5, borderwidth=0, background=self.root.cget("background"))
        self.design_text    = tkinter.Text(right, width=60, height=12, borderwidth=0, background=self.root.cget("background"))
        self.notes_text     = tkinter.Text(right, width=60, height=4, borderwidth=0, foreground="#b00020", background=self.root.cget("background"))

        self.compare_text.grid(row=5, column=0, columnspan=2, sticky="w")
        self.design_text.grid(row=7, column=0, columnspan=2, sticky="w")
        self.notes_text.grid(row=8, column=0, columnspan=2, sticky="w", pady=4)

    # Class (app) methods
    def _on_layout_change(self):
        mask = VALID_MOVES[self.layout.get()]

        for (a, t), e in self.flow_entries.items():
            if mask[a, t]:
                e.configure(state="normal")
            else:
                self.flow_vars[(a, t)].set("0")
                e.configure(state="disabled")

        self.rec = None
        self._clear_results()

    def _apply_scenario(self, scenario: Scenario):
        self.layout.set(scenario.layout)
        self.speed.set(str(int(scenario.speed_kmh)))
        self.budget.set(str(scenario.lane_budget))
        self.pedestrian.set(bool(scenario.pedestrian))

        for a in range(4):
            for t in range(3):
                self.flow_vars[(a, t)].set(str(int(scenario.flows[a, t])))

        self._on_layout_change()

    def load_preset(self, name):
        layout, speed, budget, pedestrian, flows = PRESETS[name]
        self._apply_scenario(Scenario(layout, flows, speed, budget, pedestrian))
        self.recommend()

    def random_scenario(self):
        self._apply_scenario(random_scenario(numpy.random.default_rng()))
        self.recommend()

    def read_scenario(self) -> Scenario:
        flows = numpy.zeros((4, 3))

        for (a, t), var in self.flow_vars.items():
            text = var.get().strip() or "0"
            try:
                v = float(text)
            except ValueError:
                raise ValueError(f"{APPROACHES[a]} {TURNS[t]}: '{text}' is not a number")

            if not 0 <= v <= 3000:
                # TODO: Un-hardcode and make a conf file
                raise ValueError(f"{APPROACHES[a]} {TURNS[t]}: flow must be between 0 and 3000 veh/h")

            flows[a, t] = v

        if flows.sum() <= 0:
            raise ValueError("enter some traffic flow first")

        return Scenario(
            self.layout.get(), 
            flows, 
            float(self.speed.get()), 
            int(self.budget.get()),
            self.pedestrian.get()
        )

    def recommend(self):
        try:
            scenario = self.read_scenario()
            self.status.configure(text="")
            self.rec = self.engine.recommend(scenario, self.method.get())
        except Exception as e:
            self.status.configure(text=str(e))
            return

        self.time = 0.00
        self._show(self.rec)
        self._draw()

    def toggle_play(self):
        self.playing = not self.playing
        self.play_button.configure(text="Pause" if self.playing else "Play")

    def export_png(self):
        if self.rec in None:
            messagebox.showinfo("Export", "Recommend a design first")
            return

        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG image", "*.png")])
        if not path:
            return

        try:
            states = signal_state(self.rec.scenario, self.rec.design, self.time)[0]
            render_png(build_scenario(self.rec.scenario, self.rec.design, states, CANVAS_SIZE), path, CANVAS_SIZE, CANVAS_SIZE)
            messagebox.showinfo("Export", f"Saved to {path}")
        except ImportError:
            messagebox.showerror("Export", "Exporting needs Pillow package; install by running 'pip install pillow'")

    @staticmethod
    def _set_text(widget, lines):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", "\n".join(lines))
        widget.configure(state="disabled")

    def _clear_results(self):
        self.score_label.configure(text="-", foreground="#000000")
        self.stats_label.configure(text="Press 'Recommend design'")
        self.tree.delete(*self.tree.get_children())

        for w in (self.compare_text, self.design_text, self.notes_text):
            self._set_text(w, [])

        self._draw()

    def _show(self, recommendation: Recommendation):
        result = recommendation.result
        self.score_label.configure(text=f"{recommendation.score}/100", foreground=SCORE_COLORS.get(result.los, "#000000"))
        self.stats_label.configure(
            text=f"Level of service: {result.los} | Avarage delay: {result.avarage_delay:.1f} s/veh\nWorst lane group at: {result.max_x:.2f}x capacity\nCycle: {result.cycle:.0f}s | {result.total_lanes} lanes"
        )

        self.tree.delete(*self.tree.get_children())

        for name, group in sorted(result.groups.items()):
            self.tree.insert("", "end", values=(name, f"{group["volume"]:.0f}", f"{group["capacity"]:.0f}", f"{group["x"]:.2f}", f"{group["delay"]:.1f}"), tags=("hot") if group["x"] > 0.95 else ())

        lines = []
        for key, c in recommendation.comparison.items():
            mark = "> " if key == recommendation.method else "  "
            flag = "" if c["legal"] else "\t(not allowed)"
            lines.append(f"{mark}{METHODS[key]}: {c["delay"]:.1f} s/veh, score {c["score"]}{flag}")

        self._set_text(self.compare_text, lines)
        self._set_text(self.design_text, design_summary(recommendation.scenario, recommendation.design, result))
        self._set_text(self.notes_text, recommendation.notes)

    @staticmethod
    def _paint(canvas: tkinter.Canvas, prims):
        canvas.delete("all")

        for prim in prims:
            if prim["type"] == "rect":
                canvas.create_rectangle(*prim["box"], fill=prim["fill"], outline=prim.get("outline") or "", width=0)
            elif prim["type"] == "line":
                canvas.create_line(*[c for pt in prim["points"] for c in pt], fill=prim["fill"], width=prim["width"])
            elif prim["type"] == "poly":
                canvas.create_polygon(*[c for pt in prim["points"] for c in pt], fill=prim["fill"], outline="")
            elif prim["type"] == "text":
                canvas.create_text(*prim["pos"], text=prim["text"], fill=prim["fill"], font=("Helvetica", prim["size"], "bold"))

    def _draw(self):
        if self.rec is not None:
            states = signal_state(self.rec.scenario, self.rec.design, self.time)[0]
            self._paint(self.canvas, build_scenario(self.rec.scenario, self.rec.design, states, CANVAS_SIZE))
            self._paint(self.timeline, build_timeline(self.rec.scenario, self.rec.design, self.time, CANVAS_SIZE, TIMELINE_HEIGHT))
            return

        try:
            scenario = self.read_scenario()
        except ValueError:
            return

        self._paint(self.canvas, build_scenario(scenario, fixed_design(scenario), None, CANVAS_SIZE))
        self.timeline.delete("all")

    def _tick(self):
        if self.playing and self.rec is not None:
            self.time += 0.05 * int(self.animSpeed.get().rstrip("x"))
            self._draw()
        self.root.after(50, self._tick)

if __name__ == "__main__":
    root = tkinter.Tk()
    App(root)
    root.mainloop()