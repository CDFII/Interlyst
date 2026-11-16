"""

"""
# Imports
import numpy, tkinter

from tkinter import filedialog, messagebox, ttk

from data.generate import random_scenario
from interface.toolbox import METHODS, Recommender, design_summary, signal_state
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
        root.title = APP_NAME

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
        
        ttk.Combobox(left, textvariable=self.layout, values=["4leg", "3leg"], state="readonly").grid(row=1, column=2, columnspan=2, sticky="w")
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
                entry = ttk.Entry(left, widget=6, justify="right", textvariable=self.flow_entries[(index1, index2)])
                entry.grid(row=7 + index, column=1 + index2, padx=2, pady=1)
                self.flow_entries[(index, index2)] = entry

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

        ttk.Button(left, text="Recommended design", command=self.recommend).grid(row=18, column=0, columnspan=4, sticky="we", padx=6, pady=8)

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

        # TODO: Add results
        # TODO: Add methods to app (play/pause, load scenarios, etc)