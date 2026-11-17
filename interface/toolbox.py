"""
Important mini and helper functions for main app to use
"""
# Imports
import os, pickle

from dataclasses import dataclass, field

from data.generate import scenario_features
from data.optimize import optimize
from model.codec import decode_design
from sim.baselines import fixed_design, webster_design
from sim.delay_model import (
    ACTIVE, APPROACHES, PLAN_NAMES, PLANS,
    Design, Result, Scenario,
    evaluate, intergreen, served_groups
)

# Config
METHODS = { # Method descriptions
    "ai":           "AI Model",
    "optimizer":    "Optimizer (reference)",
    "webster":      "Webster formula",
    "fixed":        "Fixed 90s timing"
}

# -- METHODS --
def score_from_delay(avarage_delay: float, legal: bool = True) -> int:
    if not legal:
        return 0
    return int(round(100 * max(0.00, 1.00 - avarage_delay / 80.00)))

def phase_label(layout: str, plan: int, p: int) -> str:
    by_app = {}
    parts = []

    for a, g in PLANS[layout][plan][p]:
        by_app.setdefault(a, set()).add(g)

    for a in sorted(by_app):
        gs = by_app[a]
        if gs == {"L", "TR"}:
            parts.append(APPROACHES[a])
        elif gs == {"L"}:
            parts.append(f"{APPROACHES[a]}-left")
        else:
            parts.append(f"{APPROACHES[a]}-thru/right")

    return ", ".join(parts)

def design_summary(scenario: Scenario, design: Design, result: Result) -> list:
    lines = [
        f"Plan: {PLAN_NAMES[scenario.layout][design.plan]}",
        f"Cycle: {result.cycle:.0f}s (lost time: {intergreen(scenario.speed_kmh):.1f}s per phase)"
    ]

    for phase, green in enumerate(design.greens):
        lines.append(f" | Phase {phase + 1}: {green:.0f}s green - {phase_label(scenario.layout, design.plan, phase)}")

    for a in ACTIVE[scenario.layout]:
        left = f"{design.left_lanes[a]} left + " if design.left_lanes[a] else "no left lane, "
        lines.append(f"{APPROACHES[a]} approach: {left}{design.thru_lanes[a]} through/right lane(s)")

    return lines

def signal_state(scenario: Scenario, design: Design, time: float):
    served = served_groups(scenario.layout, design.plan, tuple(l == 0 for l in design.left_lanes))
    lost = intergreen(scenario.speed_kmh)
    yellow = min(lost, 1.00 + max(scenario.speed_kmh / 3.60, 5.00) / 6.00)
    n = len(design.greens)
    cycle = sum(design.greens) + n * lost
    time = time % cycle
    start, current, inter, elapsed = 0.00, 0, False, 0.00

    for phase, green in enumerate(design.greens):
        if time < start + green:
            current, inter = phase, False
            break
        if time < start + green + lost:
            current, inter, elapsed = phase, True, time - start - green
            break

        start += green + lost

    nxt = (current + 1) % n
    states = {}

    for group in set().union(*served):
        if group not in served[current]:
            states[group] = "RED"
        elif not inter or group in served[nxt]:
            states[group] = "GREEN"
        else:
            states[group] = "YELLOW" if elapsed < yellow else "RED"

    return states, current, inter, cycle

# Recommendation system
@dataclass
class Recommendation:
    scenario:       Scenario
    method:         str
    design:         Design
    result:         Result
    legal:          bool
    score:          int
    notes:          list        = field(default_factory=list)
    comparison:     dict        = field(default_factory=dict)   # Method -> {"delay", "legal", "score"}

class Recommender:
    # Holds the trained model, if one exists

    def __init__(self, torch_ckpt="output/checkpoint.pt", sk_path="output/sklearn_model.pkl"):
        self.predictor, self.model_name, self.load_error = None, None, None

        if os.path.exists(torch_ckpt):
            try:
                from model.model import TorchPredictor
                self.predictor, self.model_name = TorchPredictor.load(torch_ckpt), "neural network (PyTorch)"
            except Exception as e:
                self.load_error = f"couldn't load {torch_ckpt}: {e}"

        if self.predictor is None and os.path.exists(sk_path):
            try:
                with open(sk_path, "rb") as f:
                    self.predictor, self.model_name = pickle.load(f), "cikit-learn MLPs"
            except Exception as e:
                self.load_error = f"couldn't load {sk_path}: {e}"

    @property
    def has_ai(self) -> bool:
        return self.predictor is not None

    def _design(self, scenario: Scenario, method: str):
        notes = []

        if method == "ai":
            if not self.has_ai:
                raise RuntimeError("no trained model found (try training one first!)")

            out = self.predictor(scenario_features(scenario)[None, :])
            design, info = decode_design(
                scenario,
                out["plan_logits"][0],
                out["left_logits"][0],
                out["thru_logits"][0],
                out["frac_logits"][0],
                float(out["cycle"][0])
            )

            if info["plan_rank"] > 0:
                notes.append("The model's first-choice plan broke a safety rule, so its next choice was used")

            if info["lanes_repaired"]:
                notes.append("The model's lane choice exceeded the lane budget and was adjusted")

            if info["greens_repaired"]:
                notes.append("Some green times were below the minimum and were raised")

            return design, notes

        if method == "optimizer":
            return optimize(scenario)[0], notes

        if method == "webster":
            return webster_design(scenario), notes

        if method == "fixed":
            return fixed_design(scenario), notes

        raise ValueError(method)

    def _score(self, scenario: Scenario, design: Design):
        result = evaluate(scenario, design)
        legal = result.feasible

        if not legal:
            result = evaluate(scenario, design, enforce_rules=False)

        return result, legal, score_from_delay(result.avarage_delay, legal)

    def recommend(self, scenario: Scenario, method: str = "ai") -> Recommendation:
        design, notes = self._design(scenario, method)
        result, legal, score = self._score(scenario, design)

        if not legal:
            notes.append("This design lets left turns cross opposing traffic where that is not allowed (speed >= 60km/h or heavy left x opposing flow), scoring 0.")
        if result.max_x > 1.00:
            notes.append(f"Over capacity: the busiest lane group is at {result.max_x:.2f}x capacity, queues will keep growing. Try a bigger lane budget.")

        comparison = {}

        for available_method in METHODS:
            if available_method == method:
                comparison[available_method] = {
                    "delay": result.avarage_delay,
                    "legal": legal,
                    "score": score
                }
            elif available_method == "ai" and not self.has_ai:
                continue
            else:
                r2, l2, s2 = self._score(scenario, self._design(scenario, available_method)[0])
                comparison[available_method] = {
                    "delay": r2.avarage_delay,
                    "legal": l2,
                    "score": s2
                }

        return Recommendation(
            scenario,
            method,
            design,
            result,
            legal,
            score,
            notes,
            comparison
        )