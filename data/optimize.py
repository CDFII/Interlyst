"""
Optimization service
Finds optimal intersection design for each scenario.
"""
# Imports
import numpy

from dataclasses import replace

from sim.baselines import fixed_design, heuristic_lanes, webster_design
from sim.delay_model import (
    ACTIVE, LANE_UTIL, MAX_CYCLE, MAX_LEFT_LANES, MAX_THRU_LANES, MIN_CYCLE, N_PLANS, SAT_FLOW,
    Design, Scenario,
    allocate_greens, evaluate, intergreen, min_green, objective, served_groups
)

# Config
_MOVES = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1))   # (d_left, d_thru)

# -- METHODS --
def _neighbours(l, t, budget):
    for dl, dt in _MOVES:
        nl, nt = l + dl, t + dt # NeighborLeft, NeighbotThru
        if 0 <= nl <= MAX_LEFT_LANES and 1 <= nt <= MAX_THRU_LANES and nl + nt <= budget:
            yield nl, nt

def _phase_ratios(scenario: Scenario, left, thru, plan):
    pass

def cheap_greens(scenario: Scenario, left, thru, plan):
    pass

def refine_greens(scenario: Scenario, design, steps={8.00, 4.00, 2.00, 1.00}):
    pass

def optimize(scenario: Scenario):
    pass