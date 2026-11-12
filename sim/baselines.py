"""
Baseline models, which are compared to trained AI model.
"""
# Imports
import math

from sim.delay_model import (
    ACTIVE, LANE_UTIL, LEFT_FACTOR, MAX_CYCLE, MAX_THRU_LANES, MIN_CYCLE, PLANS, SAT_FLOW,
    Design, Scenario, allocate_greens, intergreen, min_green
)

# Config
WEBSTER_LEFT_PCE        = 1.6

# -- METHODS --
def _flow_ratio(scenario: Scenario, a, l, t):
    L, T, R = scenario.flows[a]
    cap_tr = SAT_FLOW * t * LANE_UTIL[t]
    if l > 0:
        return max(L / (SAT_FLOW * LEFT_FACTOR * l * LANE_UTIL[l]), (T + R) / cap_tr)
    return (T + R + WEBSTER_LEFT_PCE * L) / cap_tr

def heuristic_lanes(scenario: Scenario):
    left, thru = [0] * 4, [0] * 4
    for active in ACTIVE[scenario.layout]:
        L, T, R = scenario.flows[active]
        l = 1 if (L >= 150 and scenario.lane_budget >= 2) else 0
        need = math.ceil((T+ R + (0 if l else 1.50 * L)) / 550.00)
        left[active] = l
        thru[active] = max(1, min(need, scenario.lane_budget - l, MAX_THRU_LANES))
    return tuple(left), tuple(thru)

def fixed_design(scenario: Scenario, cycle: float = 90.00) -> Design:
    left, thru = heuristic_lanes(scenario)
    n = len(PLANS[scenario.layout][0])
    lost = intergreen(scenario.speed_kmh)
    g = max(min_green(scenario), (cycle - n * lost) / n)
    return Design(left, thru, 0, tuple([g] * n))

def webster_design(scenario: Scenario) -> Design:
    left, thru = heuristic_lanes(scenario)
    phases = PLANS[scenario.layout][0]
    n = len(phases)
    lost, mg = intergreen(scenario.speed_kmh), min_green(scenario)

    ys = []
    for phase in phases:
        approaches = {approach for approach, _ in phase}
        ys.append(max(_flow_ratio(scenario, approach, left[approach], thru[approach]) for approach in approaches))
    y_sum = sum(ys)

    big_y   = min(y_sum, 0.90)
    cycle   = (1.50 * n * lost + 5.00) / (1.00 - big_y)
    cycle   = min(max(cycle, n * (mg + lost), MIN_CYCLE), MAX_CYCLE)
    greens  = allocate_greens([max(y, 1e-3) for y in ys], cycle, lost, mg)
    return Design(left, thru, 0, greens)