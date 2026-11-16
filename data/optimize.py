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

CYCLE_AMOUNT = 6
REFINE_CYCLES = 40

# -- METHODS --
def _neighbours(l, t, budget):
    for dl, dt in _MOVES:
        nl, nt = l + dl, t + dt # NeighborLeft, NeighbotThru
        if 0 <= nl <= MAX_LEFT_LANES and 1 <= nt <= MAX_THRU_LANES and nl + nt <= budget:
            yield nl, nt

def _phase_ratios(scenario: Scenario, left, thru, plan):
    served = served_groups(
        scenario.layout,
        plan,
        tuple(l == 0 for l in left)
    )
    n_serve = {}
    for phase in served:
        for group in phase:
            n_serve[group] = n_serve.get(group, 0) + 1

    ratios = []
    for phase in served:
        y = 0.05
        for active, kind in phase:
            L, T, R = scenario.flows[active]
            if kind == "L":
                v, n = L, left[active]
            else:
                v, n = (T + R, thru[active]) if left[active] > 0 else (L + T + R, thru[active])

            if v <= 0 or n <= 0:
                continue

            yg = v / (SAT_FLOW * n * LANE_UTIL.get(n, 0.88))
            if kind == "L" and (((active + 2) % 4), "TR") in phase:
                yg *= 2.00              # Permitted left is given longer green
            elif kind == "TR" and left[active] == 0 and L > 0:
                yg *= 1.00 + L / v      # Lane is shared with lefts

            y = max(y, yg / n_serve[(active, kind)])

        ratios.append(y)

    return ratios

def cheap_greens(scenario: Scenario, left, thru, plan):
    ratios = _phase_ratios(scenario, left, thru, plan)
    n = len(ratios)
    lost, mg = intergreen(scenario.speed_kmh), min_green(scenario)
    lo = min(max(n * (mg + lost), MIN_CYCLE), MAX_CYCLE)
    best_cycle, best_design = float("inf"), None

    for cycle in numpy.linspace(lo, MAX_CYCLE, CYCLE_AMOUNT):
        design = Design(tuple(left), tuple(thru), plan, allocate_greens(ratios, float(cycle), lost, mg))
        cycle = objective(scenario, design)
        if cycle < best_cycle:
            best_cycle, best_design = cycle, design

    return best_cycle, best_design

def refine_greens(scenario: Scenario, design: Design, steps={8.00, 4.00, 2.00, 1.00}):
    n = len(design.greens)
    lost, mg = intergreen(scenario.speed_kmh), min_green(scenario)
    min_total = MIN_CYCLE - n * lost
    max_total = MAX_CYCLE - n * lost
    best_greens = list(design.greens)
    best_c = objective(scenario, design)

    moves = []
    for index in range(n):
        moves.append((index, None))
        moves.append((None, index))
        moves.extend((index, j) for j in range(n) if j != index)

    for step in steps:
        for _ in range(REFINE_CYCLES):
            improved = False
            for gi, gj in moves:
                candidates = best_greens[:]
                
                if gi is not None:
                    candidates[gi] += step
                
                if gj is not None:
                    candidates[gj] -= step

                if min(candidates) < mg - 1e-9 or not (min_total - 1e-9 <= sum(candidates) <= max_total + 1e-9):
                    continue

                c = objective(scenario, replace(design, greens=tuple(candidates)))
                if c < best_c - 1e-9:
                    best_g, best_c, improved = candidates, c, True

            if not improved:
                break

    return best_c, replace(design, greens=tuple(best_g))

def optimize(scenario: Scenario):
    base_l, base_t = heuristic_lanes(scenario)
    candidates = []

    for plan in range(N_PLANS[scenario.layout]):
        left, thru = list(base_l), list(base_t)
        cost, _ = cheap_greens(scenario, left, thru, plan)
        improved = True

        while improved:
            improved = False
            for active in ACTIVE[scenario.layout]:
                for nl, nt in _neighbours(left[active], thru[active], scenario.lane_budget):
                    l2, t2 = left[:], thru[:]
                    l2[active], t2[active] = nl, nt
                    c2, _ = cheap_greens(scenario, l2, t2, plan)

                    if c2 < cost - 1e-9:
                        left, thru, cost, improved = l2, t2, c2, True

        _, d0 = cheap_greens(scenario, left, thru, plan)
        candidates.append(refine_greens(scenario, d0))

    for seed in (webster_design(scenario), fixed_design(scenario)):
        candidates.append(refine_greens(scenario, seed))

    _, best = min(candidates, key=lambda cd: cd[0])
    return best, evaluate(scenario, best)