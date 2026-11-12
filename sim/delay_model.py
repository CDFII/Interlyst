"""
Labeling service for training data, based on HCM (Highway Capacity Manual).
"""
# Imports
import math, numpy

from dataclasses import dataclass
from functools import lru_cache

# Config
APPROACHES              = ("N", "E", "S", "W")
TURNS                   = ("L", "T", "R")               # Left, Thru, Right

SAT_FLOW                = 1800.00                       # Vehicles/Hour per lane of green
LEFT_FACTOR             = 0.95
RIGHT_PENALTY           = 0.10
LANE_UTIL = {
    1:                  1.00,
    2:                  0.95,
    3:                  0.91
}
SNEAKERS                = 2.00
CRITICAL_GAP            = 4.10                          # Seconds
FOLLOW_UP               = 2.20                          # Seconds
ANALYSIS_PERIOD         = 0.25                          # Hours

SCORING                 = ((10, "A"), (20, "B"), (35, "C"), (55, "D"), (80, "E"))

MIN_GREEN               = 10.00                         # Seconds
MIN_GREEN_PEDESTRIAN    = 15.00                         # Seconds
MIN_CYCLE               = 40.00                         # Seconds
MAX_CYCLE               = 150.00                        # Seconds
MAX_LEFT_LANES          = 2
MAX_THRU_LANES          = 3

# Rule config:
# Unprotected left turns are not allowed, if:
# 1. Speed limit is equal or above PROTECT_SPEED, or
# 2. Left volume * opposing volume is equal or above PROTECT_CROSSING
# + Left volume is at least PROTECT_MIN_LEFT
PROTECT_SPEED           = 60.00                         # km/h
PROTECT_CROSSING        = 90000.00                      # Vehicles^2/Hour^2
PROTECT_MIN_LEFT        = 50.00                         # Vehicles/Hour

LANE_COST               = 1.00                          # Seconds, cost of adding lanes to the solution. Prevents model from adding a ton of lanes.
INFEASIBLE              = 1e6

_T_MASK = numpy.array([
    [0, 0, 0],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1]
], dtype=bool)
ACTIVE = {
    "3leg":             (1, 2, 3),
    "4leg":             (0, 1, 2, 3)
}
VALID_MOVES = {
    "3leg":             _T_MASK,
    "4leg":             numpy.ones((4, 3), dtype=bool)
}

# Datamodels
@dataclass
class Scenario:
    layout:             str                         # 3leg or 4leg
    flows:              numpy.ndarray               # (4, 3) vehicles/hour, [approach][L, T, R]
    speed_kmh:          float           = 50.00
    lane_budget:        int             = 3         # Max entering lanes per approach
    pedestrian:         bool            = False     # Is pedestrian crossing present?

    def __post_init__(self):
        if self.layout not in ACTIVE:
            raise ValueError("Layout must be '3leg' or '4leg'")

        _flows = numpy.asarray(self.flows, dtype=float).reshape(4, 3)
        self.flows = numpy.where(VALID_MOVES[self.layout], _flows, 0.00)    # Drop impossible moves
        self.lane_budget = int(self.lane_budget)

@dataclass(frozen=True)
class Design:
    left_lanes:         tuple
    thru_lanes:         tuple
    plan:               int
    greens:             tuple

@dataclass
class Result:
    avarage_delay:      float                       # Seconds/Vehicle, volume-weighted over all movements
    objective:          float                       # avarage_delay + LANE_COST * lanes (optimizer aims to minimize this)
    max_x:              float                       # Worst volume/capacity ratio of any lane group
    cycle:              float                       # Seconds
    total_lanes:        int
    los:                str                         # HCM level of service on scale A-F
    feasible:           bool
    groups:             dict                        # "N-L" -> {volume, capacity, x, delay}

# Signal plans & helper methods
def _all(*approaches):
    return [(approach, group) for approach in approaches for group in ("L", "TR")]

PLANS = {
    "3leg": [
        [_all(0, 2), _all(1, 3)]                                                                        # 0: (NS, EW)                   Two phases, permitted lefts
        [[(0, "L"), (2, "L")], [(0, "TR"), (2, "TR")], _all(1, 3)],                                     # 1: (NS-L, NS-TR, EW)          Protected NS lefts, permitted EW lefts
        [_all(0, 2), [(1, "L"), (3, "L")], [(1, "TR"), (3, "TR")]],                                     # 2: (NS, EW-L, EW_TR)          Permitted NS lefts, protected EW lefts
        [[(0, "L"), (2, "L")], [(0, "TR"), (2, "TR")], [(1, "L"), (3, "L")], [(1, "TR"), (3, "TR")]],   # 3: (NS-L, NS-TR, EW-L, EW-TR) Fully protectedlefts
        [_all(0), _all(1), _all(2), _all(3)]                                                            # 4: (N, E, S, W)               Split phasing, one approach at a time
    ],
    "4leg": [
        [_all(1, 3), _all(2)],                                                                          # 0: (E+W, S)                   Main road together, then stem
        [[(1, "L"), (1, "TR")], [(1, "TR"), (3, "TR")], _all(2)],                                       # 1: (E-L+E-TR, E-TR+W-TR, S)   Protected E-left, then main road, then stem
        [_all(1), _all(3), _all(2)]                                                                     # 2: (E, W, S)                  Split phasing
    ]
}
PLAN_NAMES = {
    "3leg": [
        "2-Phase (permitted lefts)",
        "Protected NS lefts",
        "Protected EW lefts",
        "4-Phase protected lefts",
        "Split phasing (N, E, S, W)"
    ],
    "4leg": [
        "2-Phase (Main, Stem)",
        "Protected E-left",
        "Split phasing (E, S, W)"
    ]
}
N_PLANS = {k: len(v) for k, v in PLANS.items()}

def n_phases(layout: str, plan: int) -> int:
    return len(PLANS[layout][plan])

# -- HELPER METHODS --
def intergreen(speed_kmh: float) -> float:
    v = max(speed_kmh / 3.60, 5.00)
    return 1.00 + v / (2 * 3.00) + 12.00 / v

def min_green(scenario: Scenario) -> float:
    return MIN_GREEN_PEDESTRIAN if scenario.pedestrian else MIN_GREEN

def total_lanes(scenario: Scenario, design: Design) -> int:
    return sum(design.left_lanes[active] + design.thru_lanes[active] for active in ACTIVE[scenario.layout])

def los_from_delay(delay: float) -> str:
    for limit, score_letter in SCORING:
        if delay <= limit:
            return score_letter
    return "F"

def needs_protection(speed_kmh: float, left: float, v_opp: float) -> bool:
    return (left >= PROTECT_MIN_LEFT and v_opp > 0 and (speed_kmh >= PROTECT_SPEED or left * v_opp >= PROTECT_CROSSING))

def gap_capacity(v_opp: float) -> float:
    if v_opp <= 0:
        return 3600.00 / FOLLOW_UP
    q = v_opp / 3600.00
    return v_opp * math.exp(-q * CRITICAL_GAP) / (1.00 - math.exp(-q * FOLLOW_UP))

def lane_group_delay(volume: float, capacity: float, green: float, cycle: float) -> float:
    # Magic crap
    if capacity <= 0 or green <= 0:
        return 3600.00
    x = volume / capacity
    g_c = green / cycle
    d1 = 0.50 * cycle * (1.00 - g_c) ** 2 / (1.00 - min(1.00, x) * g_c)
    d2 = 900.00 * ANALYSIS_PERIOD * ((x - 1.00) + math.sqrt((x - 1.00) ** 2 + 8.00 * 0.50 * x / (capacity * ANALYSIS_PERIOD)))

    return d1 + d2

def allocate_greens(ratios, cycle: float, lost: float, mg: float) -> tuple:
    n = len(ratios)
    free = cycle - n * lost - n * mg
    if free <= 0:
        return tuple([mg] * n)
    s = sum(ratios)
    return tuple(mg + free * r / s for r in ratios)

@lru_cache(maxsize=None)
def served_groups(layout: str, plan: int, shared: tuple):
    out = []
    for phase in PLANS[layout][plan]:
        s = set()
        for approach, group in phase:
            s.add((approach, "TR") if shared[approach] else (approach, group))
        out.append(frozenset(s))
    return tuple(out)

# -- MAIN METHODS --
def _core(scenario: Scenario, design: Design, detail: bool, enforce: bool = True):
    flows       = scenario.flows.tolist()
    greens      = design.greens
    lost        = intergreen(scenario.speed_kmh)
    cycle       = sum(greens) + len(greens) * lost
    served      = served_groups(scenario.layout, design.plan, tuple(left_lane == 0 for left_lane in design.left_lanes))
    active      = ACTIVE[scenario.layout]

    gtot        = {}
    for phase, green in enumerate(greens):
        for group in served[phase]:
            gtot[group] = gtot.get(group, 0.00) + green

    sum_v = sum_vd = max_x = 0.00
    info = {} if detail else None

    for a in active:
        L, T, R     = flows[a]
        opp         = (a + 2) % 4
        opp_on      = opp in active
        v_opp       = flows[opp][1] + flows[opp][2] if opp_on else 0.00
        g_opp       = gtot.get((opp, "TR"), 0.00) if opp_on else 0.00
        vo_rate     = 0.00 
        if v_opp > 0 and g_opp > 0:
            vo_rate = min(v_opp * cycle / g_opp, SAT_FLOW * max(1, design.thru_lanes[opp]))

        shared      = design.left_lanes[a] == 0
        specs       = []
        if shared:
            if L + T + R > 0:
                specs.append(("TR", L + T + R, design.thru_lanes[a]))
        else:
            if L > 0:
                specs.append(("L"), L, design.left_lanes[a])
            if T + R > 0:
                specs.append(("TR", T + R, design.thru_lanes[a]))

        for kind, v, lanes in specs:
            key             = (a, kind)
            g_total         = gtot.get(key, 0.00)
            if g_total <= 0 or lanes <= 0:
                return INFEASIBLE, 9.99, cycle, None

            base            = SAT_FLOW * lanes * LANE_UTIL.get(lanes, 0.88)
            share_r         = R / v if kind == "TR" else 0.00
            share_l         = L / v if (kind == "TR" and shared) else 0.00

            cap_green       = 0.00
            permitted       = False
            left_conflict   = False
            for phase, green in enumerate(greens):
                if key not in served[phase]:
                    continue
                opposed = vo_rate > 0 and (opp, "TR") in served[phase]
                if kind == "L":
                    if opposed:
                        s = min(LEFT_FACTOR * SAT_FLOW, gap_capacity(vo_rate))
                        permitted = left_conflict = True
                    else:
                        s = LEFT_FACTOR * base
                else:
                    s = base * (1.00 - RIGHT_PENALTY * share_r)
                    if share_l > 0:
                        e_l = min(6.00, 1.10 + 0.005 * vo_rate) if opposed else 1.05
                        s /= 1.00 + share_l * (e_l - 1.00)
                        if opposed:
                            left_conflict = True
                cap_green += green * s

            if enforce and left_conflict and needs_protection(scenario.speed_kmh, L, v_opp):
                return INFEASIBLE, 9.99, cycle, None                                            # Illegal permitted left
            cap = cap_green / cycle
            if permitted:
                cap += SNEAKERS * 3600.00 / cycle

            x           = v / cap
            delay       = lane_group_delay(v, cap, g_total, cycle)
            sum_v       += v
            sum_vd      += v * delay
            if x > max_x:
                max_x = x
            if detail:
                info[f"{APPROACHES[a]}-{kind}"] = {
                    "volume":       v,
                    "capacity":     cap,
                    "x":            x,
                    "delay":        delay
                }

    avarage = sum_vd / sum_v if sum_v > 0 else 0.00
    return avarage, max_x, cycle, info

def objective(scenario: Scenario, design: Design, enforce_rules: bool = True) -> float:
    avarage, _, _, _ = _core(scenario, design, False, enforce_rules)
    if avarage >= INFEASIBLE:
        return INFEASIBLE
    return avarage + LANE_COST * total_lanes(scenario, design)

def evaluate(scenario: Scenario, design: Design, enforce_rules: bool = True) -> Result:
    avarage, max_x, cycle, info = _core(scenario, design, True, enforce_rules)
    ok = avarage < INFEASIBLE
    n_lanes = total_lanes(scenario, design)
    return Result(
        avarage_delay   = avarage,
        objective       = avarage + LANE_COST * n_lanes if ok else INFEASIBLE,
        max_x           = max_x,
        cycle           = cycle,
        total_lanes     = n_lanes,
        los             = los_from_delay(avarage) if ok else "F",
        feasible        = ok,
        groups          = info or {}
    )

def check_design(scenario: Scenario, design: Design) -> list:
    bad = []
    if not 0 <= design.plan < N_PLANS[scenario.layout]:
        return ["invalid plan id"]
    if len(design.greens) != n_phases(scenario.layout, design.plan):
        bad.append("wrong number of green times for this plan")
    active = ACTIVE[scenario.layout]
    for a in range(4):
        l, t = design.left_lanes[a], design.thru_lanes[a]
        if a in active:
            if not (0 <= l <= MAX_LEFT_LANES and 1 <= t <= MAX_THRU_LANES):
                bad.append(f"{APPROACHES[a]}: lane counts out of range")
            if l + t > scenario.lane_budget:
                bad.append(f"{APPROACHES[a]}: exceeds lane budget")
        elif l or t:
            bad.append(f"{APPROACHES[a]}: approach does not exist in this layout")
    
    mg = min_green(scenario)
    if any(green < mg - 1e-6 for green in design.greens):
        bad.append("green shorter than maximum")

    cycle = sum(design.greens) + len(design.greens) * intergreen(scenario.speed_kmh)
    if cycle > MAX_CYCLE + 1e-6:
        bad.append("cycle longer than maximum")
    if cycle < MIN_CYCLE - 1e-6:
        bad.append("cycle shorter than minimum")
    return bad