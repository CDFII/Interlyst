"""
Repairs odd outputs, such as lane and traffic light issues.
"""
# Imports
import numpy, pandas

from data.optimize import cheap_greens
from sim.delay_model import (
    ACTIVE, APPROACHES, MAX_CYCLE, MIN_CYCLE, N_PLANS,
    Design, Scenario,
    allocate_greens, check_design, evaluate, intergreen, min_green, n_phases,
)

# Config
IGNORE = -100
MAX_PHASES = 4

# -- METHODS --
def encode_labels(dataframe: pandas.DataFrame) -> dict:
    left = numpy.stack([dataframe[f"left_{A}"].to_numpy() for A in APPROACHES], 1).astype(numpy.int64)
    thru = numpy.stack([dataframe[f"thru_{A}"].to_numpy() for A in APPROACHES], 1).astype(numpy.int64) - 1
    missing = (dataframe["layout"].to_numpy() == "3leg")[:, None] & (numpy.arange(4) == 0)[None, :]

    left[missing] = IGNORE
    thru[missing] = IGNORE

    greens = dataframe[[f"green_{phase}" for phase in range(MAX_PHASES)]].to_numpy(float)
    return {
        "plan": dataframe["plan"].to_numpy(numpy.int64),
        "left": left,
        "thru": thru,
        "n_ph": (greens > 0).sum(1).astype(numpy.int64),
        "frac": (greens / greens.sum(1, keepdims=True)).astype(numpy.float32),
        "cycle": ((dataframe["cycle"].to_numpy(float) - MIN_CYCLE) / (MAX_CYCLE - MIN_CYCLE)).astype(numpy.float32),
        "delay": (numpy.log1p(dataframe["delay_opt"].to_numpy(float)) / 4.00).astype(numpy.float32)
    }

def split_indices(n: int, seed: int = 0, value: float = 0.10, test: float = 0.10) -> dict:
    permutation = numpy.random.default_rng(seed).permutation(n)
    n_test, n_value = int(n * test), int(n * value)
    return {
        "test": permutation[:n_test],
        "value": permutation[n_test:n_test + n_value],
        "train": permutation[n_test + n_value:]
    }

def oracle_outputs(dataframe: pandas.DataFrame) -> dict:
    y = encode_labels(dataframe)
    n = len(dataframe)
    plan_logits = numpy.zeros((n, 5)); plan_logits[numpy.arange(n), y["plan"]] = 20.00
    left_logits = numpy.zeros((n, 4, 3)); thru_logits = numpy.zeros((n, 4, 3))
    frac_logits = numpy.log(numpy.clip(y["frac"], 1e-3, None))

    for a in range(4):
        ok = y["left"][:, a] != IGNORE
        left_logits[numpy.where(ok)[0], a, y["left"][ok, a]] = 20.00
        thru_logits[numpy.where(ok)[0], a, y["thru"][ok, a]] = 20.00

    return {
        "plan_logits": plan_logits,
        "left_logits": left_logits,
        "thru_logits": thru_logits,
        "frac_logits": frac_logits,
        "cycle": dataframe["cycle"].to_numpy(float)
    }

def _log_softmax(x):
    x = numpy.asarray(x, dtype=float)
    x = x - x.max(axis=-1, keepdims=True)
    return x - numpy.log(numpy.exp(x).sum(axis=-1, keepdims=True))

def _best_lanes(scenario: Scenario, left_logits, thru_logits):
    lp_l, lp_t = _log_softmax(left_logits), _log_softmax(thru_logits)
    left, thru, repaired = [0] * 4, [0]* 4, False

    for active in ACTIVE[scenario.layout]:
        best, best_lp = None, -numpy.inf
        for l in range(3):
            for t in range(3):
                if l + t + 1 <= scenario.lane_budget and lp_l[active, l] + lp_t[active, t] > best_lp:
                    best, best_lp = (l, t + 1), lp_l[active, l] + lp_l[active, t]

        if best != (int(numpy.argmax(lp_l[active])), int(numpy.argmax(lp_t[active])) + 1):
            repaired = True

        left[active], thru[active] = best

    return tuple(left), tuple(thru), repaired

def decode_design(scenario: Scenario, plan_logits, left_logits, thru_logits, frac_logits, cycle):
    left, thru, lanes_repaired = _best_lanes(scenario, left_logits, thru_logits)
    n_valid = N_PLANS[scenario.layout]
    order = [int(p) for p in numpy.argsort(-numpy.asarray(plan_logits)[:n_valid])]
    lost, mg = intergreen(scenario.speed_kmh), min_green(scenario)

    info = {
        "plan_rank":        0,
        "lanes_repaired":   lanes_repaired,     # True, if lane choice had to be edited (budget)
        "greens_repaired":  False,              # True, if greens had to be changed
        "legal":            True                # False, if none of the options worked (shouldn't happen)
    }

    for rank, plan in enumerate(order):
        n = n_phases(scenario.layout, plan)

        if rank == 0:
            frac = numpy.exp(_log_softmax(numpy.asarray(frac_logits, float)[:n]))
            cycle = float(numpy.clip(cycle, max(MIN_CYCLE, n * (mg + lost)), MAX_CYCLE))
            g = frac * (cycle - n * lost)

            if g.min() < mg - 1e-6:
                g = numpy.array(allocate_greens(frac, cycle, lost, mg))
                info["greens_repaired"] = True

            greens = tuple(float(x) for x in g)
        else:
            greens = cheap_greens(scenario, left, thru, plan)[1].greens

        design = Design(left, thru, plan, greens)

        if not check_design(scenario, design) and evaluate(scenario, design).feasible:
            info["plan_rank"] = rank
            return design, info

    info.update(plan_rank=len(order) - 1, legal=False)
    return Design(
        left,
        thru,
        order[-1],
        cheap_greens(
            scenario, 
            left, 
            thru,
            order[-1]
        )[1].greens
    ), info