"""
Generates random data for viewing different scenarios.

Run (from root):
    python -m data.generate --n 20000 --workers 8 --out data/dataset.csv

"""
# Imports
import argparse, time, numpy, pandas

from multiprocessing import Pool
from tqdm import tqdm

from services.config import SETTINGS
from data.datasaver import save_dataframe
from data.optimize import optimize
from sim.baselines import fixed_design, webster_design
from sim.delay_model import (
    ACTIVE, APPROACHES, SAT_FLOW, TURNS, VALID_MOVES, 
    Scenario, Result, Design, 
    evaluate
)

# Config
FLOW_COLS = [f"v_{A}{T}" for A in APPROACHES for T in TURNS]
FEATURE_COLS = ["is_4leg", "speed", "lane_budget", "pedestrian"] + FLOW_COLS

RESULT_FILENAME         = "dataset.csv"

# -- METHODS --
def _demand_index(flows, layout, budget):
    total       = flows.sum(axis=1)
    capacity    = SAT_FLOW * budget * 0.90
    if layout == "3leg":
        return (max(total[0], total[2]) + max(total[1], total[3])) / capacity
    return (max(total[1], total[3]) + total[2]) / capacity

def random_scenario(rng: numpy.random.Generator) -> Scenario:
    while True:
        layout      = str(rng.choice(["3leg", "4leg"]))
        hour        = rng.uniform(0, 24)
        f_hour      = float(numpy.interp(hour, numpy.arange(25), numpy.append(SETTINGS.data_gen_settings.diurnal, SETTINGS.data_gen_settings.diurnal[0])))
        speed       = float(rng.choice(SETTINGS.traffic_settings.speeds, p=SETTINGS.traffic_settings.speed_probs))
        lane_budget = int(rng.choice([2, 3, 4], p=[0.30, 0.40, 0.30]))
        pedestrian  = bool(rng.random() < SETTINGS.data_gen_settings.pedestrian_cross_prob)

        peak_load   = rng.uniform(0.70, 1.40)
        minor_ratio = rng.uniform(0.15, 0.80)
        main_axis   = "EW" if layout == "3leg" else str(rng.choice(["NS", "EW"]))
        heavy_left  = rng.random() < SETTINGS.data_gen_settings.heavy_left_prob
        alpha       = (7, 9, 2) if heavy_left else (2, 14, 2)                           # Dirichlet distribution; On heavy left turns, model prefers left (ofc) otherwise prefers going straigh ahead.

        mask        = VALID_MOVES[layout]
        flows       = numpy.zeros((4, 3))
        for active in ACTIVE[layout]:
            on_main         = active in ((1, 3) if main_axis == "EW" else (0, 2))
            volume          = (1.00 if on_main else minor_ratio)
            volume          *= rng.lognormal(0.00, 0.25) * rng.uniform(0.60, 1.00)      # Noise + Direction imbalance
            share           = numpy.where(mask[active], rng.dirichlet(alpha), 0.00)
            flows[active]   = volume * share / share.sum()

        if rng.random() < SETTINGS.data_gen_settings.burst_traffic_prob:
            flows[int(rng.choice(ACTIVE[layout]))] *= rng.uniform(1.5, 2.5)

        # Scale load so that it follows natural load (daytime has more traffic unlike night time)
        flows *= peak_load * f_hour / _demand_index(flows, layout, lane_budget)
        flows = numpy.round(flows / 10.00) * 10.00
        if flows.sum() >= 100:
            return Scenario(layout, flows, speed, lane_budget, pedestrian)

def scenario_features(scenario: Scenario) -> numpy.ndarray:
    return numpy.concatenate([
        [float(scenario.layout == "4leg"), scenario.speed_kmh / 100.00, scenario.lane_budget / 4.00, float(scenario.pedestrian)],
        scenario.flows.reshape(-1) / 1000.00
    ]).astype(numpy.float32)

def features_from_dataframe(dataframe: pandas.DataFrame) -> numpy.ndarray:
    x = numpy.zeros((len(dataframe), len(FEATURE_COLS)), dtype=numpy.float32)
    x[:, 0]     = (dataframe["layout"] == "4leg").to_numpy()
    x[:, 1]     = dataframe["speed_kmh"].to_numpy() / 100.00
    x[:, 2]     = dataframe["lane_budget"].to_numpy() / 4.00
    x[:, 3]     = dataframe["pedestrian"].to_numpy()
    x[:, 4:]    = dataframe[FLOW_COLS].to_numpy() / 1000.00
    return x

def row_to_scenario(row) -> Scenario:
    # Converts one row from .csv to Scenario object
    flows = numpy.array([row[column] for column in FLOW_COLS], dtype=float).reshape(4, 3)
    return Scenario(
        str(row["layout"]), 
        flows, 
        float(row["speed_kmh"]),
        int(row["lane_budget"]),
        bool(row["pedestrian"])
    )

def _make_row(scenario: Scenario, design: Design, result: Result):
    wd, fd                  = webster_design(scenario), fixed_design(scenario)
    web, fix                = evaluate(scenario, wd, enforce_rules=False), evaluate(scenario, fd, enforce_rules=False)
    web_legal, fix_legal    = evaluate(scenario, wd).feasible, evaluate(scenario, fd).feasible
    row = {
        "layout":       scenario.layout,
        "speed_kmh":    scenario.speed_kmh,
        "lane_budget":  scenario.lane_budget,
        "pedestrian":   int(scenario.pedestrian)
    }

    for a, A in enumerate(APPROACHES):
        for t, T in enumerate(TURNS):
            row[f"v_{A}{T}"] = int(scenario.flows[a, t])
    row["plan"] = design.plan

    for a, A in enumerate(APPROACHES):
        row[f"left_{A}"] = design.left_lanes[a]
    for a, A in enumerate(APPROACHES):
        row[f"thru_{A}"] = design.thru_lanes[a]
    for p in range(4):
        row[f"green_{p}"] = round(design.greens[p], 2) if p < len(design.greens) else 0.00
    row.update(
        cycle           = round(result.cycle, 2), 
        delay_opt       = round(result.avarage_delay, 3),
        objective_opt   = round(result.objective, 3),
        max_x_opt       = round(result.max_x, 3),
        los_opt         = result.los,
        total_lanes_opt = result.total_lanes,
        delay_webster   = round(web.avarage_delay, 3),
        delay_fixed     = round(fix.avarage_delay, 3),
        max_x_webster   = round(web.max_x, 3),
        webster_legal   = int(web_legal),
        fixed_legal     = int(fix_legal)
    )

    return row

def _work(args):
    index, seed     = args
    rng             = numpy.random.default_rng([seed, index])
    scenario        = random_scenario(rng)
    design, result  = optimize(scenario)

    if not result.feasible or result.max_x > SETTINGS.data_gen_settings.max_x_accept:
        return None
    return _make_row(scenario, design, result)

def build_dataset(n, seed=0, workers=None, out=RESULT_FILENAME, save_every=2000):
    rows, tried, t0 = [], 0, time.time()
    pool_size = int(n * 3) + 100

    with Pool(workers) as pool, tqdm(total=n, desc="Building Dataset") as pbar:
        for row in pool.imap(_work, ((index, seed) for index in range(pool_size)), chunksize=4):
            tried += 1

            if row is not None:
                rows.append(row)
                pbar.update(1)
                pbar.set_postfix(
                    tried = tried,
                    success_rate = f"{len(rows) / tried:.0%}"
                )

                if len(rows) % save_every == 0:
                    save_dataframe(pandas.DataFrame(rows), RESULT_FILENAME, "output", index=False)
                    #pandas.DataFrame(rows).to_csv(out, index=False)

            if len(rows) >= n:
                break

    dataframe = pandas.DataFrame(rows)
    save_dataframe(dataframe, RESULT_FILENAME, "output", index=False)
    print(f"\nSaved {len(dataframe)} scenarios to {out} ({len(dataframe) / max(tried, 1):.0%} of generated scenarios were servable) in {time.time() - t0:.0f}s")

    if len(dataframe) < n:
        print("Warning: fewer than requested; lower the demand range in random_scenario()")

    return dataframe

# -- MAIN --
if __name__ == "__main__":
    arg_parser  = argparse.ArgumentParser()
    arg_parser.add_argument("--n",          type=int,   default=200)
    arg_parser.add_argument("--seed",       type=int,   default=0)
    arg_parser.add_argument("--workers",    type=int,   default=None)
    arg_parser.add_argument("--out",                    default=RESULT_FILENAME)

    args    = arg_parser.parse_args()
    build_dataset(
        args.n, 
        args.seed, 
        args.workers, 
        args.out
    )