"""

"""
# Imports
import argparse, pandas

from data.datasaver import save_dataframe
from data.generate import features_from_dataframe, row_to_scenario
from model.codec import decode_design
from sim.delay_model import evaluate

# Config
CONSIDER_FAST_AFTER = 60    # Km/h, what is considered fast
REPORT_VALUES = {
    1: ("First-choice plan unusable in",    "fallback"),
    2: ("Lanes changed to fit budget in",   "lanes_repaired"),
    3: ("Greens fixed in",                  "greens_repaired"),
    4: ("Final design illegal in",          "legal_ai")
}

# -- METHODS --
def evaluate_ai(dataframe: pandas.DataFrame, predictor) -> pandas.DataFrame:
    out = predictor(features_from_dataframe(dataframe))
    rows = []

    for index, (_, row) in enumerate(dataframe.iterrows()):
        scenario = row_to_scenario(row)
        design, info = decode_design(
            scenario,
            out["plan_logits"][index],
            out["left_logits"][index],
            out["thru_logits"][index],
            out["frac_logits"][index],
            float(out["cycle"][index])
        )

        result = evaluate(scenario, design)
        raw = evaluate(scenario, design, enforce_rules=False)
        rows.append({
            "delay_ai": raw.avarage_delay,
            "obj_ai": result.objective if result.feasible else raw.objective,
            "legal_ai": result.feasible,
            "plan_ai": design.plan,
            "fallback": info["plan_rank"] > 0,
            "lanes_repaired": info["lanes_repaired"],
            "greens_repaired": info["greens_repaired"]
        })

    return pandas.DataFrame(rows, index=dataframe.index)

def summarize(dataframe: pandas.DataFrame, ai: pandas.DataFrame) -> pandas.DataFrame:
    data = dataframe.join(ai)
    left_cols = [column for column in dataframe.columns if column.startswith("v_") and column.endswith("L")]
    heavy_left = data[left_cols].max(axis=1) >= 200
    fast = data["speed_kmh"] >= 60

    subsets = {
        "all":                              pandas.Series(True, index=data.index),
        "easy (no heavy lefts, < 60 km/h)": ~(heavy_left | fast),
        "heavy lefts (>= 200 veh/h)":       heavy_left,
        "fast_road (>= 60 km/h)":           fast,
        "heavy load (v/c >= 0.8)":          data["max_x_opt"] >= 0.80
    }

    rows = []
    for name, m in subsets.items():
        s = data[m]

        if len(s) == 0:
            continue

        rows.append({
            "subset":                   name,
            "n":                        len(s),
            "optimizer (s/veh)":        s["delay_opt"].mean(),
            "AI (s/veh)":               s["delay_ai"].mean(),
            "Webster (s/veh)":          s["delay_webster"].mean(),
            "fixed (s/veh)":            s["delay_fixed"].mean(),
            "AI regret vs opt (%)":     100 * ((s["obj_ai"] - s["objective_opt"]) / s["objective_opt"]).mean(),
            "AI within 10% of opt (%)": 100 * ((s["obj_ai"] <= 1.10 * s["objective_opt"])).mean(),
            "AI beats Webster (%)":     100 * (s["delay_ai"] < s["delay_webster"]).mean(),
            "Webster legal (%)":        100 * s["webster_legal"].mean(),
            "plan correct (%)":         100 * (s["plan_ai"] == s["plan"]).mean()
        })

    return pandas.DataFrame(rows).set_index("subset")

def report(dataframe: pandas.DataFrame, predictor, name: str = "AI", save_as: str | None = None) -> pandas.DataFrame:
    ai = evaluate_ai(dataframe, predictor)
    table = summarize(dataframe, ai)

    print(f"\n[REPORT] {name}: {len(dataframe)} held-out scenarios>")
    print(table.round(1).to_string())
    print("\n[REPAIR STATS]:")
    for index, (name, value_key) in REPORT_VALUES.items():
        print(f"\t{name}: {ai[value_key].mean():.1%}")

    if save_as:
        save_dataframe(table.round(2), save_as)
        #table.round(2).to_csv(save_to)
        #print(f"Saved to {save_to}")
        print(f"\nReport saved at 'output/{save_as}'")

    return table

if __name__ == "__main__":
    from model.model import TorchPredictor

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--csv", default="output/dataset.csv")
    arg_parser.add_argument("--ckpt", default="output/checkpoint.pt")
    arg_parser.add_argument("--out", default="result.csv")

    args = arg_parser.parse_args()
    data = pandas.read_csv(args.csv)
    pred = TorchPredictor.load(args.ckpt)
    report(
        data.iloc[pred.split["test"]], 
        pred, 
        "neural network", 
        args.out
    )