"""

"""
# Imports
import argparse, os, pickle, numpy, pandas

from sklearn.neural_network import MLPClassifier, MLPRegressor

from data.generate import features_from_dataframe
from eval.compare import report
from model.codec import IGNORE, encode_labels, split_indices
from sim.delay_model import MIN_CYCLE, MAX_CYCLE

# Datamodels
def _log_prob(clf, x, n_classes):
    out = numpy.full((len(x), n_classes), numpy.log(1e-6))
    out[:, clf.classes_] = numpy.log(numpy.clip(clf.predict_proba(x), 1e-6, 1.00))
    return out

class SklearnPredictor:
    def __init__(self, hidden=(128, 128), max_iter=300, seed=0):
        kw = dict(
            hidden_layer_sizes  = hidden,
            max_iter            = max_iter,
            random_state        = seed,
            early_stopping      = True
        )

        self.kw = kw
        self.plan = MLPClassifier(**kw)
        self.lanes = {(h, a): MLPClassifier(**kw) for h in ("left", "thru") for a in range(4)}
        self.timing = MLPRegressor(**kw)

    def fit(self, x, y):
        self.plan.fit(x, y["plan"])

        for (h, a), clf in self.lanes.items():
            ok = y[h][:, a] != IGNORE
            clf.fit(x[ok], y[h][ok, a])

        self.timing.fit(x, numpy.column_stack([y["cycle"], y["frac"]]))
        return self

    def __call__(self, x):
        n = len(x)
        out = {
            "plan_logits": _log_prob(self.plan, x, 5),
            "left_logits": numpy.zeros((n, 4, 3)),
            "thru_logits": numpy.zeros((n, 4, 3))
        }

        for (h, a), clf in self.lanes.items():
            out[f"{h}_logits"][:, a, :] = _log_prob(clf, x, 3)

        t = self.timing.predict(x)
        out["cycle"] = MIN_CYCLE + (MAX_CYCLE - MIN_CYCLE) * t[:, 0]
        out["frac_logits"] = numpy.log(numpy.clip(t[:, 1:], 1e-3, None))
        return out

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--csv", default="output/dataset.csv")
    arg_parser.add_argument("--seed", type=int, default=0)
    arg_parser.add_argument("--save", default="output/sklearn_model.pkl", help="GUI looks for the model from here")

    args = arg_parser.parse_args()
    dataframe = pandas.read_csv(args.csv)
    split = split_indices(len(dataframe), args.seed)
    train = dataframe.iloc[split["train"]]

    import model.sklearn_baseline as importable
    model = importable.SklearnPredictor(seed=args.seed).fit(features_from_dataframe(train), encode_labels(train))
    report(dataframe.iloc[split["test"]], model, "scikit-learn MLPs", "results_sklearn.csv")
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)

    with open(args.save, "wb") as f:
        pickle.dump(model, f)

    print(f"Saved to {args.save}")