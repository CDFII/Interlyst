"""

"""
# Imports
import argparse, time, numpy, pandas, torch

from torch.utils.data import DataLoader, TensorDataset

from data.generate import features_from_dataframe
from eval.compare import report
from model.codec import encode_labels, split_indices
from model.model import IntersectionNet, TorchPredictor, multitask_loss

# Config
KEYS = ["plan", "left", "thru", "n_ph", "frac", "cycle", "delay"]

# -- METHODS --
def to_tensors(dataframe, device):
    x = torch.as_tensor(features_from_dataframe(dataframe), deivce=device)
    y = encode_labels(dataframe)
    return x, {k: torch.as_tensor(y[k], device=device) for k in KEYS}

def run_epoch(net, loader, opt=None, sched=None):
    train = opt is not None
    net.train(train)
    total, n, parts = 0.00, 0, {}

    with torch.set_grad_enabled(train):
        for batch in loader:
            x, y = batch[0], dict(zip(KEYS, batch[1:]))
            loss, terms = multitask_loss(net(x), y)

            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.00)
                opt.step()
                sched.step()

            b = len(x)
            total += float(loss) * b
            n += b

            for k, v in terms.items():
                parts[k] = parts.get(k, 0.00) + v * b

    return total / n, {k: v / n for k, v in parts.items()}

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--csv", default="data/dataset.csv")
    arg_parser.add_argument("--epochs", type=int, default=150)
    arg_parser.add_argument("--batch", type=int, default=256)
    arg_parser.add_argument("--lr", type=float, default=2e-3)
    arg_parser.add_argument("--hidden", type=int, default=256)
    arg_parser.add_argument("--seed", type=int, default=0)
    arg_parser.add_argument("--out", default="model/checkpoint.pt")

    args = arg_parser.parse_args()
    config = {
        "hidden": args.hidden
    }

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataframe = pandas.read_csv(args.csv)
    split = split_indices(len(dataframe), args.seed)
    print(f"\t{len(dataframe)} scenarios: {len(split["train"])} train\n\t{len(split["val"])} val\n\t{len(split["test"])} test\n\tDevice: {device}")

    data = {k: to_tensors(dataframe.iloc[index], device) for k, index in split.items() if k != "test"}
    loaders = {
        "train": DataLoader(
            TensorDataset(data["train"][0], *[data["train"][1][k] for k in KEYS]),
            batch_size = args.batch,
            shuffle = True
        ),
        "val": DataLoader(
            TensorDataset(data["val"][0], *[data["val"][1][k] for k in KEYS]),
            batch_size=1024
        )
    }

    net = IntersectionNet(**config).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr = args.lr,
        total_steps = args.epochs * len(loaders["train"])
    )

    best, best_state, history, t0 = float("inf"), None, [], time.time()
    for epoch in range(1, args.epochs + 1):
        tr, _ = run_epoch(net, loaders["train"], opt, sched)
        va, parts = run_epoch(net, loaders["val"])
        history.append({
            "epoch": epoch,
            "train_loss": tr,
            "val_loss": va,
            **{f"val_{k}": v for k, v in parts.items()}
        })

        if va < best:
            best, best_state = va, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch: {epoch:4d} | train: {tr:.4f} | val: {va:.4f}\n(plan: {parts["plan"]:.3f} | left: {parts["left"]:.3f} | thru: {parts["thru"]:.3f})\n{time.time() - t0:.0f}s", flush=True)

    net.load_state_dict(best_state)
    torch.save({
        "model": best_state,
        "config": config,
        "split": {k: v.tolist() for k, v in split.items()}
    }, args.out)
    pandas.DataFrame(history).to_csv("model/history.csv", index=False)
    print(f"Saved to {args.out} (best val loss {best:.4f} and model/history.csv)")

    report(
        dataframe.iloc[split["test"]],
        TorchPredictor(net, device),
        "neural network",
        "eval/results.csv"
    )