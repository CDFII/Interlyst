"""

"""
# Imports
import argparse, time, numpy, pandas, torch

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from services.config import SETTINGS
from data.datasaver import save_dataframe
from data.generate import features_from_dataframe
from eval.compare import report
from model.codec import encode_labels, split_indices
from model.model import IntersectionNet, TorchPredictor, multitask_loss

# Config
KEYS = ["plan", "left", "thru", "n_ph", "frac", "cycle", "delay"]

# -- METHODS --
def to_tensors(dataframe, device):
    x = torch.as_tensor(features_from_dataframe(dataframe), device=device)
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
    arg_parser.add_argument("--epochs", type=int,   default=SETTINGS.ai_settings.epochs)
    arg_parser.add_argument("--batch",  type=int,   default=SETTINGS.ai_settings.batch)
    arg_parser.add_argument("--lr",     type=float, default=SETTINGS.ai_settings.lr)
    arg_parser.add_argument("--hidden", type=int,   default=SETTINGS.ai_settings.hidden)
    arg_parser.add_argument("--seed",   type=int,   default=SETTINGS.ai_settings.seed)
    arg_parser.add_argument("--csv",                default="output/dataset.csv")
    arg_parser.add_argument("--out",                default="output/checkpoint.pt")

    args = arg_parser.parse_args()
    config = {
        "hidden": args.hidden
    }

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataframe = pandas.read_csv(args.csv)
    split = split_indices(len(dataframe), args.seed)

    # Output some data prior training begins
    split_headers = ("train", "value", "test")
    arg_headers = ("epochs", "batch", "lr", "hidden", "seed")

    print(f"\nDevice: {device}\nScenarios: {len(dataframe)}, overview:")
    for index, header in enumerate(split_headers):      # Print training data splits
        print(f" | {header}: {len(split[header])}")
    print(f"Training settings:")
    for index, header in enumerate(arg_headers):        # Print training arguments
        print(f" | {header}: {getattr(args, header)}")

    data = {k: to_tensors(dataframe.iloc[index], device) for k, index in split.items() if k != "test"}
    loaders = {
        "train": DataLoader(
            TensorDataset(data["train"][0], *[data["train"][1][k] for k in KEYS]),
            batch_size = args.batch,
            shuffle = True
        ),
        "value": DataLoader(
            TensorDataset(data["value"][0], *[data["value"][1][k] for k in KEYS]),
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

    pbar = tqdm(range(1, args.epochs + 1), desc="Training... 0_o")
    best, best_state, history, t0 = float("inf"), None, [], time.time()

    for epoch in pbar:
        tr, _ = run_epoch(net, loaders["train"], opt, sched)
        va, parts = run_epoch(net, loaders["value"])
        history.append({
            "epoch":        epoch,
            "train_loss":   tr,
            "val_loss":     va,
            **{f"val_{k}": v for k, v in parts.items()}
        })

        if va < best:
            best, best_state = va, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

        elapsed = time.time() - t0
        pbar.set_postfix(
            train   = f"{tr:.4f}",
            val     = f"{va:.4f}",
            plan    = f"{parts['plan']:.3f}",
            left    = f"{parts['left']:.3f}",
            thru    = f"{parts['thru']:.3f}"
        )

    net.load_state_dict(best_state)
    torch.save({
        "model": best_state,
        "config": config,
        "split": {k: v.tolist() for k, v in split.items()}
    }, args.out)
    save_dataframe(pandas.DataFrame(history), "history.csv", "output", index=False)
    print(f"Saved to {args.out} (best val loss {best:.4f} and output/history.csv)")

    report(
        dataframe.iloc[split["test"]],
        TorchPredictor(net, device),
        "neural network",
        "results_torch.csv"
    )