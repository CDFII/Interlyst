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
                # TODO: COMPLETE THIS CRAP