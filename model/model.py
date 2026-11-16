"""

"""
# Imports
import numpy, torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func

from model.codec import IGNORE
from sim.delay_model import MIN_CYCLE, MAX_CYCLE

# Config
N_IN = 16
LOSS_WEIGHTS = {
    "plan":     1.00,
    "left":     1.00,
    "thru":     1.00,
    "frac":     20.00,
    "cycle":    20.00,
    "delay":    5.00
}

# Datamodels
def multitask_loss(out: dict, y: dict, weights: dict = LOSS_WEIGHTS):
    terms = {
        "plan":     torch_nn_func.cross_entropy(out["plan"], y["plan"]),
        # For approaches that don't exist (such as north entrance/exit in 3leg intersections), carry label IGNORE
        "left":     torch_nn_func.cross_entropy(out["left"].reshape(-1, 3), y["left"].reshape(-1), ignore_index=IGNORE),
        "thru":     torch_nn_func.cross_entropy(out["thru"].reshape(-1, 3), y["thru"].reshape(-1), ignore_index=IGNORE),
        "cycle":    torch_nn_func.mse_loss(out["cycle"], y["cycle"]),
        "delay":    torch_nn_func.mse_loss(out["delay"], y["delay"])
    }

    mask = torch.arange(4, device=y["n_ph"].device)[None, :] < y["n_ph"][:, None]
    p = torch.softmax(out["frac"].masked_fill(~mask, -1e9), dim=-1)
    terms["frac"] = (((p - y["frac"]) ** 2) * mask).sum() / mask.sum()
    total = sum(weights[k] * v for k, v in terms.items())

    return total, {k: float(v) for k, v in terms.items()}

class IntersectionNet(torch_nn.Module):
    def __init__(self, n_in: int = N_IN, hidden: int = 256, dropout: float = 0.05):
        super().__init__()

        self.trunk = torch_nn.Sequential(
            torch_nn.Linear(n_in, hidden),          torch_nn.ReLU(), torch_nn.Dropout(dropout),
            torch_nn.Linear(hidden, hidden),        torch_nn.ReLU(), torch_nn.Dropout(dropout),
            torch_nn.Linear(hidden, hidden // 2),   torch_nn.ReLU()
        )

        hidden_floored = hidden // 2
        self.plan   = torch_nn.Linear(hidden_floored, 5)
        self.left   = torch_nn.Linear(hidden_floored, 4 * 3)
        self.thru   = torch_nn.Linear(hidden_floored, 4 * 3)
        self.frac   = torch_nn.Linear(hidden_floored, 4)
        self.cycle  = torch_nn.Linear(hidden_floored, 1)
        self.delay  = torch_nn.Linear(hidden_floored, 1)

    def forward(self, x):
        z = self.trunk(x)

        return {
            "plan":     self.plan(z),
            "left":     self.left(z).view(-1, 4, 3),
            "thru":     self.thru(z).view(-1, 4, 3),
            "frac":     self.frac(z),
            "cycle":    self.cycle(z).squeeze(-1),
            "delay":    self.delay(z).squeeze(-1)
        }

class TorchPredictor:
    def __init__(self, net: IntersectionNet, device: str = "cpu"):
        self.net, self.device = net.to(device).eval(), device

    @classmethod
    def load(cls, path: str, device: str = "cpu"):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        net = IntersectionNet(**checkpoint["config"])
        net.load_state_dict(checkpoint["model"])
        predictor = cls(net, device)
        predictor.split = checkpoint.get("split")

        return predictor

    @torch.no_grad()
    def __call__(self, x: numpy.ndarray) -> dict:
        out = self.net(torch.as_tensor(x, dtype=torch.float32, device=self.device))

        return {
            "plan_logits":      torch.log_softmax(out["plan"], -1).cpu().numpy(),
            "left_logits":      out["left"].cpu().numpy(),
            "thru_logits":      out["thru"].cpu().numpy(),
            "frac_logits":      out["frac"].cpu().numpy(),
            "cycle":            (MIN_CYCLE + (MAX_CYCLE - MIN_CYCLE) * out["cycle"]).cpu().numpy(),
            "delay_pred":       torch.expm1(out["delay"] * 4.00).clamp(min=0).cpu().numpy(),            # s/veh
        }